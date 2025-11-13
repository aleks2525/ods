#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================
# 🚨 Отключаем многопоточность BLAS ДО импортов
# =========================
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import argparse
import random
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
import heapq
import tempfile
import gc
import time
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files
# ML
import xgboost as xgb
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import faiss
import faiss.contrib.torch_utils  # not needed, but ensures GPU init
# =========================
def info(msg: str):
    print(f"[INFO] {msg}", flush=True)
def warn(msg: str):
    print(f"[WARN] {msg}", flush=True)
# =========================
# Конфигурация
# =========================
REPO_ID = "deepvk/VK-LSVD"
REPO_TYPE = "dataset"
DATA_ROOT = "VK-LSVD"
DEFAULT_SUBSAMPLE = "up-0.9_ip-0.9"
OUTPUT_PATH = "submission_l4.parquet"
ARROW_SCHEMA = pa.schema([
    pa.field("item_id", pa.uint32()),
    pa.field("user_id", pa.list_(pa.uint32()))
])
CANDIDATES_PER_ITEM = 600  # ↓ для ускорения
USERS_PER_ITEM = 100
MAX_ASSIGN_PER_USER = 100
EMB_DIM = 64
SIM_TOPK = 600
DEFAULT_WEIGHTS = {
    "W_ALS": 8.0,
    "W_SIMHITS": 4.0,
    "W_AUTHOR": 2.0,
    "W_POPULARITY": 0.1,
    "W_TIME_DECAY": 1.0,
    "W_COOC": 1.5,
    "W_AUTHOR_FAN_RANK": 0.8,
}
EXPLICIT_ACTIONS = ["like", "share", "bookmark", "click_on_author"]

# =========================
# Утилиты (без изменений)
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
def ensure_download(filename: str) -> str:
    local_path = os.path.join(DATA_ROOT, filename)
    ensure_dir(os.path.dirname(local_path))
    if os.path.exists(local_path):
        return local_path
    info(f"Скачиваем: {filename}")
    hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=filename, local_dir=DATA_ROOT)
    return local_path
def auto_pick_train_weeks(subsample_name: str) -> List[str]:
    info(f"Авто-определяем недели train для {subsample_name}...")
    files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    prefix = f"subsamples/{subsample_name}/train/"
    weeks = sorted({os.path.basename(f).replace(".parquet", "") for f in files if f.startswith(prefix) and f.endswith(".parquet")})
    if not weeks:
        alt = subsample_name.replace("0.9", "-0.9")
        alt_weeks = sorted({os.path.basename(f).replace(".parquet", "") for f in files if f.startswith(f"subsamples/{alt}/train/") and f.endswith(".parquet")})
        if alt_weeks:
            warn(f"Используем альтернативный subsample: {alt}")
            return alt_weeks
        else:
            raise RuntimeError("Не найдены train-недели")
    return weeks
def ensure_dataset_files(subsample_name: str, weeks: Optional[List[str]], val_week: Optional[str] = None):
    files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    if weeks is None:
        weeks = auto_pick_train_weeks(subsample_name)
    for w in weeks:
        ensure_download(f"subsamples/{subsample_name}/train/{w}.parquet")
    path_users_meta = ensure_download("metadata/users_metadata.parquet")
    path_items_meta = ensure_download("metadata/items_metadata.parquet")
    path_item_embs = ensure_download("metadata/item_embeddings.npz")
    example_candidates = [f for f in files if "example" in os.path.basename(f).lower() and f.endswith(".parquet")]
    example_path = ensure_download(example_candidates[0]) if example_candidates else os.path.join(DATA_ROOT, "example.parquet")
    val_path = None
    if val_week:
        try:
            val_path = ensure_download(f"subsamples/{subsample_name}/validation/{val_week}.parquet")
        except:
            pass
    return weeks, path_items_meta, path_users_meta, path_item_embs, example_path, val_path

# =========================
# Загрузка метаданных
# =========================
def load_items_metadata(path_items_meta: str) -> Tuple[Dict[int, int], pl.DataFrame]:
    df = pl.read_parquet(path_items_meta).select(["item_id", "author_id"])
    item_to_author = {int(r[0]): int(r[1]) for r in df.iter_rows()}
    return item_to_author, df
def load_embeddings(path_npz: str, dim: int):
    data = np.load(path_npz)
    item_ids = data["item_id"].astype(np.int64)
    embs = data["embedding"][:, :dim].astype(np.float32)
    id2idx = {int(i): idx for idx, i in enumerate(item_ids)}
    return item_ids, embs, id2idx

# =========================
# Scoring functions
# =========================
def als_score(uid: int, iid: int, user_factors, item_factors, user_idx_map, item_idx_map) -> float:
    if uid not in user_idx_map or iid not in item_idx_map:
        return 0.0
    uix = user_idx_map[uid]
    iix = item_idx_map[iid]
    return float(user_factors[uix].dot(item_factors[iix]))

def ibcf_score(uid: int, iid: int, similar_items: List[Tuple[int, float]], item_to_users: Dict[int, Set[int]]) -> float:
    score = 0.0
    for co_iid, sim in similar_items:
        if uid in item_to_users.get(co_iid, set()):
            score += sim
    return score

# =========================
# FAISS-GPU для похожих
# =========================
def build_faiss_index(item_vecs: np.ndarray, item_ids: np.ndarray):
    info("Строим FAISS-GPU index...")
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(EMB_DIM)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index.add(item_vecs)
    return gpu_index, item_ids

def get_similar_items_faiss(iid: int, item_id_to_index: Dict[int, int], gpu_index, item_ids, topk: int) -> List[Tuple[int, float]]:
    idx = item_id_to_index.get(iid)
    if idx is None:
        return []
    query_vec = item_vecs[idx:idx+1]
    D, I = gpu_index.search(query_vec, topk + 1)
    # skip self
    similar = []
    for d, i in zip(D[0][1:], I[0][1:]):
        similar.append((int(item_ids[i]), float(d)))
    return similar

# =========================
# Targeted co-occurrence
# =========================
def build_cooccurrence_targeted(
    subsample_name: str,
    weeks: List[str],
    example_items: Set[int],
    min_cooc: int = 2,
    chunk_size: int = 500_000
) -> Dict[int, Dict[int, int]]:
    info(f"Строим targeted co-occurrence для {len(example_items)} items...")
    example_items_set = set(example_items)
    cooc_counter = defaultdict(lambda: defaultdict(int))
    for w in weeks:
        p = os.path.join(DATA_ROOT, "subsamples", subsample_name, "train", f"{w}.parquet")
        lazy_df = (
            pl.scan_parquet(p)
            .select(["user_id", "item_id"])
            .filter(pl.col("item_id").is_in(example_items_set))
        )
        for chunk in lazy_df.collect(streaming=True).iter_slices(chunk_size):
            user_items = chunk.group_by("user_id").agg(pl.col("item_id").alias("items"))
            for row in user_items.iter_rows(named=True):
                items = set(row["items"])
                if len(items) < 2:
                    continue
                for e in items:
                    if e not in example_items_set:
                        continue
                    for c in items:
                        if c != e:
                            cooc_counter[e][c] += 1
            del chunk, user_items
            gc.collect()
    filtered_cooc = {}
    for e, neighbors in cooc_counter.items():
        filtered = {c: cnt for c, cnt in neighbors.items() if cnt >= min_cooc}
        if filtered:
            filtered_cooc[e] = filtered
    info(f"Targeted co-occurrence: {len(filtered_cooc)} items")
    return filtered_cooc

# =========================
# Фичи
# =========================
def extract_features(
    uid: int,
    iid: int,
    similar_items: List[Tuple[int, float]],
    item_to_users: Dict[int, Set[int]],
    item_to_author: Dict[int, int],
    author_affinity: Dict[int, Dict[int, float]],
    item_popularity: Dict[int, int],
    user_popularity: Dict[int, int],
    user_factors, item_factors,
    user_idx_map, item_idx_map,
    week_decay_map: Dict[Tuple[int, int], float],
    weights: Dict[str, float],
    cooc_map: Dict[int, Dict[int, int]],
    author_to_sorted_fans: Dict[int, List[Tuple[int, float]]],
) -> List[float]:
    features = []
    als_sc = als_score(uid, iid, user_factors, item_factors, user_idx_map, item_idx_map)
    features.append(als_sc * weights["W_ALS"])
    ibcf_sc = ibcf_score(uid, iid, similar_items, item_to_users)
    features.append(ibcf_sc * weights["W_SIMHITS"])
    author = item_to_author.get(iid)
    author_sc = author_affinity.get(uid, {}).get(author, 0.0) if author is not None else 0.0
    features.append(math.sqrt(author_sc + 1e-9) * weights["W_AUTHOR"])
    pop_sc = math.log(1 + item_popularity.get(iid, 0)) * weights["W_POPULARITY"]
    features.append(pop_sc)
    user_pop_sc = math.log(1 + user_popularity.get(uid, 0)) * 0.01
    features.append(user_pop_sc)
    week_weight = week_decay_map.get((uid, iid), 1.0)
    features.append(week_weight * weights["W_TIME_DECAY"])
    n_sim = sum(1 for si, _ in similar_items if uid in item_to_users.get(si, set()))
    features.append(n_sim * 0.1)
    cooc_sc = 0.0
    if iid in cooc_map:
        for co_iid, cnt in cooc_map[iid].items():
            if uid in item_to_users.get(co_iid, set()):
                cooc_sc += cnt
    features.append(cooc_sc * weights["W_COOC"])
    author_fan_rank_norm = 1.0
    if author is not None and author in author_to_sorted_fans:
        fan_list = author_to_sorted_fans[author]
        uids = [u for u, _ in fan_list]
        if uid in uids:
            rank = uids.index(uid)
            author_fan_rank_norm = rank / len(fan_list)
    features.append((1.0 - author_fan_rank_norm) * weights["W_AUTHOR_FAN_RANK"])
    return features

# =========================
# Подготовка данных (без Optuna)
# =========================
def prepare_data(
    example_items: List[int],
    candidate_users_per_item: Dict[int, List[int]],
    item_to_author: Dict[int, int],
    author_affinity: Dict[int, Dict[int, float]],
    item_popularity: Dict[int, int],
    user_popularity: Dict[int, int],
    user_factors, item_factors,
    user_idx_map, item_idx_map,
    similar_cache: Dict[int, List[Tuple[int, float]]],
    item_to_users: Dict[int, Set[int]],
    week_decay_map: Dict[Tuple[int, int], float],
    cooc_map: Dict[int, Dict[int, int]],
    author_to_sorted_fans: Dict[int, List[Tuple[int, float]]],
    true_relevant_users: Dict[int, Set[int]],
    weights: Dict[str, float]
):
    X, y, groups = [], [], []
    for iid in tqdm(example_items, desc="Подготовка данных"):
        candidates = candidate_users_per_item[iid]
        group_size = 0
        for uid in candidates:
            features = extract_features(
                uid, iid, similar_cache.get(iid, []),
                item_to_users, item_to_author, author_affinity,
                item_popularity, user_popularity,
                user_factors, item_factors,
                user_idx_map, item_idx_map,
                week_decay_map, weights, cooc_map, author_to_sorted_fans
            )
            X.append(features)
            rel = 1 if uid in true_relevant_users.get(iid, set()) else 0
            y.append(rel)
            group_size += 1
        groups.append(group_size)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(groups, dtype=np.uint32)

# =========================
# Постпроцессинг (без изменений)
# =========================
def apply_smart_assignment_limit(
    item_to_scored_users: Dict[int, List[Tuple[float, int]]],
    users_per_item: int,
    max_assign_per_user: int,
    popular_users: List[int],
    item_candidates_extended: Dict[int, List[Tuple[float, int]]],
    item_to_users_gt: Optional[Dict[int, Set[int]]] = None
) -> Dict[int, List[int]]:
    assignments = {}
    user_counts = defaultdict(int)
    for iid, scored in item_to_scored_users.items():
        top_users = [uid for _, uid in scored[:users_per_item]]
        assignments[iid] = top_users
        for uid in top_users:
            user_counts[uid] += 1
    overloaded_users = {u for u, cnt in user_counts.items() if cnt > max_assign_per_user}
    if not overloaded_users:
        return assignments
    user_item_heap = defaultdict(list)
    for iid, user_list in assignments.items():
        uid_to_score = {uid: score for score, uid in item_to_scored_users[iid]}
        for uid in user_list:
            if uid in overloaded_users:
                score = uid_to_score.get(uid, 0.0)
                heapq.heappush(user_item_heap[uid], (score, iid))
    for uid in overloaded_users:
        excess = user_counts[uid] - max_assign_per_user
        to_remove = heapq.nsmallest(excess, user_item_heap[uid])
        for _, iid in to_remove:
            current_list = [u for u in assignments[iid] if u != uid]
            user_counts[uid] -= 1
            best_replacement = None
            extended = item_candidates_extended.get(iid, [])
            all_candidates = [(0.0, u) for u in popular_users[:1_000_000]] + extended
            for _, candidate_uid in all_candidates:
                if user_counts[candidate_uid] >= max_assign_per_user:
                    continue
                if item_to_users_gt and candidate_uid in item_to_users_gt.get(iid, set()):
                    best_replacement = candidate_uid
                    break
                if best_replacement is None:
                    best_replacement = candidate_uid
            if best_replacement is not None:
                current_list.append(best_replacement)
                user_counts[best_replacement] += 1
            else:
                for candidate_uid in popular_users[:1_000_000]:
                    if user_counts[candidate_uid] < max_assign_per_user:
                        current_list.append(candidate_uid)
                        user_counts[candidate_uid] += 1
                        break
            assignments[iid] = current_list[:users_per_item]
    return assignments

# =========================
# Основной пайплайн
# =========================
def run_pipeline(subsample_name: str, weeks: Optional[List[str]], output_path: str, val_week: str = "week_25"):
    weeks, path_items_meta, _, path_item_embs, example_path, val_path = ensure_dataset_files(subsample_name, weeks, val_week)
    info(f"Недели train: {weeks}")
    item_to_author, _ = load_items_metadata(path_items_meta)
    item_ids, item_vecs, item_id_to_index = load_embeddings(path_item_embs, EMB_DIM)
    recent_weeks = weeks[-5:] if len(weeks) > 5 else weeks
    info(f"Обучаемся на {len(recent_weeks)} неделях")
    user_to_items = defaultdict(set)
    item_to_users = defaultdict(set)
    user_total_interactions = defaultdict(int)
    author_affinity = defaultdict(lambda: defaultdict(float))
    for w in recent_weeks:
        p = os.path.join(DATA_ROOT, "subsamples", subsample_name, "train", f"{w}.parquet")
        cols = ["user_id", "item_id"] + EXPLICIT_ACTIONS
        df_week = pl.read_parquet(p, columns=cols)
        df_week = df_week.filter(pl.any_horizontal([pl.col(a) for a in EXPLICIT_ACTIONS]))
        for row in df_week.iter_rows(named=True):
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            user_to_items[uid].add(iid)
            item_to_users[iid].add(uid)
            user_total_interactions[uid] += 1
            author = item_to_author.get(iid)
            if author is not None:
                wgt = sum(1.0 for action in EXPLICIT_ACTIONS if bool(row[action]))
                if wgt > 0:
                    author_affinity[uid][author] += wgt
        del df_week
        gc.collect()
    info("✅ Индексы построены.")
    popular_users = sorted(user_total_interactions.keys(), key=lambda u: -user_total_interactions[u])
    item_popularity = {iid: len(users) for iid, users in item_to_users.items()}
    user_popularity = user_total_interactions
    info("Обучаем ALS на GPU...")
    all_users = sorted(user_to_items.keys())
    all_items = sorted(item_to_users.keys())
    user_idx_map = {u: i for i, u in enumerate(all_users)}
    item_idx_map = {it: i for i, it in enumerate(all_items)}
    rows, cols, data = [], [], []
    for uid, items in user_to_items.items():
        for iid in items:
            if uid in user_idx_map and iid in item_idx_map:
                rows.append(user_idx_map[uid])
                cols.append(item_idx_map[iid])
                data.append(1)
    mat = csr_matrix((data, (rows, cols)), shape=(len(all_users), len(all_items)))
    als_model = AlternatingLeastSquares(factors=EMB_DIM, iterations=15, use_gpu=True, random_state=42)
    als_model.fit(mat)
    user_factors = als_model.user_factors
    item_factors = als_model.item_factors
    info("✅ ALS на GPU завершена.")
    week_decay_map = {}
    max_week_idx = len(recent_weeks) - 1
    decay_lambda = 0.2
    for idx, w in enumerate(recent_weeks):
        p = os.path.join(DATA_ROOT, "subsamples", subsample_name, "train", f"{w}.parquet")
        cols = ["user_id", "item_id"] + EXPLICIT_ACTIONS
        df_week = pl.read_parquet(p, columns=cols)
        df_week = df_week.filter(pl.any_horizontal([pl.col(a) for a in EXPLICIT_ACTIONS]))
        for row in df_week.iter_rows(named=True):
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            weight = math.exp(-decay_lambda * (max_week_idx - idx))
            current_weight = week_decay_map.get((uid, iid), 0.0)
            if weight > current_weight:
                week_decay_map[(uid, iid)] = weight
        del df_week
        gc.collect()
    info(f"✅ Week decay map построен.")
    ex_table = pq.read_table(example_path)
    example_items = [int(x) for x in ex_table.column("item_id").to_numpy()]
    info(f"Таргетных items: {len(example_items)}")
    # FAISS-GPU
    info("Строим FAISS-GPU index для похожих...")
   cpu_index = faiss.IndexFlatIP(EMB_DIM)
cpu_index.add(item_vecs)
    similar_cache = {}
    for iid in tqdm(example_items, desc="Похожие (FAISS-GPU)"):
        idx = item_id_to_index.get(iid)
        if idx is None:
            similar_cache[iid] = []
            continue
        D, I = cpu_index.search(item_vecs[idx:idx+1], SIM_TOPK + 1)
        similar = []
        for d, i in zip(D[0][1:], I[0][1:]):
            similar.append((int(item_ids[i]), float(d)))
        similar_cache[iid] = similar
    true_relevant_users = defaultdict(set)
    for iid, users in item_to_users.items():
        true_relevant_users[iid] |= users
    if val_path:
        info("Добавляем pseudo-labels из validation...")
        try:
            lazy_df = (
                pl.scan_parquet(val_path)
                .select(["user_id", "item_id"] + EXPLICIT_ACTIONS)
                .filter(pl.any_horizontal([pl.col(a) for a in EXPLICIT_ACTIONS]))
            )
            for chunk in lazy_df.collect(streaming=True).iter_slices(500_000):
                for row in chunk.iter_rows(named=True):
                    true_relevant_users[int(row["item_id"])].add(int(row["user_id"]))
                del chunk
                gc.collect()
            info("✅ Pseudo-labels добавлены.")
        except Exception as e:
            warn(f"Ошибка validation: {e}")
    # Targeted co-occurrence
    example_items_set = set(example_items)
    cooc_map = build_cooccurrence_targeted(subsample_name, recent_weeks, example_items_set, min_cooc=2)
    author_to_sorted_fans = {}
    for uid, amap in author_affinity.items():
        for author_id, sc in amap.items():
            author_to_sorted_fans.setdefault(author_id, []).append((uid, sc))
    for author_id, lst in author_to_sorted_fans.items():
        lst.sort(key=lambda x: -x[1])
        if len(lst) > 500:
            author_to_sorted_fans[author_id] = lst[:500]
    info("Генерация кандидатов...")
    candidate_users_per_item = {}
    for iid in tqdm(example_items, desc="Кандидаты"):
        cand_users = set()
        if iid in item_to_users:
            cand_users |= item_to_users[iid]
        for si, _ in similar_cache.get(iid, []):
            if si in item_to_users:
                cand_users |= item_to_users[si]
        author = item_to_author.get(iid)
        if author is not None and author in author_to_sorted_fans:
            for u, _ in author_to_sorted_fans[author]:
                cand_users.add(u)
        if len(cand_users) < CANDIDATES_PER_ITEM:
            for u in popular_users:
                cand_users.add(u)
                if len(cand_users) >= CANDIDATES_PER_ITEM:
                    break
        candidate_users_per_item[iid] = list(cand_users)
    # Подготовка данных и обучение XGBoost GPU
    info("Подготовка данных для XGBoost...")
    X, y, groups = prepare_data(
        example_items, candidate_users_per_item,
        item_to_author, author_affinity,
        item_popularity, user_popularity,
        user_factors, item_factors,
        user_idx_map, item_idx_map,
        similar_cache, item_to_users,
        week_decay_map, cooc_map, author_to_sorted_fans,
        dict(true_relevant_users),
        DEFAULT_WEIGHTS
    )
    info(f"Данные: {X.shape}, групп: {len(groups)}")
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(groups)
    params = {
        'tree_method': 'gpu_hist',
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@100',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 150,
        'random_state': 42,
        'verbosity': 0,
    }
    info("Обучаем XGBoost на GPU...")
    model = xgb.train(params, dtrain, num_boost_round=150)
    info("✅ XGBoost обучен.")
    # Скоринг
    info("Скоринг на GPU...")
    item_to_scored_users = {}
    item_candidates_extended = {}
    for iid in tqdm(example_items, desc="Скоринг"):
        candidates = candidate_users_per_item[iid]
        features_list = [
            extract_features(
                uid, iid, similar_cache.get(iid, []),
                item_to_users, item_to_author, author_affinity,
                item_popularity, user_popularity,
                user_factors, item_factors,
                user_idx_map, item_idx_map,
                week_decay_map, DEFAULT_WEIGHTS, cooc_map, author_to_sorted_fans
            )
            for uid in candidates
        ]
        X_score = np.array(features_list, dtype=np.float32)
        dtest = xgb.DMatrix(X_score)
        scores = model.predict(dtest)
        scored = [(float(s), int(uid)) for s, uid in zip(scores, candidates)]
        scored.sort(key=lambda x: -x[0])
        item_to_scored_users[iid] = scored[:USERS_PER_ITEM]
        item_candidates_extended[iid] = scored[USERS_PER_ITEM:300]
    # Постпроцессинг
    per_item_users = apply_smart_assignment_limit(
        item_to_scored_users, USERS_PER_ITEM, MAX_ASSIGN_PER_USER,
        popular_users, item_candidates_extended,
        item_to_users_gt=dict(true_relevant_users)
    )
    # Финальная запись
    final_per_item_users = {}
    for iid in example_items:
        base = per_item_users.get(iid, [])
        seen = set()
        uniq = []
        for u in base:
            if u not in seen:
                uniq.append(u)
                seen.add(u)
            if len(uniq) == USERS_PER_ITEM:
                break
        if len(uniq) < USERS_PER_ITEM:
            for u in popular_users:
                if u not in seen:
                    uniq.append(u)
                    seen.add(u)
                if len(uniq) == USERS_PER_ITEM:
                    break
            while len(uniq) < USERS_PER_ITEM:
                uniq.append(0)
        final_per_item_users[iid] = uniq[:USERS_PER_ITEM]
    def write_submission(example_item_ids, per_item_users, output_path, popular_users):
        out_item_ids_u32 = np.array(example_item_ids, dtype=np.uint32)
        user_lists_u32 = []
        for iid in example_item_ids:
            users = per_item_users.get(iid, [])
            seen = set()
            uniq = []
            for u in users:
                if u not in seen:
                    uniq.append(u)
                    seen.add(u)
                if len(uniq) == USERS_PER_ITEM:
                    break
            if len(uniq) < USERS_PER_ITEM:
                for u in popular_users:
                    if u not in seen:
                        uniq.append(u)
                        seen.add(u)
                    if len(uniq) == USERS_PER_ITEM:
                        break
                while len(uniq) < USERS_PER_ITEM:
                    uniq.append(0)
            user_lists_u32.append([np.uint32(u) for u in uniq])
        table = pa.Table.from_pydict({"item_id": out_item_ids_u32, "user_id": user_lists_u32}, schema=ARROW_SCHEMA)
        pq.write_table(table, output_path, compression="zstd")
        info(f"Сохранено: {output_path}")
        # Валидация
        sub = pl.read_parquet(output_path)
        ex = sub.explode('user_id')
        ex_i = ex.group_by('item_id').len()
        ex_u = ex.group_by('user_id').len()
        assert ex.group_by('item_id', 'user_id').len().select('len').max().item() == 1, "doubles"
        assert ex_i.select('len').min().item() == 100, "le recs on item"
        assert ex_i.select('len').max().item() == 100, "ge recs on item"
        assert ex_u.select('len').max().item() == 100, "users capacity"
        info("✅ Валидация пройдена.")
    write_submission(example_items, final_per_item_users, output_path, popular_users)
    info("🏆 Готово. Submission сохранён.")

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=str, default=DEFAULT_SUBSAMPLE)
    ap.add_argument("--weeks", type=str, nargs="*", default=None)
    ap.add_argument("--val-week", type=str, default="week_25")
    ap.add_argument("--output", type=str, default=OUTPUT_PATH)
    return ap.parse_args()
def main():
    args = parse_args()
    run_pipeline(args.subsample, args.weeks, args.output, args.val_week)
if __name__ == "__main__":
    main()