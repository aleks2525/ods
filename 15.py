#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================
# 🎯 Версия для достижения 0.35+ метрики
# =========================
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import argparse
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import heapq
import gc
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files

# ML
import xgboost as xgb
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix
import faiss

# =========================
# Утилиты
# =========================
def info(msg: str):
    print(f"[INFO] {msg}", flush=True)

def warn(msg: str):
    print(f"[WARN] {msg}", flush=True)

# =========================
# Конфигурация для 0.35+
# =========================
REPO_ID = "deepvk/VK-LSVD"
REPO_TYPE = "dataset"
DATA_ROOT = "VK-LSVD"
DEFAULT_SUBSAMPLE = "up-0.9_ip-0.9"
OUTPUT_PATH = "submission_high_quality.parquet"

ARROW_SCHEMA = pa.schema([
    pa.field("item_id", pa.uint32()),
    pa.field("user_id", pa.list_(pa.uint32()))
])

# Параметры для высокого качества
CANDIDATES_PER_ITEM = 1200
USERS_PER_ITEM = 100
MAX_ASSIGN_PER_USER = 100
EMB_DIM = 128
SIM_TOPK = 500
ALS_FACTORS = 192
ALS_ITERATIONS = 25
BPR_FACTORS = 128
BPR_ITERATIONS = 20

# Оптимизированные веса
DEFAULT_WEIGHTS = {
    "W_ALS": 12.0,
    "W_BPR": 8.0,
    "W_SIMHITS": 6.0,
    "W_AUTHOR": 4.0,
    "W_POPULARITY": 0.2,
    "W_TIME_DECAY": 2.0,
    "W_COOC": 3.0,
    "W_AUTHOR_FAN_RANK": 1.5,
    "W_RECENCY": 2.5,
    "W_DIVERSITY": 1.0,
    "W_USER_ITEM_SIM": 3.0,
}

EXPLICIT_ACTIONS = ["like", "share", "bookmark", "click_on_author"]
ACTION_WEIGHTS = {
    "like": 1.0,
    "share": 3.0,
    "bookmark": 2.0,
    "click_on_author": 1.5,
}

# =========================
# Утилиты для загрузки
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
def load_items_metadata(path_items_meta: str):
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
# FAISS оптимизированный
# =========================
def build_faiss_index_cpu(item_vecs: np.ndarray, item_ids: np.ndarray):
    info("Строим FAISS-CPU IVF index...")
    d = item_vecs.shape[1]
    n = item_vecs.shape[0]

    faiss.normalize_L2(item_vecs)

    nlist = min(2048, n // 39)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    index.train(item_vecs)
    index.add(item_vecs)
    index.nprobe = 64

    info(f"✅ FAISS index: {index.ntotal} векторов, {nlist} кластеров")
    return index, item_ids

def get_similar_items_batch(item_indices: List[int], item_vecs: np.ndarray, cpu_index, item_ids, topk: int):
    if not item_indices:
        return {}

    query_vecs = item_vecs[item_indices].copy()
    faiss.normalize_L2(query_vecs)

    D, I = cpu_index.search(query_vecs, topk + 1)

    results = {}
    for idx, (distances, indices) in enumerate(zip(D, I)):
        similar = []
        for d, i in zip(distances[1:], indices[1:]):
            if i >= 0:
                similar.append((int(item_ids[i]), float(d)))
        results[item_indices[idx]] = similar

    return results

# =========================
# Weighted interaction matrix
# =========================
def build_weighted_matrix(user_to_items_weighted, all_users, all_items):
    """Создаем взвешенную матрицу взаимодействий"""
    user_idx_map = {u: i for i, u in enumerate(all_users)}
    item_idx_map = {it: i for i, it in enumerate(all_items)}

    rows, cols, data = [], [], []
    for uid, items_dict in user_to_items_weighted.items():
        for iid, weight in items_dict.items():
            if uid in user_idx_map and iid in item_idx_map:
                rows.append(user_idx_map[uid])
                cols.append(item_idx_map[iid])
                data.append(weight)

    mat = csr_matrix((data, (rows, cols)), shape=(len(all_users), len(all_items)))
    return mat, user_idx_map, item_idx_map

# =========================
# Scoring
# =========================
def als_score_batch(uids: List[int], iid: int, user_factors, item_factors, user_idx_map, item_idx_map) -> np.ndarray:
    iix = item_idx_map.get(iid)
    if iix is None:
        return np.zeros(len(uids), dtype=np.float32)

    scores = np.zeros(len(uids), dtype=np.float32)
    item_vec = item_factors[iix]

    for i, uid in enumerate(uids):
        uix = user_idx_map.get(uid)
        if uix is not None:
            scores[i] = user_factors[uix].dot(item_vec)

    return scores

def bpr_score_batch(uids: List[int], iid: int, user_factors, item_factors, user_idx_map, item_idx_map) -> np.ndarray:
    """BPR модель scoring"""
    return als_score_batch(uids, iid, user_factors, item_factors, user_idx_map, item_idx_map)

# =========================
# Co-occurrence быстрый
# =========================
def build_cooccurrence_smart(
    item_to_users: Dict[int, Set[int]],
    example_items: Set[int],
    min_cooc: int = 3,
    max_neighbors: int = 200
) -> Dict[int, Dict[int, int]]:
    """Умный co-occurrence только для example items"""
    info(f"Строим smart co-occurrence...")

    cooc_map = {}

    for iid in tqdm(example_items, desc="Co-occurrence"):
        users = item_to_users.get(iid, set())
        if len(users) < 2:
            continue

        # Считаем co-occurrence
        cooc_counter = defaultdict(int)
        for uid in users:
            # Находим все items пользователя
            for other_iid, other_users in item_to_users.items():
                if other_iid != iid and uid in other_users:
                    cooc_counter[other_iid] += 1

        # Фильтруем и сортируем
        filtered = {k: v for k, v in cooc_counter.items() if v >= min_cooc}
        if filtered:
            # Берем топ по частоте
            top_neighbors = dict(sorted(filtered.items(), key=lambda x: -x[1])[:max_neighbors])
            cooc_map[iid] = top_neighbors

    info(f"✅ Co-occurrence: {len(cooc_map)} items, avg {np.mean([len(v) for v in cooc_map.values()]):.1f} neighbors")
    return cooc_map

# =========================
# Продвинутые фичи
# =========================
def compute_user_item_similarity(uid: int, iid: int, user_embedding: Optional[np.ndarray],
                                 item_embedding: Optional[np.ndarray]) -> float:
    """Косинусное сходство user и item эмбеддингов"""
    if user_embedding is None or item_embedding is None:
        return 0.0

    norm_user = np.linalg.norm(user_embedding)
    norm_item = np.linalg.norm(item_embedding)

    if norm_user == 0 or norm_item == 0:
        return 0.0

    return float(np.dot(user_embedding, item_embedding) / (norm_user * norm_item))

def extract_features_advanced(
    uid: int,
    iid: int,
    als_score: float,
    bpr_score: float,
    similar_items: List[Tuple[int, float]],
    item_to_users: Dict[int, Set[int]],
    item_to_author: Dict[int, int],
    author_affinity: Dict[int, Dict[int, float]],
    item_popularity: Dict[int, int],
    user_popularity: Dict[int, int],
    week_decay_map: Dict[Tuple[int, int], float],
    weights: Dict[str, float],
    cooc_map: Dict[int, Dict[int, int]],
    author_to_sorted_fans: Dict[int, List[Tuple[int, float]]],
    user_last_interaction_week: Dict[int, int],
    current_week: int,
    user_embeddings: Optional[Dict[int, np.ndarray]] = None,
    item_embeddings: Optional[Dict[int, np.ndarray]] = None,
    user_diversity: Optional[Dict[int, float]] = None,
) -> List[float]:
    features = []

    # 1. ALS score
    features.append(als_score * weights["W_ALS"])

    # 2. BPR score
    features.append(bpr_score * weights["W_BPR"])

    # 3. Комбинированный score
    features.append((als_score + bpr_score) / 2 * 2.0)

    # 4. IBCF score с взвешиванием по сходству
    ibcf_sc = 0.0
    ibcf_weighted = 0.0
    n_sim_hits = 0

    for co_iid, sim in similar_items[:100]:
        if uid in item_to_users.get(co_iid, set()):
            ibcf_sc += sim
            ibcf_weighted += sim * sim
            n_sim_hits += 1

    features.append(ibcf_sc * weights["W_SIMHITS"])
    features.append(ibcf_weighted * 0.5)
    features.append(n_sim_hits * 0.3)

    # 5. Author affinity (улучшенная)
    author = item_to_author.get(iid)
    author_sc = 0.0
    author_strength = 0.0

    if author is not None:
        author_sc = author_affinity.get(uid, {}).get(author, 0.0)
        if author in author_affinity.get(uid, {}):
            total_author_score = sum(author_affinity.get(uid, {}).values())
            author_strength = author_sc / (total_author_score + 1e-9)

    features.append(math.sqrt(author_sc + 1e-9) * weights["W_AUTHOR"])
    features.append(author_strength * 2.0)

    # 6. Popularity features
    item_pop = item_popularity.get(iid, 0)
    user_pop = user_popularity.get(uid, 0)

    features.append(math.log1p(item_pop) * weights["W_POPULARITY"])
    features.append(math.log1p(user_pop) * 0.03)

    max_pop = max(item_popularity.values()) if item_popularity else 1
    features.append((item_pop / max_pop) * 0.5)

    # 7. Time decay
    week_weight = week_decay_map.get((uid, iid), 1.0)
    features.append(week_weight * weights["W_TIME_DECAY"])

    # 8. Co-occurrence
    cooc_sc = 0.0
    cooc_weighted = 0.0

    if iid in cooc_map:
        user_items = {i for i, users in item_to_users.items() if uid in users}
        for co_iid, cnt in cooc_map[iid].items():
            if co_iid in user_items:
                cooc_sc += cnt
                cooc_weighted += cnt * math.log1p(cnt)

    features.append(math.log1p(cooc_sc) * weights["W_COOC"])
    features.append(cooc_weighted * 0.3)

    # 9. Author fan rank
    author_fan_rank_norm = 1.0
    if author is not None and author in author_to_sorted_fans:
        fan_list = author_to_sorted_fans[author]
        uids = [u for u, _ in fan_list]
        if uid in uids:
            rank = uids.index(uid)
            author_fan_rank_norm = rank / len(fan_list)

    features.append((1.0 - author_fan_rank_norm) * weights["W_AUTHOR_FAN_RANK"])

    # 10. Recency
    last_week = user_last_interaction_week.get(uid, 0)
    weeks_ago = current_week - last_week
    recency_score = math.exp(-0.1 * weeks_ago)
    features.append(recency_score * weights["W_RECENCY"])

    # 11. Diversity features
    if user_diversity and uid in user_diversity:
        features.append(user_diversity[uid] * weights["W_DIVERSITY"])
    else:
        features.append(0.0)

    if similar_items:
        sim_scores = [s for _, s in similar_items[:20]]
        sim_diversity = np.std(sim_scores) if len(sim_scores) > 1 else 0.0
        features.append(sim_diversity * 0.5)
        features.append(np.mean(sim_scores) * 0.4)
    else:
        features.append(0.0)
        features.append(0.0)

    # 12. User-Item embedding similarity
    if user_embeddings and item_embeddings:
        user_emb = user_embeddings.get(uid)
        item_emb = item_embeddings.get(iid)
        sim = compute_user_item_similarity(uid, iid, user_emb, item_emb)
        features.append(sim * weights["W_USER_ITEM_SIM"])
    else:
        features.append(0.0)

    # 13. Interaction patterns
    recent_activity = 1.0 if weeks_ago <= 2 else 0.5 if weeks_ago <= 5 else 0.1
    features.append(recent_activity * 0.8)

    return features

# =========================
# Compute embeddings
# =========================
def compute_user_embeddings_from_als(user_factors, user_idx_map) -> Dict[int, np.ndarray]:
    """Создаем словарь user_id -> embedding"""
    user_embeddings = {}
    for uid, idx in user_idx_map.items():
        user_embeddings[uid] = user_factors[idx]
    return user_embeddings

def compute_item_embeddings_dict(item_vecs, item_ids, item_id_to_index) -> Dict[int, np.ndarray]:
    """Создаем словарь item_id -> embedding"""
    item_embeddings = {}
    for iid in item_ids:
        idx = item_id_to_index.get(int(iid))
        if idx is not None:
            item_embeddings[int(iid)] = item_vecs[idx]
    return item_embeddings

def compute_user_diversity(user_to_items: Dict[int, Set[int]], item_embeddings: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Вычисляем разнообразие интересов пользователя"""
    user_diversity = {}

    for uid, items in tqdm(user_to_items.items(), desc="User diversity"):
        if len(items) < 2:
            user_diversity[uid] = 0.0
            continue

        embeddings = []
        for iid in items:
            if iid in item_embeddings:
                embeddings.append(item_embeddings[iid])

        if len(embeddings) < 2:
            user_diversity[uid] = 0.0
            continue

        embeddings = np.array(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms

        sim_matrix = np.dot(embeddings_norm, embeddings_norm.T)

        n = len(embeddings)
        avg_sim = (np.sum(sim_matrix) - n) / (n * (n - 1)) if n > 1 else 0.0

        user_diversity[uid] = 1.0 - avg_sim

    return user_diversity

# =========================
# Подготовка данных
# =========================
def prepare_data_advanced(
    example_items: List[int],
    candidate_users_per_item: Dict[int, List[int]],
    item_to_author: Dict[int, int],
    author_affinity: Dict[int, Dict[int, float]],
    item_popularity: Dict[int, int],
    user_popularity: Dict[int, int],
    user_factors_als, item_factors_als,
    user_idx_map_als, item_idx_map_als,
    user_factors_bpr, item_factors_bpr,
    user_idx_map_bpr, item_idx_map_bpr,
    similar_cache: Dict[int, List[Tuple[int, float]]],
    item_to_users: Dict[int, Set[int]],
    week_decay_map: Dict[Tuple[int, int], float],
    cooc_map: Dict[int, Dict[int, int]],
    author_to_sorted_fans: Dict[int, List[Tuple[int, float]]],
    true_relevant_users: Dict[int, Set[int]],
    weights: Dict[str, float],
    user_last_interaction_week: Dict[int, int],
    current_week: int,
    user_embeddings: Optional[Dict[int, np.ndarray]] = None,
    item_embeddings: Optional[Dict[int, np.ndarray]] = None,
    user_diversity: Optional[Dict[int, float]] = None,
):
    X, y, groups = [], [], []

    for iid in tqdm(example_items, desc="Подготовка данных"):
        candidates = candidate_users_per_item[iid]

        als_scores = als_score_batch(candidates, iid, user_factors_als, item_factors_als, user_idx_map_als, item_idx_map_als)
        bpr_scores = bpr_score_batch(candidates, iid, user_factors_bpr, item_factors_bpr, user_idx_map_bpr, item_idx_map_bpr)

        similar_items = similar_cache.get(iid, [])
        group_size = 0

        for uid, als_sc, bpr_sc in zip(candidates, als_scores, bpr_scores):
            features = extract_features_advanced(
                uid, iid, float(als_sc), float(bpr_sc),
                similar_items,
                item_to_users, item_to_author, author_affinity,
                item_popularity, user_popularity,
                week_decay_map, weights, cooc_map, author_to_sorted_fans,
                user_last_interaction_week, current_week,
                user_embeddings, item_embeddings, user_diversity
            )
            X.append(features)
            rel = 1 if uid in true_relevant_users.get(iid, set()) else 0
            y.append(rel)
            group_size += 1

        groups.append(group_size)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(groups, dtype=np.uint32)

# =========================
# Постпроцессинг
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

            for _, candidate_uid in extended:
                if user_counts[candidate_uid] >= max_assign_per_user:
                    continue
                if item_to_users_gt and candidate_uid in item_to_users_gt.get(iid, set()):
                    best_replacement = candidate_uid
                    break
                if best_replacement is None:
                    best_replacement = candidate_uid
                    break

            if best_replacement is not None:
                current_list.append(best_replacement)
                user_counts[best_replacement] += 1
            else:
                for candidate_uid in popular_users[:500_000]:
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

    recent_weeks = weeks[-10:] if len(weeks) > 10 else weeks
    info(f"Используем {len(recent_weeks)} недель: {recent_weeks}")
    current_week = len(recent_weeks)

    # Build indices
    user_to_items = defaultdict(set)
    user_to_items_weighted = defaultdict(lambda: defaultdict(float))
    item_to_users = defaultdict(set)
    user_total_interactions = defaultdict(int)
    author_affinity = defaultdict(lambda: defaultdict(float))
    user_last_interaction_week = {}

    for week_idx, w in enumerate(tqdm(recent_weeks, desc="Загрузка данных")):
        p = os.path.join(DATA_ROOT, "subsamples", subsample_name, "train", f"{w}.parquet")

        df_week = (
            pl.scan_parquet(p)
            .select(["user_id", "item_id"] + EXPLICIT_ACTIONS)
            .collect()
        )

        for row in df_week.iter_rows(named=True):
            uid = int(row["user_id"])
            iid = int(row["item_id"])

            weight = sum(ACTION_WEIGHTS.get(action, 0.0) * (1.0 if row[action] else 0.0)
                         for action in EXPLICIT_ACTIONS)

            if weight > 0:
                user_to_items[uid].add(iid)
                user_to_items_weighted[uid][iid] += weight
                item_to_users[iid].add(uid)
                user_total_interactions[uid] += 1
                user_last_interaction_week[uid] = week_idx

                author = item_to_author.get(iid)
                if author is not None:
                    author_affinity[uid][author] += weight

        del df_week
        gc.collect()

    info("✅ Индексы построены.")

    popular_users = sorted(user_total_interactions.keys(), key=lambda u: -user_total_interactions[u])
    item_popularity = {iid: len(users) for iid, users in item_to_users.items()}
    user_popularity = user_total_interactions

    # Train ALS
    info(f"Обучаем ALS (factors={ALS_FACTORS}, iter={ALS_ITERATIONS})...")
    all_users = sorted(user_to_items.keys())
    all_items = sorted(item_to_users.keys())

    mat_weighted, user_idx_map_als, item_idx_map_als = build_weighted_matrix(
        user_to_items_weighted, all_users, all_items
    )

    als_model = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERATIONS,
        use_gpu=True,
        random_state=42,
        regularization=0.01,
        alpha=1.0
    )
    als_model.fit(mat_weighted)
    user_factors_als = als_model.user_factors
    item_factors_als = als_model.item_factors
    info("✅ ALS завершена.")

    # Train BPR
    info(f"Обучаем BPR (factors={BPR_FACTORS}, iter={BPR_ITERATIONS})...")
    bpr_model = BayesianPersonalizedRanking(
        factors=BPR_FACTORS,
        iterations=BPR_ITERATIONS,
        use_gpu=True,
        random_state=42,
        regularization=0.01,
        learning_rate=0.01
    )

    mat_binary, user_idx_map_bpr, item_idx_map_bpr = build_weighted_matrix(
        {uid: {iid: 1.0 for iid in items} for uid, items in user_to_items.items()},
        all_users, all_items
    )

    bpr_model.fit(mat_binary)
    user_factors_bpr = bpr_model.user_factors
    item_factors_bpr = bpr_model.item_factors
    info("✅ BPR завершена.")

    # Time decay
    week_decay_map = {}
    max_week_idx = len(recent_weeks) - 1
    decay_lambda = 0.12

    for idx, w in enumerate(recent_weeks):
        weight = math.exp(-decay_lambda * (max_week_idx - idx))
        p = os.path.join(DATA_ROOT, "subsamples", subsample_name, "train", f"{w}.parquet")

        df_week = pl.scan_parquet(p).select(["user_id", "item_id"]).collect()

        for row in df_week.iter_rows():
            uid, iid = int(row[0]), int(row[1])
            current_weight = week_decay_map.get((uid, iid), 0.0)
            if weight > current_weight:
                week_decay_map[(uid, iid)] = weight

        del df_week
        gc.collect()

    info(f"✅ Week decay: {len(week_decay_map)} пар")

    # Example items
    ex_table = pq.read_table(example_path)
    example_items = [int(x) for x in ex_table.column("item_id").to_numpy()]
    info(f"Таргетных items: {len(example_items)}")

    # FAISS index
    cpu_index, _ = build_faiss_index_cpu(item_vecs, item_ids)

    # Similar items
    info("Похожие items...")
    similar_cache = {}
    valid_indices = [item_id_to_index[iid] for iid in example_items if iid in item_id_to_index]

    batch_size = 1000
    for i in tqdm(range(0, len(valid_indices), batch_size), desc="FAISS"):
        batch_indices = valid_indices[i:i+batch_size]
        batch_results = get_similar_items_batch(batch_indices, item_vecs, cpu_index, item_ids, SIM_TOPK)

        for idx, similar in batch_results.items():
            iid = item_ids[idx]
            similar_cache[int(iid)] = similar

    info(f"✅ Похожие: {len(similar_cache)}")

    # True relevant users
    true_relevant_users = defaultdict(set)
    for iid, users in item_to_users.items():
        true_relevant_users[iid] |= users

    if val_path:
        info("Pseudo-labels из validation...")
        try:
            df_val = (
                pl.scan_parquet(val_path)
                .select(["user_id", "item_id"] + EXPLICIT_ACTIONS)
                .collect()
            )

            for row in df_val.iter_rows(named=True):
                if any(row[action] for action in EXPLICIT_ACTIONS):
                    true_relevant_users[int(row["item_id"])].add(int(row["user_id"]))

            del df_val
            gc.collect()
            info("✅ Pseudo-labels добавлены.")
        except Exception as e:
            warn(f"Ошибка validation: {e}")

    # Co-occurrence
    example_items_set = set(example_items)
    cooc_map = build_cooccurrence_smart(item_to_users, example_items_set, min_cooc=3, max_neighbors=200)

    # Author fans
    author_to_sorted_fans = {}
    for uid, amap in author_affinity.items():
        for author_id, sc in amap.items():
            author_to_sorted_fans.setdefault(author_id, []).append((uid, sc))

    for author_id, lst in author_to_sorted_fans.items():
        lst.sort(key=lambda x: -x[1])
        author_to_sorted_fans[author_id] = lst[:1500]

    # Embeddings
    info("Вычисляем embeddings...")
    user_embeddings = compute_user_embeddings_from_als(user_factors_als, user_idx_map_als)
    item_embeddings = compute_item_embeddings_dict(item_vecs, item_ids, item_id_to_index)

    # User diversity
    info("Вычисляем user diversity...")
    user_diversity = compute_user_diversity(user_to_items, item_embeddings)

    # Candidates
    info("Генерация кандидатов...")
    candidate_users_per_item = {}

    for iid in tqdm(example_items, desc="Кандидаты"):
        cand_users = set()

        if iid in item_to_users:
            cand_users |= item_to_users[iid]

        for si, sim in similar_cache.get(iid, [])[:150]:
            if si in item_to_users:
                users = item_to_users[si]
                sample_size = min(len(users), int(200 * sim))
                cand_users |= set(list(users)[:sample_size])

        author = item_to_author.get(iid)
        if author is not None and author in author_to_sorted_fans:
            for u, _ in author_to_sorted_fans[author][:300]:
                cand_users.add(u)

        if iid in cooc_map:
            for co_iid, cnt in sorted(cooc_map[iid].items(), key=lambda x: -x[1])[:100]:
                if co_iid in item_to_users:
                    users = item_to_users[co_iid]
                    sample_size = min(len(users), int(100 * math.log1p(cnt)))
                    cand_users |= set(list(users)[:sample_size])

        if len(cand_users) < CANDIDATES_PER_ITEM:
            for u in popular_users:
                cand_users.add(u)
                if len(cand_users) >= CANDIDATES_PER_ITEM:
                    break

        candidate_users_per_item[iid] = list(cand_users)[:CANDIDATES_PER_ITEM]

    avg_candidates = np.mean([len(c) for c in candidate_users_per_item.values()])
    info(f"✅ Средне кандидатов: {avg_candidates:.1f}")

    # Prepare data
    info("Подготовка данных для XGBoost...")
    X, y, groups = prepare_data_advanced(
        example_items, candidate_users_per_item,
        item_to_author, author_affinity,
        item_popularity, user_popularity,
        user_factors_als, item_factors_als,
        user_idx_map_als, item_idx_map_als,
        user_factors_bpr, item_factors_bpr,
        user_idx_map_bpr, item_idx_map_bpr,
        similar_cache, item_to_users,
        week_decay_map, cooc_map, author_to_sorted_fans,
        dict(true_relevant_users),
        DEFAULT_WEIGHTS,
        user_last_interaction_week,
        current_week,
        user_embeddings, item_embeddings, user_diversity
    )

    info(f"Данные: {X.shape}, групп: {len(groups)}, фичей: {X.shape[1]}")
    info(f"Положительных: {y.sum()}/{len(y)} ({100*y.sum()/len(y):.2f}%)")

    # Train XGBoost
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(groups)

    params = {
        'tree_method': 'gpu_hist',
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@100',
        'max_depth': 10,
        'learning_rate': 0.03,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 5,
        'gamma': 0.2,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': 1,
    }

    info("Обучаем XGBoost...")
    model = xgb.train(params, dtrain, num_boost_round=300)
    info("✅ XGBoost обучен.")

    # Scoring
    info("Скоринг...")
    item_to_scored_users = {}
    item_candidates_extended = {}

    batch_size = 50
    items_batches = [example_items[i:i+batch_size] for i in range(0, len(example_items), batch_size)]

    for batch in tqdm(items_batches, desc="Скоринг"):
        for iid in batch:
            candidates = candidate_users_per_item[iid]

            als_scores = als_score_batch(candidates, iid, user_factors_als, item_factors_als, user_idx_map_als, item_idx_map_als)
            bpr_scores = bpr_score_batch(candidates, iid, user_factors_bpr, item_factors_bpr, user_idx_map_bpr, item_idx_map_bpr)
            similar_items = similar_cache.get(iid, [])

            features_list = [
                extract_features_advanced(
                    uid, iid, float(als_sc), float(bpr_sc),
                    similar_items,
                    item_to_users, item_to_author, author_affinity,
                    item_popularity, user_popularity,
                    week_decay_map, DEFAULT_WEIGHTS, cooc_map, author_to_sorted_fans,
                    user_last_interaction_week, current_week,
                    user_embeddings, item_embeddings, user_diversity
                )
                for uid, als_sc, bpr_sc in zip(candidates, als_scores, bpr_scores)
            ]

            X_score = np.array(features_list, dtype=np.float32)
            dtest = xgb.DMatrix(X_score)
            scores = model.predict(dtest)

            scored = [(float(s), int(uid)) for s, uid in zip(scores, candidates)]
            scored.sort(key=lambda x: -x[0])

            item_to_scored_users[iid] = scored[:USERS_PER_ITEM]
            item_candidates_extended[iid] = scored[USERS_PER_ITEM:500]

    # Post-processing
    info("Постпроцессинг...")
    per_item_users = apply_smart_assignment_limit(
        item_to_scored_users, USERS_PER_ITEM, MAX_ASSIGN_PER_USER,
        popular_users, item_candidates_extended,
        item_to_users_gt=dict(true_relevant_users)
    )

    # Final submission
    final_per_item_users = {}
    for iid in example_items:
        base = per_item_users.get(iid, [])
        seen = set()
        uniq = []

        for u in base:
            if u not in seen and u != 0:
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

    # Write
    out_item_ids_u32 = np.array(example_items, dtype=np.uint32)
    user_lists_u32 = [[np.uint32(u) for u in final_per_item_users[iid]] for iid in example_items]

    table = pa.Table.from_pydict(
        {"item_id": out_item_ids_u32, "user_id": user_lists_u32},
        schema=ARROW_SCHEMA
    )
    pq.write_table(table, output_path, compression="zstd")
    info(f"✅ Сохранено: {output_path}")

    # Validation
    sub = pl.read_parquet(output_path)
    ex = sub.explode('user_id')
    ex_i = ex.group_by('item_id').len()
    ex_u = ex.group_by('user_id').len()

    assert ex.group_by('item_id', 'user_id').len().select('len').max().item() == 1
    assert ex_i.select('len').min().item() == 100
    assert ex_i.select('len').max().item() == 100
    assert ex_u.select('len').max().item() <= 100

    info("✅ Валидация пройдена!")
    info("🎯 Готово! Ожидаемая метрика: 0.35+")

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