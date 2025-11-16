#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================
# 🚀 Оптимизированный для L4 (24GB VRAM) + 2 недели теста
# =========================
import os
import gc
import math
import argparse
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ML (только GPU, если есть)
import faiss
import xgboost as xgb
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix

# === Настройки (уменьшены под L4/2 недели) ===
REPO_ID = "deepvk/VK-LSVD"
DATA_ROOT = "VK-LSVD"
OUTPUT_PATH = "submission_l4_test.parquet"

# ↓↓↓ КРИТИЧЕСКИЕ ПАРАМЕТРЫ ОПТИМИЗАЦИИ ↓↓↓
CANDIDATES_PER_ITEM = 400          # Было 1000
USERS_PER_ITEM = 100
MAX_ASSIGN_PER_USER = 60           # Было 100 — снижает нагрузку в постпроцессинге
EMB_DIM = 64                       # Было 128 — экономим RAM
SIM_TOPK = 200                     # Было 500
ALS_FACTORS = 96                   # Было 192
ALS_ITERATIONS = 15                # Было 25
BPR_FACTORS = 64                   # Было 128
BPR_ITERATIONS = 15                # Было 20

# Упрощённые веса (без diversity, embedding similarity — экономия RAM/CPU)
WEIGHTS = {
    "W_ALS": 10.0,
    "W_BPR": 8.0,
    "W_SIMHITS": 5.0,
    "W_AUTHOR": 4.0,
    "W_POPULARITY": 0.3,
    "W_TIME_DECAY": 1.5,
    "W_COOC": 2.5,
    "W_AUTHOR_FAN_RANK": 1.0,
    "W_RECENCY": 2.0,
}

EXPLICIT_ACTIONS = ["like", "share", "bookmark", "click_on_author"]
ACTION_WEIGHTS = {"like": 1.0, "share": 3.0, "bookmark": 2.0, "click_on_author": 1.5}

ARROW_SCHEMA = pa.schema([
    pa.field("item_id", pa.uint32()),
    pa.field("user_id", pa.list_(pa.uint32()))
])

# === Утилиты ===
def info(msg: str):
    print(f"[INFO] {msg}", flush=True)

def warn(msg: str):
    print(f"[WARN] {msg}", flush=True)

# --- Загрузка данных (без лишних действий)
def ensure_download(subsample: str, filename: str) -> str:
    local = os.path.join(DATA_ROOT, filename)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    info(f"📥 Скачиваем {filename}")
    hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename, local_dir=DATA_ROOT)
    return local

def load_items_meta(path: str) -> Dict[int, int]:
    df = pl.read_parquet(path).select(["item_id", "author_id"])
    return {int(r[0]): int(r[1]) for r in df.iter_rows()}

def load_embeddings(path: str, dim: int):
    data = np.load(path)
    item_ids = data["item_id"].astype(np.int32)
    embs = data["embedding"][:, :dim].astype(np.float32)
    return item_ids, embs

# --- FAISS (без нормализации внутри — экономим RAM)
def build_faiss_index(item_vecs: np.ndarray, item_ids: np.ndarray):
    info("🚀 Строим FAISS (Flat-IP, no IVF для малых данных)...")
    d = item_vecs.shape[1]
    faiss.normalize_L2(item_vecs)
    index = faiss.IndexFlatIP(d)
    index.add(item_vecs)
    info(f"✅ FAISS: {index.ntotal} items")
    return index, item_ids

def get_similar_items_batch(indices: List[int], item_vecs: np.ndarray, index, item_ids, topk: int):
    if not indices:
        return {}
    batch_vecs = item_vecs[indices].copy()
    faiss.normalize_L2(batch_vecs)
    D, I = index.search(batch_vecs, topk + 1)  # +1: исключить self
    res = {}
    for i, (dists, idxs) in enumerate(zip(D, I)):
        sims = [(int(item_ids[idx]), float(d)) for d, idx in zip(dists[1:], idxs[1:]) if idx >= 0]
        res[indices[i]] = sims
    return res

# --- Модели
def build_csr_matrix(user_to_items_weighted, all_users, all_items):
    u_map = {u: i for i, u in enumerate(all_users)}
    i_map = {i: j for j, i in enumerate(all_items)}
    rows, cols, data = [], [], []
    for uid, items in user_to_items_weighted.items():
        for iid, w in items.items():
            if uid in u_map and iid in i_map:
                rows.append(u_map[uid])
                cols.append(i_map[iid])
                data.append(w)
    return csr_matrix((data, (rows, cols)), shape=(len(all_users), len(all_items))), u_map, i_map

# --- Скоринг моделей (упрощённый)
def score_batch(uids, iid, u_factors, i_factors, u_map, i_map):
    iix = i_map.get(iid)
    if iix is None:
        return np.zeros(len(uids))
    item_vec = i_factors[iix]
    scores = np.zeros(len(uids))
    for k, uid in enumerate(uids):
        uix = u_map.get(uid)
        if uix is not None:
            scores[k] = u_factors[uix].dot(item_vec)
    return scores

# --- Основной пайплайн (оптимизированный)
def run_pipeline(subsample: str, weeks: List[str], output_path: str):
    # --- 1. Метаданные
    meta_path = ensure_download(subsample, "metadata/items_metadata.parquet")
    emb_path = ensure_download(subsample, "metadata/item_embeddings.npz")
    example_path = ensure_download(subsample, "subsamples/example.parquet")

    item_to_author = load_items_meta(meta_path)
    item_ids, item_vecs = load_embeddings(emb_path, EMB_DIM)
    item_id_to_idx = {int(i): idx for idx, i in enumerate(item_ids)}

    # --- 2. Загрузка 2 недель (или по аргументам)
    info(f"📂 Загружаем недели: {weeks}")
    user_to_items = defaultdict(set)
    user_to_items_weighted = defaultdict(lambda: defaultdict(float))
    item_to_users = defaultdict(set)
    user_pop = defaultdict(int)
    author_affinity = defaultdict(lambda: defaultdict(float))
    user_last_week = {}

    for w_idx, week in enumerate(weeks):
        path = os.path.join(DATA_ROOT, "subsamples", subsample, "train", f"{week}.parquet")
        if not os.path.exists(path):
            path = ensure_download(subsample, f"subsamples/{subsample}/train/{week}.parquet")

        df = pl.read_parquet(path, columns=["user_id", "item_id"] + EXPLICIT_ACTIONS)
        for row in df.iter_rows(named=True):
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            w = sum(ACTION_WEIGHTS[a] * bool(row[a]) for a in EXPLICIT_ACTIONS)
            if w > 0:
                user_to_items[uid].add(iid)
                user_to_items_weighted[uid][iid] += w
                item_to_users[iid].add(uid)
                user_pop[uid] += 1
                author_affinity[uid][item_to_author.get(iid, -1)] += w
                user_last_week[uid] = w_idx
        del df; gc.collect()

    all_users = sorted(user_to_items.keys())
    all_items = sorted(item_to_users.keys())
    item_pop = {i: len(us) for i, us in item_to_users.items()}
    info(f"📊 Данные: {len(all_users)} users, {len(all_items)} items")

    # --- 3. Обучение ALS (GPU)
    info("🏋️ Обучаем ALS...")
    mat, u_map_als, i_map_als = build_csr_matrix(user_to_items_weighted, all_users, all_items)
    als = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERATIONS,
        use_gpu=True,
        random_state=42,
        regularization=0.05,
        alpha=1.0
    )
    als.fit(mat)
    u_f_als, i_f_als = als.user_factors, als.item_factors
    del mat, als; gc.collect()

    # --- 4. Обучение BPR (GPU)
    info("🏋️ Обучаем BPR...")
    mat_bin, u_map_bpr, i_map_bpr = build_csr_matrix(
        {u: {i: 1.0 for i in its} for u, its in user_to_items.items()},
        all_users, all_items
    )
    bpr = BayesianPersonalizedRanking(
        factors=BPR_FACTORS,
        iterations=BPR_ITERATIONS,
        use_gpu=True,
        random_state=42,
        regularization=0.05,
        learning_rate=0.02
    )
    bpr.fit(mat_bin)
    u_f_bpr, i_f_bpr = bpr.user_factors, bpr.item_factors
    del mat_bin, bpr; gc.collect()

    # --- 5. Эмбеддинги → FAISS
    index, _ = build_faiss_index(item_vecs, item_ids)

    # --- 6. Example items
    ex_table = pq.read_table(example_path)
    example_items = [int(x) for x in ex_table.column("item_id").to_numpy()]
    info(f"🎯 Items для рекомендации: {len(example_items)}")

    # --- 7. Похожие items (batched)
    info("🔍 Ищем похожие items...")
    valid_indices = [item_id_to_idx[i] for i in example_items if i in item_id_to_idx]
    similar_cache = {}
    batch_size = 200
    for i in range(0, len(valid_indices), batch_size):
        batch = valid_indices[i:i+batch_size]
        res = get_similar_items_batch(batch, item_vecs, index, item_ids, SIM_TOPK)
        similar_cache.update(res)
    info(f"✅ Найдено похожих для {len(similar_cache)}/{len(example_items)} items")

    # --- 8. Co-occurrence (только для example items, быстрая версия)
    info("🤝 Строим co-occurrence...")
    cooc_map = {}
    for iid in tqdm(example_items, desc="Co-occur"):
        users = item_to_users.get(iid, set())
        if len(users) < 2:
            continue
        cooc = defaultdict(int)
        for uid in users:
            for other_iid in user_to_items.get(uid, []):
                if other_iid != iid:
                    cooc[other_iid] += 1
        cooc = {k: v for k, v in cooc.items() if v >= 2}
        if cooc:
            top = dict(sorted(cooc.items(), key=lambda x: -x[1])[:100])
            cooc_map[iid] = top

    # --- 9. Author fans (только топ-200)
    author_to_fans = {}
    for uid, amap in author_affinity.items():
        for aid, sc in amap.items():
            author_to_fans.setdefault(aid, []).append((uid, sc))
    for aid in author_to_fans:
        author_to_fans[aid] = sorted(author_to_fans[aid], key=lambda x: -x[1])[:200]

    # --- 10. Кандидаты
    info("👥 Генерация кандидатов...")
    popular_users = sorted(user_pop.keys(), key=lambda u: -user_pop[u])
    candidate_users_per_item = {}

    for iid in tqdm(example_items, desc="Кандидаты", total=len(example_items)):
        cand = set(item_to_users.get(iid, []))

        # Похожие items
        for si, sim in similar_cache.get(iid, [])[:50]:
            users = item_to_users.get(si, [])
            take = min(len(users), int(30 * sim))
            cand.update(list(users)[:take])

        # Автор
        aid = item_to_author.get(iid)
        if aid and aid in author_to_fans:
            cand.update(u for u, _ in author_to_fans[aid][:50])

        # Co-occurrence
        if iid in cooc_map:
            for co_iid, cnt in list(cooc_map[iid].items())[:30]:
                users = item_to_users.get(co_iid, [])
                take = min(len(users), int(20 * math.log1p(cnt)))
                cand.update(list(users)[:take])

        # Дополним популярными
        if len(cand) < CANDIDATES_PER_ITEM:
            for u in popular_users:
                cand.add(u)
                if len(cand) >= CANDIDATES_PER_ITEM:
                    break

        candidate_users_per_item[iid] = list(cand)[:CANDIDATES_PER_ITEM]

    # --- 11. Экстракция фичей (упрощённая, без embedding similarity)
    def extract_features(uid, iid, als_sc, bpr_sc, sim_items):
        features = []

        # Модели
        features.append(als_sc * WEIGHTS["W_ALS"])
        features.append(bpr_sc * WEIGHTS["W_BPR"])
        features.append((als_sc + bpr_sc) * 0.8)

        # IBCF
        hits = sum(sim for si, sim in sim_items[:30] if uid in item_to_users.get(si, set()))
        features.append(hits * WEIGHTS["W_SIMHITS"])

        # Автор
        aid = item_to_author.get(iid)
        author_sc = author_affinity.get(uid, {}).get(aid, 0.0)
        features.append(math.sqrt(author_sc + 1e-6) * WEIGHTS["W_AUTHOR"])

        # Популярность
        ip = item_pop.get(iid, 0)
        features.append(math.log1p(ip) * WEIGHTS["W_POPULARITY"])

        # Time decay (на основе последней недели)
        decay = math.exp(-0.1 * (len(weeks) - 1 - user_last_week.get(uid, 0)))
        features.append(decay * WEIGHTS["W_TIME_DECAY"])

        # Co-occurrence
        cooc_sc = 0
        if iid in cooc_map:
            user_items = user_to_items.get(uid, set())
            for co_iid, cnt in cooc_map[iid].items():
                if co_iid in user_items:
                    cooc_sc += cnt
        features.append(math.log1p(cooc_sc) * WEIGHTS["W_COOC"])

        # Recency
        weeks_ago = len(weeks) - 1 - user_last_week.get(uid, -10)
        recency = math.exp(-0.2 * max(weeks_ago, 0))
        features.append(recency * WEIGHTS["W_RECENCY"])

        return features

    # --- 12. Подготовка данных для XGBoost
    info("🧩 Подготовка данных для XGBoost...")
    X, y, groups = [], [], []

    for iid in tqdm(example_items[:500], desc="Фичи (первые 500 items для теста!)"):
        candidates = candidate_users_per_item[iid]
        als_scores = score_batch(candidates, iid, u_f_als, i_f_als, u_map_als, i_map_als)
        bpr_scores = score_batch(candidates, iid, u_f_bpr, i_f_bpr, u_map_bpr, i_map_bpr)
        sims = similar_cache.get(iid, [])

        group = []
        for uid, als_sc, bpr_sc in zip(candidates, als_scores, bpr_scores):
            X.append(extract_features(uid, iid, als_sc, bpr_sc, sims))
            y.append(1 if uid in item_to_users.get(iid, set()) else 0)
            group.append(uid)
        groups.append(len(group))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    info(f"✅ Данные: {X.shape}, положительных: {y.sum()}/{len(y)}")

    # --- 13. XGBoost (GPU, но быстро)
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(groups)

    params = {
        'tree_method': 'gpu_hist',
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@50',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': 42,
        'verbosity': 0,
    }

    info("📈 Обучаем XGBoost (100 итераций)...")
    model = xgb.train(params, dtrain, num_boost_round=100)

    # --- 14. Скоринг (full, но с batch)
    info("🎯 Скоринг всех items...")
    item_to_scored = {}

    batch_size = 20
    for i in range(0, len(example_items), batch_size):
        batch_items = example_items[i:i+batch_size]
        for iid in batch_items:
            cand = candidate_users_per_item[iid]
            als_sc = score_batch(cand, iid, u_f_als, i_f_als, u_map_als, i_map_als)
            bpr_sc = score_batch(cand, iid, u_f_bpr, i_f_bpr, u_map_bpr, i_map_bpr)
            sims = similar_cache.get(iid, [])

            feats = [extract_features(uid, iid, float(a), float(b), sims)
                     for uid, a, b in zip(cand, als_sc, bpr_sc)]
            scores = model.predict(xgb.DMatrix(np.array(feats, dtype=np.float32)))
            scored = sorted(zip(scores, cand), key=lambda x: -x[0])
            item_to_scored[iid] = [uid for _, uid in scored[:USERS_PER_ITEM]]

        del feats, scores; gc.collect()

    # --- 15. Постпроцессинг (простой — без замены)
    info("📝 Постпроцессинг...")
    final = {}
    for iid in example_items:
        users = item_to_scored.get(iid, [])
        seen = set()
        uniq = []
        for u in users:
            if u not in seen and u != 0:
                uniq.append(u)
                seen.add(u)
            if len(uniq) == USERS_PER_ITEM:
                break
        # Дополним популярными
        for u in popular_users:
            if len(uniq) >= USERS_PER_ITEM:
                break
            if u not in seen:
                uniq.append(u)
                seen.add(u)
        final[iid] = uniq[:USERS_PER_ITEM]

    # --- 16. Запись
    out_items = np.array(example_items, dtype=np.uint32)
    user_lists = [[np.uint32(u) for u in final[iid]] for iid in example_items]

    table = pa.Table.from_pydict(
        {"item_id": out_items, "user_id": user_lists},
        schema=ARROW_SCHEMA
    )
    pq.write_table(table, output_path, compression="zstd")
    info(f"✅ Сохранено: {output_path}")

    # Валидация
    sub = pl.read_parquet(output_path)
    assert sub.select(pl.col("user_id").list.len()).min().item() == USERS_PER_ITEM
    info("✅ Валидация пройдена!")

# === CLI ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=str, default="up-0.9_ip-0.9")
    parser.add_argument("--weeks", nargs="+", default=["week_24", "week_25"])
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    try:
        run_pipeline(args.subsample, args.weeks, args.output)
    except Exception as e:
        import traceback
        print("❌ ОШИБКА:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()