# app_fastapi.py
from typing import List, Optional, Tuple
import math
import time

import numpy as np
import pandas as pd
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, hmac
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends


# =====================================================
# =================== ALGORITMOS ======================
# =====================================================

def heuristic_kmeans_capacitado(
    clients, clusters, compat, max_iter=30, seed: int = 42,
    ls_max_iter: int = 30, repair_rounds: int = 2
):
    """
    Heurística K-Means Capacitada otimizada:
      - Vetorização de distâncias
      - Pré-filtragem de compatibilidade
      - np.bincount para cargas
      - argpartition em vez de argsort
    """
    pts = stack_latlon(clients)
    n, k = len(clients), len(clusters)
    _ = np.random.default_rng(seed)

    d1 = clients["demanda1"].to_numpy(float)
    d2 = clients["demanda2"].to_numpy(float)
    c1 = clusters["capacidade1"].to_numpy(float)
    c2 = clusters["capacidade2"].to_numpy(float)

    # --- inicialização KMeans++ com cobertura ---
    centers = kmeanspp_geo_coverage(pts, k, seed=seed)
    assign = np.full(n, -1, dtype=int)

    # ordem por demanda (maiores primeiro)
    order = np.argsort(-(d1 + d2))

    # Pré-filtragem: lista de clusters compatíveis por cliente
    valid_clusters = [np.where(compat[i])[0] for i in range(n)]

    for it in range(max_iter):
        loads1 = np.zeros(k, dtype=float)
        loads2 = np.zeros(k, dtype=float)
        new_assign = np.full(n, -1, dtype=int)

        # --- matriz de distâncias cliente–cluster ---
        dist_matrix = haversine_matrix(pts, centers)  # vetorizado

        # alocação gulosa respeitando capacidade/skills
        for i in order:
            best_j, best_d = -1, 1e18
            for j in valid_clusters[i]:
                if np.isnan(centers[j]).any():
                    continue
                nd1, nd2 = loads1[j] + d1[i], loads2[j] + d2[i]
                if _capacity_ok(nd1, nd2, c1[j], c2[j]):
                    d = dist_matrix[i, j]
                    if d < best_d:
                        best_d, best_j = d, j
            if best_j >= 0:
                new_assign[i] = best_j
                loads1[best_j] += d1[i]
                loads2[best_j] += d2[i]

        # ----- reparo: tentar inserir não alocados -----
        not_assigned = np.where(new_assign < 0)[0].tolist()
        for _ in range(repair_rounds):
            if not not_assigned:
                break
            still = []
            for i in not_assigned:
                inserted = False
                dist_row = dist_matrix[i, valid_clusters[i]]
                candidates = valid_clusters[i][np.argpartition(dist_row, range(len(dist_row)))]
                for j in candidates:
                    if np.isnan(centers[j]).any():
                        continue
                    if _capacity_ok(loads1[j] + d1[i], loads2[j] + d2[i], c1[j], c2[j]):
                        new_assign[i] = j
                        loads1[j] += d1[i]; loads2[j] += d2[i]
                        inserted = True
                        break
                if not inserted:
                    still.append(i)
            not_assigned = still

        # atualiza centróides por mediana geográfica
        new_centers = update_centroids_geomed(pts, new_assign, k, old_centers=centers)

        # critério de parada
        if np.all(assign == new_assign):
            assign = new_assign
            centers = new_centers
            break

        assign, centers = new_assign, new_centers

    # ----- busca local: relocate/swap -----
    loads1 = np.bincount(assign[assign >= 0], weights=d1[assign >= 0], minlength=k)
    loads2 = np.bincount(assign[assign >= 0], weights=d2[assign >= 0], minlength=k)

    improved = True; it_ls = 0
    # garante matriz de distâncias atual
    dist_matrix = haversine_matrix(pts, centers)
    while improved and it_ls < ls_max_iter:
        improved = False; it_ls += 1

        # --- relocate ---
        for i in range(n):
            j = assign[i]
            if j < 0:
                continue
            best_gain = 0.0; best_l = -1
            for l in valid_clusters[i]:
                if l == j or np.isnan(centers[l]).any():
                    continue
                if _capacity_ok(loads1[l] + d1[i], loads2[l] + d2[i], c1[l], c2[l]):
                    cur = dist_matrix[i, j]
                    new = dist_matrix[i, l]
                    gain = cur - new
                    if gain > 1e-6:
                        best_gain, best_l = gain, l
            if best_l >= 0:
                loads1[j] -= d1[i]; loads2[j] -= d2[i]
                loads1[best_l] += d1[i]; loads2[best_l] += d2[i]
                assign[i] = best_l
                improved = True
        if improved:
            centers = update_centroids_geomed(pts, assign, k, old_centers=centers)
            dist_matrix = haversine_matrix(pts, centers)

        # --- swap simples ---
        for i in range(n):
            j1 = assign[i]
            if j1 < 0:
                continue
            for t in range(i + 1, n):
                j2 = assign[t]
                if j2 < 0 or j2 == j1:
                    continue
                if j2 not in valid_clusters[i] or j1 not in valid_clusters[t]:
                    continue
                if _capacity_ok(loads1[j1] - d1[i] + d1[t], loads2[j1] - d2[i] + d2[t], c1[j1], c2[j1]) and \
                   _capacity_ok(loads1[j2] - d1[t] + d1[i], loads2[j2] - d2[t] + d2[i], c1[j2], c2[j2]):
                    cur = dist_matrix[i, j1] + dist_matrix[t, j2]
                    new = dist_matrix[i, j2] + dist_matrix[t, j1]
                    if new + 1e-6 < cur:
                        loads1[j1] = loads1[j1] - d1[i] + d1[t]
                        loads2[j1] = loads2[j1] - d2[i] + d2[t]
                        loads1[j2] = loads1[j2] - d1[t] + d1[i]
                        loads2[j2] = loads2[j2] - d2[t] + d2[i]
                        assign[i], assign[t] = j2, j1
                        improved = True
        if improved:
            centers = update_centroids_geomed(pts, assign, k, old_centers=centers)
            dist_matrix = haversine_matrix(pts, centers)

    total_cost = _compute_cost(pts, assign, centers)
    alocados = int(np.sum(assign >= 0))
    print(f"[Heurístico Capacitado] Obj={total_cost:.2f} | LS={it_ls} | Alocados={alocados}/{n}")
    return assign, centers, total_cost


def stack_latlon(df: pd.DataFrame) -> np.ndarray:
    return df[["lat", "lon"]].astype("float64").to_numpy(copy=True)


def kmeanspp_geo_coverage(points: np.ndarray, k: int, seed: int = 42, coverage_boost: float = 0.25) -> np.ndarray:
    """
    KMeans++ com leve viés para cobertura geográfica: mistura D^2 com farthest-point sampling.
    """
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    centers = np.empty((k, 2), dtype=float)
    centers[0] = points[rng.integers(0, n)]
    d2 = np.full(n, np.inf, dtype=float)
    for c in range(1, k):
        d2 = np.minimum(d2, np.linalg.norm(points - centers[c - 1], axis=1) ** 2)
        farthest_idx = np.argsort(d2)[-max(1, int(coverage_boost * n / k)):]
        if rng.random() < 0.5 and len(farthest_idx) > 0:
            idx = rng.choice(farthest_idx)
        else:
            s = d2.sum()
            probs = d2 / (s if s > 0 else 1.0)
            idx = rng.choice(n, p=probs)
        centers[c] = points[idx]
    return centers


def haversine_matrix(pts, centers):
    """Matriz [n x k] de distâncias haversine vetorizadas."""
    lat1, lon1 = np.radians(pts[:, 0])[:, None], np.radians(pts[:, 1])[:, None]
    lat2, lon2 = np.radians(centers[:, 0])[None, :], np.radians(centers[:, 1])[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))  # km


def _capacity_ok(load1, load2, c1, c2) -> bool:
    return (load1 <= c1 + 1e-9) and (load2 <= c2 + 1e-9)


def _compute_cost(pts: np.ndarray, assign: np.ndarray, centers: np.ndarray) -> float:
    total = 0.0
    for i, j in enumerate(assign):
        if j >= 0 and not np.isnan(centers[j]).any():
            total += haversine(pts[i], centers[j])
    return total


def update_centroids_geomed(points_latlon: np.ndarray, assign: np.ndarray, k: int, old_centers: Optional[np.ndarray] = None) -> np.ndarray:
    """Atualiza centróides por mediana geográfica por cluster."""
    centers = np.full((k, 2), np.nan, dtype=np.float64)
    for j in range(k):
        idx = np.where(assign == j)[0]
        if idx.size > 0:
            centers[j] = geometric_median(points_latlon[idx])
        else:
            if old_centers is not None:
                centers[j] = old_centers[j]
    return centers


def haversine(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(h))


def geometric_median(points: np.ndarray, eps: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """Weiszfeld robust 2D geometric median; fallback para média se degenerado."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.array([np.nan, np.nan], dtype=float)
    x = pts.mean(axis=0)  # chute inicial
    for _ in range(max_iter):
        d = np.linalg.norm(pts - x, axis=1)
        if np.any(d < eps):
            return pts[np.argmin(d)]
        w = 1.0 / d
        x_new = (w[:, None] * pts).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < eps:
            return x_new
        x = x_new
    return x


def solve_grasp_vnd_greedy_fixedK(
    clients: pd.DataFrame,
    clusters: pd.DataFrame,
    compat: np.ndarray,
    *,
    alpha: float = 0.25,
    multi_starts: int = 6,
    seed: int = 42,
    repair_rounds: int = 2,
    vnd_max_iter: int = 25,
    relocate_k: int = 6,
    swap_pairs: int = 64,
    sample_candidates: int = 128,
    tol_improv: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    GRASP + VND com K FIXO (sem apagar clusters)
    """
    rng_global = np.random.default_rng(seed)

    # dados base
    pts = stack_latlon(clients)
    n, k = len(clients), len(clusters)

    d1 = clients["demanda1"].to_numpy(float)
    d2 = clients["demanda2"].to_numpy(float)
    c1 = clusters["capacidade1"].to_numpy(float)
    c2 = clusters["capacidade2"].to_numpy(float)

    def _capacity_ok(load1, load2, C1, C2) -> bool:
        return (load1 <= C1 + 1e-9) and (load2 <= C2 + 1e-9)

    def _cost(assign: np.ndarray, centers: np.ndarray) -> float:
        return sum(
            haversine(pts[i], centers[j])
            for i, j in enumerate(assign)
            if j >= 0 and not np.isnan(centers[j]).any()
        )

    def _recompute_loads(assign: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L1 = np.zeros(k, float)
        L2 = np.zeros(k, float)
        for j in range(k):
            idx = np.where(assign == j)[0]
            if idx.size > 0:
                L1[j] = d1[idx].sum()
                L2[j] = d2[idx].sum()
        return L1, L2

    # ---------- garantir que nenhum cluster fique vazio ----------
    def enforce_nonempty(assign: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        loads1, loads2 = _recompute_loads(assign)
        empty = [j for j in range(k) if np.sum(assign == j) == 0]

        for j in empty:
            if np.isnan(centers[j]).any():
                centers[j] = pts.mean(axis=0)

            best_i, best_src, best_delta = -1, -1, float("inf")
            for i in range(n):
                src = assign[i]
                if src == -1 or src == j:
                    continue
                if not compat[i, j]:
                    continue
                if not _capacity_ok(loads1[j] + d1[i], loads2[j] + d2[i], c1[j], c2[j]):
                    continue
                if np.isnan(centers[src]).any() or np.isnan(centers[j]).any():
                    continue
                cur = haversine(pts[i], centers[src])
                new = haversine(pts[i], centers[j])
                delta = new - cur
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_src = src

            if best_i >= 0:
                assign[best_i] = j
                loads1[best_src] -= d1[best_i]; loads2[best_src] -= d2[best_i]
                loads1[j] += d1[best_i];       loads2[j] += d2[best_i]
                centers = update_centroids_geomed(pts, assign, k, old_centers=centers)

        return assign, centers

    # -------------- CONSTRUÇÃO GULOSA --------------
    def greedy_construct(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        assign = np.full(n, -1, dtype=int)
        centers = np.full((k, 2), np.nan)
        used = np.zeros(n, dtype=bool)

        order_clusters = rng.permutation(k)

        def _seed_score(i, jj):
            return (d1[i] / max(c1[jj], 1e-9)) + (d2[i] / max(c2[jj], 1e-9))

        for jj in order_clusters:
            viable = np.where((~used) & compat[:, jj] & (d1 <= c1[jj]) & (d2 <= c2[jj]))[0]
            if viable.size == 0:
                continue

            scores = np.array([_seed_score(i, jj) for i in viable])
            order = np.argsort(-scores)
            rcl_len = max(1, int(np.ceil(alpha * len(order))))
            s_idx = viable[order[:rcl_len]]
            seed_i = rng.choice(s_idx)

            assign[seed_i] = jj
            used[seed_i] = True
            load1 = d1[seed_i]
            load2 = d2[seed_i]
            members = [seed_i]
            center = pts[seed_i]

            while True:
                pool = np.where((~used) & compat[:, jj] &
                                (d1 + load1 <= c1[jj] + 1e-12) &
                                (d2 + load2 <= c2[jj] + 1e-12))[0]
                if pool.size == 0:
                    break
                m = min(sample_candidates, pool.size)
                cand = rng.choice(pool, size=m, replace=False)

                best_i, best_dist = -1, float("inf")
                for i in cand:
                    dist = haversine(pts[i], center)
                    if dist < best_dist:
                        best_i, best_dist = i, dist

                if best_i == -1:
                    break

                assign[best_i] = jj
                used[best_i] = True
                members.append(best_i)
                load1 += d1[best_i]; load2 += d2[best_i]
                center = pts[members].mean(axis=0)

            centers[jj] = geometric_median(pts[np.array(members)]) if members else np.array([np.nan, np.nan])

        # reparo simples para não alocados
        for _ in range(repair_rounds):
            loads1, loads2 = _recompute_loads(assign)
            improved_any = False
            for i in range(n):
                if assign[i] != -1:
                    continue
                best_j, best_d = -1, float("inf")
                for j in range(k):
                    if not compat[i, j] or np.isnan(centers[j]).any():
                        continue
                    if _capacity_ok(loads1[j] + d1[i], loads2[j] + d2[i], c1[j], c2[j]):
                        dist = haversine(pts[i], centers[j])
                        if dist < best_d:
                            best_d, best_j = dist, j
                if best_j >= 0:
                    assign[i] = best_j
                    loads1[best_j] += d1[i]; loads2[best_j] += d2[i]
                    improved_any = True
            if not improved_any:
                break

        centers = update_centroids_geomed(pts, assign, k, old_centers=centers)
        assign, centers = enforce_nonempty(assign, centers)
        centers = update_centroids_geomed(pts, assign, k, old_centers=centers)
        return assign, centers

    # -------------- VND --------------
    def vnd_refine(assign: np.ndarray, centers: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        loads1, loads2 = _recompute_loads(assign)

        it = 0
        improved_global = True
        while improved_global and it < vnd_max_iter:
            it += 1
            improved_global = False

            # RELOCATE
            for i in range(n):
                j = assign[i]
                if j < 0 or np.isnan(centers[j]).any():
                    continue

                d_to_centers = []
                for l in range(k):
                    if np.isnan(centers[l]).any():
                        continue
                    d_to_centers.append((l, haversine(pts[i], centers[l])))
                if not d_to_centers:
                    continue
                d_to_centers.sort(key=lambda t: t[1])
                cand_clusters = [l for l, _ in d_to_centers[:max(1, relocate_k)]]

                cur = haversine(pts[i], centers[j])
                best_gain, best_l = 0.0, -1
                for l in cand_clusters:
                    if l == j or not compat[i, l]:
                        continue
                    if not _capacity_ok(loads1[l] + d1[i], loads2[l] + d2[i], c1[l], c2[l]):
                        continue
                    newd = haversine(pts[i], centers[l])
                    gain = cur - newd
                    if gain > best_gain + tol_improv:
                        best_gain, best_l = gain, l

                if best_l >= 0:
                    loads1[j] -= d1[i]; loads2[j] -= d2[i]
                    loads1[best_l] += d1[i]; loads2[best_l] += d2[i]
                    assign[i] = best_l
                    improved_global = True

            if improved_global:
                centers = update_centroids_geomed(pts, assign, k, old_centers=centers)

            # SWAP (amostrado)
            idx_assigned = np.where(assign >= 0)[0]
            if idx_assigned.size >= 2:
                m = min(swap_pairs, (idx_assigned.size * (idx_assigned.size - 1)) // 2)
                rng_local = np.random.default_rng(rng.integers(1, 1_000_000))
                for _ in range(m):
                    i, t = rng_local.choice(idx_assigned, size=2, replace=False)
                    j1, j2 = assign[i], assign[t]
                    if j1 == j2 or j1 < 0 or j2 < 0:
                        continue
                    if (not compat[i, j2]) or (not compat[t, j1]):
                        continue
                    if not _capacity_ok(loads1[j1] - d1[i] + d1[t], loads2[j1] - d2[i] + d2[t], c1[j1], c2[j1]):
                        continue
                    if not _capacity_ok(loads1[j2] - d1[t] + d1[i], loads2[j2] - d2[t] + d2[i], c1[j2], c2[j2]):
                        continue
                    if np.isnan(centers[j1]).any() or np.isnan(centers[j2]).any():
                        continue
                    cur = haversine(pts[i], centers[j1]) + haversine(pts[t], centers[j2])
                    new = haversine(pts[i], centers[j2]) + haversine(pts[t], centers[j1])
                    if cur - new > tol_improv:
                        assign[i], assign[t] = j2, j1
                        loads1[j1] = loads1[j1] - d1[i] + d1[t]
                        loads2[j1] = loads2[j1] - d2[i] + d2[t]
                        loads1[j2] = loads1[j2] - d1[t] + d1[i]
                        loads2[j2] = loads2[j2] - d2[t] + d2[i]
                        improved_global = True

            if improved_global:
                centers = update_centroids_geomed(pts, assign, k, old_centers=centers)

            if improved_global:
                assign, centers = enforce_nonempty(assign, centers)
                centers = update_centroids_geomed(pts, assign, k, old_centers=centers)

        return assign, centers

    # -------------- GRASP MULTI-START --------------
    best_assign, best_centers, best_obj = None, None, float("inf")
    for s in range(multi_starts):
        rng = np.random.default_rng(int(seed + 101 * s))
        assign, centers = greedy_construct(rng)
        assign, centers = vnd_refine(assign, centers, rng)
        assign, centers = enforce_nonempty(assign, centers)
        centers = update_centroids_geomed(pts, assign, k, old_centers=centers)

        obj = _cost(assign, centers)
        if obj < best_obj - tol_improv:
            best_obj = obj
            best_assign = assign.copy()
            best_centers = centers.copy()

    print(f"[GRASP-VND-Greedy(K-FIXO)] custo={best_obj:.2f} | alocados={(best_assign>=0).sum()}/{n}")
    return best_assign, best_centers, best_obj


def greedy_teitz_bart_sa_fast(
    clients: pd.DataFrame,
    clusters: pd.DataFrame,
    compat: np.ndarray,
    *,
    seed: int = 42,
    sa_iters: int = 1000,
    sa_T0: float = 1.0,
    sa_Tf: float = 1e-3,
    reheats: int = 2,
    n_starts: int = 8,
    neighbor_base: int | None = None,
    tabu_horizon: int = 100,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Greedy + refino local + SA + compactação
    """
    try:
        from joblib import Parallel, delayed
        _JOBLIB = True
    except Exception:
        _JOBLIB = False

    rng_global = np.random.default_rng(seed)
    pts = stack_latlon(clients)
    n, k = len(clients), len(clusters)

    d1 = clients["demanda1"].to_numpy(float)
    d2 = clients["demanda2"].to_numpy(float)
    c1 = clusters["capacidade1"].to_numpy(float)
    c2 = clusters["capacidade2"].to_numpy(float)

    def _capacity_ok(load1, load2, C1, C2) -> bool:
        return (load1 <= C1 + 1e-9) and (load2 <= C2 + 1e-9)

    def _cost(assign: np.ndarray, centers: np.ndarray) -> float:
        return _compute_cost(pts, assign, centers)

    def _recompute_loads(assign: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L1 = np.zeros(k, float)
        L2 = np.zeros(k, float)
        for j in range(k):
            idx = np.where(assign == j)[0]
            if idx.size > 0:
                L1[j] = d1[idx].sum()
                L2[j] = d2[idx].sum()
        return L1, L2

    def _greedy_init(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        assign = np.full(n, -1, dtype=int)
        centers = np.full((k, 2), np.nan)
        used = np.zeros(n, dtype=bool)
        order_clusters = rng.permutation(k)

        for jj in order_clusters:
            viable = np.where((compat[:, jj]) & (d1 <= c1[jj]) & (d2 <= c2[jj]) & (~used))[0]
            if viable.size == 0:
                continue
            seed_i = rng.choice(viable)
            assign[seed_i] = jj
            used[seed_i] = True

            L1, L2 = d1[seed_i], d2[seed_i]
            members = [seed_i]
            ctr = pts[seed_i]

            improved = True
            while improved:
                improved = False
                best_i, best_dist = -1, float("inf")
                pool = np.where(~used & compat[:, jj])[0]
                if pool.size == 0:
                    break
                sample_sz = min(max(64, int(np.sqrt(pool.size))), pool.size)
                cand = rng.choice(pool, size=sample_sz, replace=False)
                for i in cand:
                    if _capacity_ok(L1 + d1[i], L2 + d2[i], c1[jj], c2[jj]):
                        dist = haversine(pts[i], ctr)
                        if dist < best_dist:
                            best_dist, best_i = dist, i
                if best_i >= 0:
                    assign[best_i] = jj
                    used[best_i] = True
                    members.append(best_i)
                    L1 += d1[best_i]; L2 += d2[best_i]
                    ctr = pts[members].mean(axis=0)
                    improved = True

            centers[jj] = geometric_median(pts[np.array(members)]) if members else np.array([np.nan, np.nan])

        return assign, update_centroids_geomed(pts, assign, k)

    def _local_refine(assign: np.ndarray, centers: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        loads1, loads2 = _recompute_loads(assign)
        improved = True
        it = 0
        while improved and it < 3:
            it += 1
            improved = False

            idx_assigned = np.where(assign >= 0)[0]
            if idx_assigned.size == 0:
                break
            sample_i = rng.choice(idx_assigned, size=min(len(idx_assigned), 64), replace=False)
            for i in sample_i:
                j = assign[i]
                cur = haversine(pts[i], centers[j]) if j >= 0 and not np.isnan(centers[j]).any() else 1e18
                cand_clusters = rng.choice(np.arange(k), size=min(k, 8), replace=False)
                for l in cand_clusters:
                    if l == j or not compat[i, l] or np.isnan(centers[l]).any():
                        continue
                    if _capacity_ok(loads1[l] + d1[i], loads2[l] + d2[i], c1[l], c2[l]):
                        new = haversine(pts[i], centers[l])
                        if new + 1e-6 < cur:
                            assign[i] = l
                            loads1[j] -= d1[i]; loads2[j] -= d2[i]
                            loads1[l] += d1[i]; loads2[l] += d2[i]
                            improved = True
                            break
            if improved:
                centers = update_centroids_geomed(pts, assign, k)

            idx_assigned = np.where(assign >= 0)[0]
            if idx_assigned.size < 2:
                break
            pairs = rng.choice(idx_assigned, size=(min(len(idx_assigned) // 2, 32), 2), replace=False)
            for i, t in pairs:
                j1, j2 = assign[i], assign[t]
                if j1 == j2 or j1 < 0 or j2 < 0:
                    continue
                if not compat[i, j2] or not compat[t, j1]:
                    continue
                if _capacity_ok(loads1[j1] - d1[i] + d1[t], loads2[j1] - d2[i] + d2[t], c1[j1], c2[j1]) and \
                   _capacity_ok(loads1[j2] - d1[t] + d1[i], loads2[j2] - d2[t] + d2[i], c1[j2], c2[j2]):
                    cur = haversine(pts[i], centers[j1]) + haversine(pts[t], centers[j2])
                    new = haversine(pts[i], centers[j2]) + haversine(pts[t], centers[j1])
                    if new + 1e-6 < cur:
                        assign[i], assign[t] = j2, j1
                        loads1[j1] = loads1[j1] - d1[i] + d1[t]
                        loads2[j1] = loads2[j1] - d2[i] + d2[t]
                        loads1[j2] = loads1[j2] - d1[t] + d1[i]
                        loads2[j2] = loads2[j2] - d2[t] + d2[i]
                        improved = True
            if improved:
                centers = update_centroids_geomed(pts, assign, k)

        return assign, centers

    def _greedy_compact_refine(assign: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        loads1, loads2 = _recompute_loads(assign)

        bad_clients = []
        for j in range(k):
            idx = np.where(assign == j)[0]
            if idx.size == 0 or np.isnan(centers[j]).any():
                continue
            dists = np.array([haversine(pts[i], centers[j]) for i in idx], dtype=float)
            med = np.median(dists)
            bad = idx[dists > med + 1e-12]
            if bad.size > 0:
                bad_clients.extend(bad.tolist())

        if not bad_clients:
            return assign, centers

        bad_clients = np.array(bad_clients, dtype=int)

        candidate_moves = []
        for i in bad_clients:
            j0 = assign[i]
            if j0 < 0 or np.isnan(centers[j0]).any():
                continue
            compat_i = compat[i, :].copy()
            cap_ok = (loads1 + d1[i] <= c1 + 1e-9) & (loads2 + d2[i] <= c2 + 1e-9)
            mask = compat_i & cap_ok
            mask[j0] = False
            feasible_js = np.where(mask)[0]
            if feasible_js.size == 0:
                continue
            dlist = []
            for l in feasible_js:
                if np.isnan(centers[l]).any():
                    continue
                dlist.append((l, haversine(pts[i], centers[l])))
            if not dlist:
                continue
            l_best, d_best = min(dlist, key=lambda t: t[1])
            candidate_moves.append((i, l_best, d_best))

        if not candidate_moves:
            return assign, centers

        candidate_moves.sort(key=lambda t: t[2])

        moved = np.zeros(n, dtype=bool)
        for (i, l) in [(t[0], t[1]) for t in candidate_moves]:
            if moved[i]:
                continue
            j = assign[i]
            if j == l or j < 0:
                continue
            if np.isnan(centers[l]).any():
                continue
            if not compat[i, l]:
                continue
            if _capacity_ok(loads1[l] + d1[i], loads2[l] + d2[i], c1[l], c2[l]):
                assign[i] = l
                loads1[j] -= d1[i]; loads2[j] -= d2[i]
                loads1[l] += d1[i]; loads2[l] += d2[i]
                moved[i] = True

        centers = update_centroids_geomed(pts, assign, k, old_centers=centers)
        return assign, centers

    def _sa_single_run(run_seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
        rng = np.random.default_rng(run_seed)
        assign, centers = _greedy_init(rng)
        assign, centers = _greedy_compact_refine(assign, centers)

        best_assign = assign.copy()
        best_centers = centers.copy()
        best_cost = _cost(assign, centers)

        assign, centers = _local_refine(assign, centers, rng)
        cur_cost = _cost(assign, centers)
        if cur_cost < best_cost:
            best_cost, best_assign, best_centers = cur_cost, assign.copy(), centers.copy()

        T = sa_T0
        total_steps = sa_iters
        base_neighbors = neighbor_base if neighbor_base is not None else max(64, int(np.sqrt(max(1, n))))
        tabu = set()
        tabu_queue: List[tuple] = []

        def _push_tabu(key: tuple):
            tabu.add(key)
            tabu_queue.append(key)
            if len(tabu_queue) > tabu_horizon:
                old = tabu_queue.pop(0)
                if old in tabu:
                    tabu.remove(old)

        loads1, loads2 = _recompute_loads(assign)
        reheats_left = max(0, int(reheats))

        for step in range(total_steps):
            prog = step / max(1, total_steps - 1)
            T = sa_T0 * (sa_Tf / sa_T0) ** prog
            neigh_size = max(16, int(base_neighbors * max(0.1, T / sa_T0)))

            accepted = False

            for _ in range(neigh_size):
                if rng.random() < 0.6:
                    idx_assigned = np.where(assign >= 0)[0]
                    if idx_assigned.size == 0:
                        continue
                    i = int(rng.choice(idx_assigned))
                    j = int(assign[i])
                    l = int(rng.integers(0, k))
                    if l == j or not compat[i, l]:
                        continue
                    key = ("mv", i, j, l)
                    if key in tabu:
                        continue
                    if not _capacity_ok(loads1[l] + d1[i], loads2[l] + d2[i], c1[l], c2[l]):
                        continue
                    if np.isnan(centers[j]).any() or np.isnan(centers[l]).any():
                        continue
                    cur = haversine(pts[i], centers[j])
                    new = haversine(pts[i], centers[l])
                    delta = new - cur
                    if delta < 0 or rng.random() < np.exp(-delta / max(1e-9, T)):
                        assign[i] = l
                        loads1[j] -= d1[i]; loads2[j] -= d2[i]
                        loads1[l] += d1[i]; loads2[l] += d2[i]
                        _push_tabu(key)
                        accepted = True
                        break
                else:
                    idx_assigned = np.where(assign >= 0)[0]
                    if idx_assigned.size < 2:
                        continue
                    i, t = rng.choice(idx_assigned, size=2, replace=False)
                    j1, j2 = int(assign[i]), int(assign[t])
                    if j1 == j2 or j1 < 0 or j2 < 0:
                        continue
                    if not compat[i, j2] or not compat[t, j1]:
                        continue
                    key = ("sw", i, t, j1, j2)
                    if key in tabu:
                        continue
                    if not _capacity_ok(loads1[j1] - d1[i] + d1[t], loads2[j1] - d2[i] + d2[t], c1[j1], c2[j1]):
                        continue
                    if not _capacity_ok(loads1[j2] - d1[t] + d1[i], loads2[j2] - d2[t] + d2[i], c1[j2], c2[j2]):
                        continue
                    if np.isnan(centers[j1]).any() or np.isnan(centers[j2]).any():
                        continue
                    cur = haversine(pts[i], centers[j1]) + haversine(pts[t], centers[j2])
                    new = haversine(pts[i], centers[j2]) + haversine(pts[t], centers[j1])
                    delta = new - cur
                    if delta < 0 or rng.random() < np.exp(-delta / max(1e-9, T)):
                        assign[i], assign[t] = j2, j1
                        loads1[j1] = loads1[j1] - d1[i] + d1[t]
                        loads2[j1] = loads2[j1] - d2[i] + d2[t]
                        loads1[j2] = loads1[j2] - d1[t] + d1[i]
                        loads2[j2] = loads2[j2] - d2[t] + d2[i]
                        _push_tabu(key)
                        accepted = True
                        break

            if accepted and (step % 20 == 0 or step == total_steps - 1):
                centers = update_centroids_geomed(pts, assign, k)

            if step % 20 == 0 or step == total_steps - 1:
                cur_cost = _cost(assign, centers)
                if cur_cost < best_cost - 1e-6:
                    best_cost = cur_cost
                    best_assign = assign.copy()
                    best_centers = centers.copy()

            if (reheats_left > 0) and (step > 0) and (step % max(1, total_steps // (reheats + 1)) == 0):
                reheats_left -= 1
                T = max(T, sa_T0 * 0.5)

        assign, centers = _local_refine(best_assign.copy(), best_centers.copy(), rng)
        assign, centers = _greedy_compact_refine(assign, centers)

        final_cost = _cost(assign, centers)
        if final_cost < best_cost:
            best_cost = final_cost
            best_assign = assign
            best_centers = centers

        return best_assign, best_centers, best_cost

    seeds = [int(seed + 101 * s) for s in range(n_starts)]
    if _JOBLIB and n_starts > 1:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_sa_single_run)(s) for s in seeds)
    else:
        results = [_sa_single_run(s) for s in seeds]

    best = min(results, key=lambda t: t[2])
    return best


def refine_fixed_clusters_tb_mip_fast(
    clients: pd.DataFrame,
    clusters: pd.DataFrame,
    compat: np.ndarray,
    assign_init: np.ndarray,
    centers_fixed: np.ndarray,
    *,
    penalty_unassigned: float = 1000.0,
    max_iter_tb: int = 1000,
    mip_time_limit: int = 10,
    seed: int = 42,
    relocate_sample_frac: float = 0.35,
    knn_neighbors: int = 100,
    mip_topk: int = 3,
    candidate_topk: int = 5
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Refino com clusters fixos (rápido)
    """
    rng = np.random.default_rng(seed)
    pts = stack_latlon(clients).astype(float)
    n, k = len(clients), len(clusters)

    d1 = clients["demanda1"].to_numpy(float)
    d2 = clients["demanda2"].to_numpy(float)
    c1 = clusters["capacidade1"].to_numpy(float)
    c2 = clusters["capacidade2"].to_numpy(float)

    def _haversine_matrix(pts_xy: np.ndarray, centers_xy: np.ndarray) -> np.ndarray:
        R = 6371.0088
        lat1 = np.radians(pts_xy[:, 0])[:, None]
        lon1 = np.radians(pts_xy[:, 1])[:, None]
        lat2 = np.radians(centers_xy[:, 0])[None, :]
        lon2 = np.radians(centers_xy[:, 1])[None, :]
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
        return 2.0 * R * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))

    dist_ck = _haversine_matrix(pts, centers_fixed)
    centers_rank_per_client = np.argsort(dist_ck, axis=1)

    try:
        from scipy.spatial import cKDTree as KDTree
        _KD_OK = True
        tree = KDTree(pts)
    except Exception:
        _KD_OK = False
        tree = None

    def _knn_indices(i: int, m: int) -> np.ndarray:
        if _KD_OK:
            _, idx = tree.query(pts[i], k=min(m + 1, n))
            idx = np.atleast_1d(idx)
            idx = idx[idx != i]
            return idx[:m]
        vec = pts - pts[i]
        sq = np.einsum('ij,ij->i', vec, vec)
        order = np.argsort(sq)
        order = order[order != i]
        return order[:m]

    assign = assign_init.copy()
    loads1 = np.zeros(k, dtype=float)
    loads2 = np.zeros(k, dtype=float)
    for j in range(k):
        idx = np.where(assign == j)[0]
        if idx.size:
            loads1[j] = d1[idx].sum()
            loads2[j] = d2[idx].sum()

    def _capacity_ok(l1, l2, C1, C2) -> bool:
        return (l1 <= C1 + 1e-9) and (l2 <= C2 + 1e-9)

    sample_size = max(1, int(np.ceil(relocate_sample_frac * n)))
    cand_topk = int(max(1, min(candidate_topk, k)))
    knn_nei = int(max(1, min(knn_neighbors, n - 1)))

    it = 0
    improved = True
    while improved and it < max_iter_tb:
        it += 1
        improved = False

        idx_assigned = np.where(assign >= 0)[0]
        if idx_assigned.size > 0:
            sample = rng.choice(idx_assigned, size=min(sample_size, idx_assigned.size), replace=False)
            for i in sample:
                j = assign[i]
                if j < 0:
                    continue
                cur_cost = dist_ck[i, j]
                cand_centers = centers_rank_per_client[i][:cand_topk]
                best_delta = 0.0
                best_l = -1
                for l in cand_centers:
                    if l == j:
                        continue
                    if not compat[i, l]:
                        continue
                    if not _capacity_ok(loads1[l] + d1[i], loads2[l] + d2[i], c1[l], c2[l]):
                        continue
                    new_cost = dist_ck[i, l]
                    delta = cur_cost - new_cost
                    if delta > best_delta + 1e-9:
                        best_delta = delta
                        best_l = l
                if best_l >= 0:
                    assign[i] = best_l
                    loads1[j] -= d1[i]; loads2[j] -= d2[i]
                    loads1[best_l] += d1[i]; loads2[best_l] += d2[i]
                    improved = True

        idx_assigned = np.where(assign >= 0)[0]
        if idx_assigned.size >= 2:
            sample = rng.choice(idx_assigned, size=min(sample_size, idx_assigned.size), replace=False)
            for i in sample:
                j1 = assign[i]
                if j1 < 0:
                    continue
                neigh = _knn_indices(i, knn_nei)
                for t in neigh:
                    if t <= i:
                        continue
                    j2 = assign[t]
                    if j2 < 0 or j2 == j1:
                        continue
                    if (not compat[i, j2]) or (not compat[t, j1]):
                        continue
                    if not _capacity_ok(loads1[j1] - d1[i] + d1[t], loads2[j1] - d2[i] + d2[t], c1[j1], c2[j1]):
                        continue
                    if not _capacity_ok(loads1[j2] - d1[t] + d1[i], loads2[j2] - d2[t] + d2[i], c1[j2], c2[j2]):
                        continue
                    cur = dist_ck[i, j1] + dist_ck[t, j2]
                    new = dist_ck[i, j2] + dist_ck[t, j1]
                    if new + 1e-9 < cur:
                        assign[i], assign[t] = j2, j1
                        loads1[j1] = loads1[j1] - d1[i] + d1[t]
                        loads2[j1] = loads2[j1] - d2[i] + d2[t]
                        loads1[j2] = loads1[j2] - d1[t] + d1[i]
                        loads2[j2] = loads2[j2] - d2[t] + d2[i]
                        improved = True

    # MINI-MIP para não alocados
    unassigned = np.where(assign < 0)[0]
    if unassigned.size > 0:
        try:
            from ortools.linear_solver import pywraplp
            solver = pywraplp.Solver.CreateSolver("CBC")

            cand_per_i = {}
            for i in unassigned:
                ranked = centers_rank_per_client[i]
                feasible = [j for j in ranked if compat[i, j]][:max(1, mip_topk)]
                cand_per_i[i] = feasible

            x = {}
            u = {}
            for i in unassigned:
                u[i] = solver.IntVar(0, 1, f"u_{i}")
                for j in cand_per_i[i]:
                    x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

            for i in unassigned:
                vars_i = [x[i, j] for j in cand_per_i[i]] if cand_per_i[i] else []
                solver.Add(sum(vars_i) + u[i] == 1)

            for j in range(k):
                lhs1 = sum(d1[i] * x[i, j] for i in unassigned if j in cand_per_i[i])
                lhs2 = sum(d2[i] * x[i, j] for i in unassigned if j in cand_per_i[i])
                solver.Add(lhs1 + loads1[j] <= c1[j])
                solver.Add(lhs2 + loads2[j] <= c2[j])

            obj_terms = []
            for i in unassigned:
                for j in cand_per_i[i]:
                    obj_terms.append(dist_ck[i, j] * x[i, j])
                obj_terms.append(penalty_unassigned * u[i])

            solver.Minimize(solver.Sum(obj_terms))
            solver.SetTimeLimit(int(1000 * min(max(1, mip_time_limit), 5)))
            status = solver.Solve()

            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                for i in unassigned:
                    for j in cand_per_i[i]:
                        if x[i, j].solution_value() > 0.5:
                            assign[i] = j
                            loads1[j] += d1[i]; loads2[j] += d2[i]
                            break
            else:
                for i in unassigned:
                    best_j, best_d = -1, float("inf")
                    for j in cand_per_i[i]:
                        if _capacity_ok(loads1[j] + d1[i], loads2[j] + d2[i], c1[j], c2[j]):
                            dd = dist_ck[i, j]
                            if dd < best_d:
                                best_d, best_j = dd, j
                    if best_j >= 0:
                        assign[i] = best_j
                        loads1[best_j] += d1[i]; loads2[best_j] += d2[i]
        except Exception:
            for i in unassigned:
                ranked = centers_rank_per_client[i]
                cand = [j for j in ranked if compat[i, j]][:max(1, mip_topk)]
                best_j, best_d = -1, float("inf")
                for j in cand:
                    if _capacity_ok(loads1[j] + d1[i], loads2[j] + d2[i], c1[j], c2[j]):
                        dd = dist_ck[i, j]
                        if dd < best_d:
                            best_d, best_j = dd, j
                if best_j >= 0:
                    assign[i] = best_j
                    loads1[best_j] += d1[i]; loads2[best_j] += d2[i]

    total_cost = 0.0
    for i, j in enumerate(assign):
        if j >= 0 and not np.isnan(centers_fixed[j]).any():
            total_cost += dist_ck[i, j]

    print(f"[FixedClusters+TB+MIP_fast] Obj={total_cost:.2f} | Alocados={(assign>=0).sum()}/{n}")
    return assign, centers_fixed, total_cost


def hybrid_greedy_sa_tb_mip_fast(
    clients: pd.DataFrame,
    clusters: pd.DataFrame,
    compat: np.ndarray,
    *,
    seed: int = 42,
    sa_iters: int = 800,
    sa_T0: float = 1.0,
    sa_Tf: float = 1e-3,
    reheats: int = 2,
    n_starts: int = 8,
    neighbor_base: int | None = None,
    tabu_horizon: int = 50,
    n_jobs: int = -1,
    penalty_unassigned: float = 1000.0,
    max_iter_tb: int = 1000,
    mip_time_limit: int = 10,
    relocate_sample_frac: float = 0.35,
    knn_neighbors: int = 20,
    mip_topk: int = 10,
    candidate_topk: int = 10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Pipeline híbrido: Greedy+SA → FixedTB+MIP
    """
    assign_gtsa, centers_gtsa, obj_gtsa = greedy_teitz_bart_sa_fast(
        clients, clusters, compat,
        seed=seed,
        sa_iters=sa_iters,
        sa_T0=sa_T0,
        sa_Tf=sa_Tf,
        reheats=reheats,
        n_starts=n_starts,
        neighbor_base=neighbor_base,
        tabu_horizon=tabu_horizon,
        n_jobs=n_jobs
    )

    assign_final, centers_final, obj_final = refine_fixed_clusters_tb_mip_fast(
        clients, clusters, compat,
        assign_init=assign_gtsa,
        centers_fixed=centers_gtsa,
        penalty_unassigned=penalty_unassigned,
        max_iter_tb=max_iter_tb,
        mip_time_limit=mip_time_limit,
        seed=seed,
        relocate_sample_frac=relocate_sample_frac,
        knn_neighbors=knn_neighbors,
        mip_topk=mip_topk,
        candidate_topk=candidate_topk
    )

    print(f"[Hybrid Greedy+SA → FixedTB+MIP_fast] Obj={obj_final:.2f} | "
          f"Alocados={(assign_final >= 0).sum()}/{len(clients)}")

    return assign_final, centers_final, obj_final


def build_compat_matrix(clients_skills: List, clusters_req: List) -> np.ndarray:
    n, k = len(clients_skills), len(clusters_req)
    M = np.zeros((n, k), dtype=bool)
    for i in range(n):
        si = set(clients_skills[i])  # garante set
        for j in range(k):
            req = set(clusters_req[j])
            M[i, j] = req.issubset(si)
    return M


def sanitize(obj):
    """Recursivamente converte NaN/inf → None e np types → nativos"""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, set):
        return [sanitize(v) for v in list(obj)]
    elif isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


# =====================================================
# =================== MODELOS API =====================
# =====================================================

class Client(BaseModel):
    id: str
    nome: str
    lat: float
    lon: float
    demanda1: float
    demanda2: float
    skills: Optional[List[str]] = []


class Cluster(BaseModel):
    id: str
    capacidade1: float
    capacidade2: float
    skills_requeridas: Optional[List[str]] = []
    lat: Optional[float] = None
    lon: Optional[float] = None


class SolveRequest(BaseModel):
    algoritmo: str
    clients: List[Client]
    clusters: List[Cluster]


# =====================================================
# =================== FASTAPI APP =====================
# =====================================================

app = FastAPI(title="Clustering Capacitado API", version="1.0.0")

# (opcional) CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # ajuste para domínios específicos em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


security = HTTPBearer()
API_TOKEN = os.getenv("API_TOKEN", "meu_token_supersecreto")  # defina no Heroku Config Vars

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if not hmac.compare_digest(token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Token inválido ou ausente")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve")
def solve(req: SolveRequest, _=Depends(verify_token)):
    if req.token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")

    # DataFrames
    clients_df = pd.DataFrame([c.dict() for c in req.clients])
    clusters_df = pd.DataFrame([c.dict() for c in req.clusters])

    # Sets para skills
    clients_df["skills"] = clients_df["skills"].apply(lambda x: set(x or []))
    clusters_df["skills_requeridas"] = clusters_df["skills_requeridas"].apply(lambda x: set(x or []))

    # Matriz de compatibilidade
    compat = build_compat_matrix(
        clients_df["skills"].tolist(),
        clusters_df["skills_requeridas"].tolist()
    )

    t0 = time.perf_counter()
    algoritmo = req.algoritmo

    if algoritmo == "heuristic":
        assign, centers, obj = heuristic_kmeans_capacitado(clients_df, clusters_df, compat)
    elif algoritmo == "grasp_vnd":
        assign, centers, obj = solve_grasp_vnd_greedy_fixedK(clients_df, clusters_df, compat)
    elif algoritmo == "hybrid":
        assign, centers, obj = hybrid_greedy_sa_tb_mip_fast(clients_df, clusters_df, compat)
    elif algoritmo == "refine_tb":
        if "lat" not in clusters_df or "lon" not in clusters_df:
            raise HTTPException(status_code=400, detail="Clusters precisam ter lat/lon no JSON")
        centers_fixed = clusters_df[["lat", "lon"]].to_numpy(float)
        assign, centers, obj = refine_fixed_clusters_tb_mip_fast(
            clients_df, clusters_df, compat,
            assign_init=np.full(len(clients_df), -1, dtype=int),
            centers_fixed=centers_fixed
        )
    else:
        raise HTTPException(status_code=400, detail="Algoritmo inválido")

    t1 = time.perf_counter()

    # Monta resposta
    clusters_out = []
    for j, row in clusters_df.iterrows():
        members = [clients_df.iloc[i]["id"] for i in range(len(assign)) if assign[i] == j]
        lat, lon = None, None
        if j < len(centers):
            if not np.isnan(centers[j][0]): lat = float(centers[j][0])
            if not np.isnan(centers[j][1]): lon = float(centers[j][1])
        clusters_out.append({
            "id": row["id"],
            "capacidade1": row["capacidade1"],
            "capacidade2": row["capacidade2"],
            "lat": lat,
            "lon": lon,
            "clientes_associados": members
        })

    clientes_out = []
    for i, row in clients_df.iterrows():
        clientes_out.append({
            "id": row["id"], "nome": row["nome"],
            "lat": row["lat"], "lon": row["lon"],
            "demanda1": row["demanda1"], "demanda2": row["demanda2"],
            "skills": list(row["skills"]),
            "cluster_atribuido": int(assign[i]) if assign[i] >= 0 else None
        })

    centros_ativos = sum(1 for c in clusters_out if len(c["clientes_associados"]) > 0)

    payload = sanitize({
        "algoritmo": algoritmo,
        "obj": obj,
        "tempo": t1 - t0,
        "alocados": int((assign >= 0).sum()),
        "centros": centros_ativos,
        "clusters": clusters_out,
        "clientes": clientes_out
    })
    return JSONResponse(content=payload)


# Execução local:
# uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
