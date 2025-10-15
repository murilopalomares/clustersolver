# uc_kmeans_hibrido.py
import pandas as pd
import numpy as np
import random, os, webbrowser, folium
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import time

# =====================
# Erros
# =====================
class SchemaError(ValueError): pass
class SolverError(RuntimeError): pass

# =====================
# OR-Tools Solver
# =====================
try:
    from ortools.linear_solver import pywraplp
    _ORTOOLS_AVAILABLE = True
except Exception:
    _ORTOOLS_AVAILABLE = False

@dataclass(frozen=True)
class ParsedData:
    clients_df: pd.DataFrame
    clusters_df: pd.DataFrame

def _as_set_of_str(x: Any) -> Set[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return set()
    if isinstance(x, (list, set, tuple)):
        return {str(s).strip() for s in x if str(s).strip()}
    if isinstance(x, str):
        return {s.strip() for s in x.split(",") if s.strip()}
    return {str(x)}

def parse_clusters_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["id", "capacidade1", "capacidade2", "skills_requeridas"]
    for col in required:
        if col not in df.columns:
            raise SchemaError(f"Coluna obrigatória ausente em clusters: {col}")

    # skills_requeridas como set
    df["skills_requeridas"] = df["skills_requeridas"].apply(_as_set_of_str)

    # tratar NaNs em capacidades
    df["capacidade1"] = pd.to_numeric(df["capacidade1"], errors="coerce").fillna(0.0)
    df["capacidade2"] = pd.to_numeric(df["capacidade2"], errors="coerce").fillna(0.0)

    return df

def stack_latlon(df: pd.DataFrame) -> np.ndarray:
    return df[["lat", "lon"]].astype("float64").to_numpy(copy=True)

def haversine(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
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
            # um ponto muito perto -> esse ponto
            return pts[np.argmin(d)]
        w = 1.0 / d
        x_new = (w[:, None] * pts).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < eps:
            return x_new
        x = x_new
    return x

def update_centroids_geomed(points_latlon: np.ndarray, assign: np.ndarray, k: int, old_centers: Optional[np.ndarray]=None) -> np.ndarray:
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

def kmeanspp_init(points: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """KMeans++ clássico, D^2 sampling, cobertura melhor de espaço."""
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    centers = np.empty((k, 2), dtype=float)
    # 1) escolhe primeiro centro aleatório
    centers[0] = points[rng.integers(0, n)]
    # 2) demais pelo D^2
    d2 = np.full(n, np.inf, dtype=float)
    for c in range(1, k):
        d2 = np.minimum(d2, np.linalg.norm(points - centers[c-1], axis=1)**2)
        probs = d2 / d2.sum()
        idx = rng.choice(n, p=probs)
        centers[c] = points[idx]
    return centers

def kmeanspp_geo_coverage(points: np.ndarray, k: int, seed: int = 42, coverage_boost: float = 0.25) -> np.ndarray:
    """
    KMeans++ com leve viés para cobertura geográfica:
    mistura D^2 com farthest-point sampling.
    """
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    centers = np.empty((k, 2), dtype=float)
    centers[0] = points[rng.integers(0, n)]
    d2 = np.full(n, np.inf, dtype=float)
    for c in range(1, k):
        # update D^2
        d2 = np.minimum(d2, np.linalg.norm(points - centers[c-1], axis=1)**2)
        # farthest mask
        farthest_idx = np.argsort(d2)[-max(1, int(coverage_boost*n/k)):]  # top mais distantes
        # mistura: 50% escolhe do farthest, 50% por probabilidade D^2
        if rng.random() < 0.5 and len(farthest_idx) > 0:
            idx = rng.choice(farthest_idx)
        else:
            probs = d2 / (d2.sum() if d2.sum() > 0 else 1.0)
            idx = rng.choice(n, p=probs)
        centers[c] = points[idx]
    return centers

def parse_clients_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["id","nome","lat","lon","demanda1","demanda2","skills"]
    for col in required:
        if col not in df.columns:
            raise SchemaError(f"Coluna obrigatória ausente: {col}")

    # skills como conjunto de strings
    df["skills"] = df["skills"].apply(_as_set_of_str)

    # Tratamento NaN em demandas → 0
    df["demanda1"] = pd.to_numeric(df["demanda1"], errors="coerce").fillna(0.0)
    df["demanda2"] = pd.to_numeric(df["demanda2"], errors="coerce").fillna(0.0)

    return df

def build_compat_matrix(clients_skills: List, clusters_req: List) -> np.ndarray:
    n, k = len(clients_skills), len(clusters_req)
    M = np.zeros((n, k), dtype=bool)
    for i in range(n):
        si = set(clients_skills[i])  # garante set
        for j in range(k):
            req = set(clusters_req[j])
            M[i, j] = req.issubset(si)
    return M

def make_map(clients_df, centroids, assign, title, html_path):
    pts = stack_latlon(clients_df)
    m = folium.Map(location=[float(pts[:, 0].mean()), float(pts[:, 1].mean())], zoom_start=10)

    base_colors = [
        "red", "blue", "green", "purple", "orange", "darkred", "lightred",
        "beige", "darkblue", "darkgreen", "cadetblue", "darkpurple",
        "pink", "lightblue", "lightgreen", "gray", "black"
    ]
    k = centroids.shape[0]
    cluster_colors = {j: base_colors[j % len(base_colors)] for j in range(k)}

    # clientes + linhas
    for i, row in clients_df.iterrows():
        j = assign[i] if assign is not None else -1
        color = cluster_colors[j] if j >= 0 else "gray"
        folium.CircleMarker(
            [row["lat"], row["lon"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"{row['id']} ({row['demanda1']},{row['demanda2']})"
        ).add_to(m)

        if j >= 0 and not np.isnan(centroids[j]).any():
            folium.PolyLine(
                [[row["lat"], row["lon"]], [centroids[j, 0], centroids[j, 1]]],
                color=color,
                weight=2,
                opacity=0.6
            ).add_to(m)

    # centros
    for j, c in enumerate(centroids):
        if not np.isnan(c).any():
            color = cluster_colors[j]
            folium.Marker(
                [c[0], c[1]],
                tooltip=f"Centro {j+1}",
                icon=folium.Icon(color=color, icon="star", prefix="fa")
            ).add_to(m)

    m.save(html_path)
    webbrowser.open("file://" + os.path.abspath(html_path))
    print(f"[OK] {title} -> {html_path}")

def _capacity_ok(load1, load2, c1, c2) -> bool:
    return (load1 <= c1 + 1e-9) and (load2 <= c2 + 1e-9)

def _compute_cost(pts: np.ndarray, assign: np.ndarray, centers: np.ndarray) -> float:
    total = 0.0
    for i, j in enumerate(assign):
        if j >= 0 and not np.isnan(centers[j]).any():
            total += haversine(pts[i], centers[j])
    return total

def heuristic_kmeans_capacitado(clients, clusters, compat, max_iter=30, seed: int = 42,
                                ls_max_iter: int = 30, repair_rounds: int = 2):
    """
    Heurística K-Means Capacitada otimizada:
      - Vetorização de distâncias
      - Pré-filtragem de compatibilidade
      - np.bincount para cargas
      - argpartition em vez de argsort
    """
    pts = stack_latlon(clients)
    n, k = len(clients), len(clusters)
    rng = np.random.default_rng(seed)

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

        # ----- reparo: tentar inserir não alocados empurrando pequenos -----
        not_assigned = np.where(new_assign < 0)[0].tolist()
        for _ in range(repair_rounds):
            if not not_assigned:
                break
            still = []
            for i in not_assigned:
                inserted = False
                # candidatos: clusters compatíveis ordenados por distância
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
            break

        assign, centers = new_assign, new_centers

    # ----- busca local: relocate/swap -----
    loads1 = np.bincount(assign[assign >= 0], weights=d1[assign >= 0], minlength=k)
    loads2 = np.bincount(assign[assign >= 0], weights=d2[assign >= 0], minlength=k)

    improved = True; it_ls = 0
    while improved and it_ls < ls_max_iter:
        improved = False; it_ls += 1

        # --- relocate ---
        for i in range(n):
            j = assign[i]
            if j < 0: continue
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
                # aplica move
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
            if j1 < 0: continue
            for t in range(i+1, n):
                j2 = assign[t]
                if j2 < 0 or j2 == j1: continue
                if j2 not in valid_clusters[i] or j1 not in valid_clusters[t]:
                    continue
                # checar capacidade pós-swap
                if _capacity_ok(loads1[j1] - d1[i] + d1[t], loads2[j1] - d2[i] + d2[t], c1[j1], c2[j1]) and \
                   _capacity_ok(loads1[j2] - d1[t] + d1[i], loads2[j2] - d2[t] + d2[i], c1[j2], c2[j2]):
                    cur = dist_matrix[i, j1] + dist_matrix[t, j2]
                    new = dist_matrix[i, j2] + dist_matrix[t, j1]
                    if new + 1e-6 < cur:
                        # aplica swap
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
    print(f"[Heurístico Capacitado] Obj={total_cost:.2f} | it={it+1} | LS={it_ls} | Alocados={alocados}/{n}")
    return assign, centers, total_cost

def heuristic_kmeans_capacitado_revivendoclusters(
    clients, clusters, compat,
    max_iter=30, seed: int = 42,
    ls_max_iter: int = 30, repair_rounds: int = 2
):
    """
    Heurística K-Means Capacitado (versão robusta):
      - Vetorização de distâncias
      - Pré-filtragem de compatibilidade
      - np.bincount para cargas
      - argpartition em vez de argsort
      - Reativação automática de clusters vazios (garante k clusters)
    """
    pts = stack_latlon(clients)
    n, k = len(clients), len(clusters)
    rng = np.random.default_rng(seed)

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
        dist_matrix = haversine_matrix(pts, centers)

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

        # ----- reparo: tentar inserir não alocados empurrando pequenos -----
        not_assigned = np.where(new_assign < 0)[0].tolist()
        for _ in range(repair_rounds):
            if not not_assigned:
                break
            still = []
            for i in not_assigned:
                inserted = False
                # candidatos: clusters compatíveis ordenados por distância
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

        # ====== NOVO BLOCO: reativar clusters vazios ======
        new_centers = update_centroids_geomed(pts, new_assign, k, old_centers=centers)

        empty_clusters = np.where(np.isnan(new_centers).any(axis=1))[0]
        if len(empty_clusters) > 0:
            # pega clientes mais distantes de seus clusters atuais
            dist_to_center = np.array([
                dist_matrix[i, new_assign[i]] if new_assign[i] >= 0 else 1e9 for i in range(n)
            ])
            farthest_clients = np.argsort(dist_to_center)[::-1]
            # realoca um cliente distante para cada cluster vazio
            for cid, i in zip(empty_clusters, farthest_clients):
                new_assign[i] = cid
                new_centers[cid] = pts[i]
                loads1[cid] = d1[i]; loads2[cid] = d2[i]
            print(f"[Iter {it}] {len(empty_clusters)} clusters reativados.")

        # critério de parada
        if np.all(assign == new_assign):
            break

        assign, centers = new_assign, new_centers

    # ----- busca local: relocate/swap -----
    loads1 = np.bincount(assign[assign >= 0], weights=d1[assign >= 0], minlength=k)
    loads2 = np.bincount(assign[assign >= 0], weights=d2[assign >= 0], minlength=k)

    improved = True; it_ls = 0
    while improved and it_ls < ls_max_iter:
        improved = False; it_ls += 1

        # --- relocate ---
        for i in range(n):
            j = assign[i]
            if j < 0: continue
            best_gain = 0.0; best_l = -1
            for l in valid_clusters[i]:
                if l == j or np.isnan(centers[l]).any():
                    continue
                if _capacity_ok(loads1[l] + d1[i], loads2[l] + d2[i], c1[l], c2[l]):
                    cur = haversine(pts[i], centers[j])
                    new = haversine(pts[i], centers[l])
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

        # --- swap simples ---
        for i in range(n):
            j1 = assign[i]
            if j1 < 0: continue
            for t in range(i + 1, n):
                j2 = assign[t]
                if j2 < 0 or j2 == j1: continue
                if j2 not in valid_clusters[i] or j1 not in valid_clusters[t]:
                    continue
                if _capacity_ok(loads1[j1] - d1[i] + d1[t], loads2[j1] - d2[i] + d2[t], c1[j1], c2[j1]) and \
                   _capacity_ok(loads1[j2] - d1[t] + d1[i], loads2[j2] - d2[t] + d2[i], c1[j2], c2[j2]):
                    cur = haversine(pts[i], centers[j1]) + haversine(pts[t], centers[j2])
                    new = haversine(pts[i], centers[j2]) + haversine(pts[t], centers[j1])
                    if new + 1e-6 < cur:
                        loads1[j1] = loads1[j1] - d1[i] + d1[t]
                        loads2[j1] = loads2[j1] - d2[i] + d2[t]
                        loads1[j2] = loads1[j2] - d1[t] + d1[i]
                        loads2[j2] = loads2[j2] - d2[t] + d2[i]
                        assign[i], assign[t] = j2, j1
                        improved = True
        if improved:
            centers = update_centroids_geomed(pts, assign, k, old_centers=centers)

    total_cost = _compute_cost(pts, assign, centers)
    alocados = int(np.sum(assign >= 0))
    ativos = len(np.unique(assign[assign >= 0]))

    print(f"[Heurístico Capacitado] Obj={total_cost:.2f} | it={it+1} | LS={it_ls} | Alocados={alocados}/{n} | Clusters Ativos={ativos}/{k}")
    return assign, centers, total_cost

def haversine_matrix(pts, centers):
    """Matriz [n x k] de distâncias haversine vetorizadas."""
    lat1, lon1 = np.radians(pts[:, 0])[:, None], np.radians(pts[:, 1])[:, None]
    lat2, lon2 = np.radians(centers[:, 0])[None, :], np.radians(centers[:, 1])[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))  # km

def solve_grasp_vnd_greedy_fixedK(
    clients: pd.DataFrame,
    clusters: pd.DataFrame,
    compat: np.ndarray,
    *,
    alpha: float = 0.25,        # GRASP (0 = guloso puro; 0.2–0.4 dá diversidade)
    multi_starts: int = 6,      # recomeços independentes
    seed: int = 42,
    repair_rounds: int = 2,     # reparo de não alocados
    vnd_max_iter: int = 25,     # iterações VND
    relocate_k: int = 6,        # só checar k centros mais próximos no relocate
    swap_pairs: int = 64,       # número de pares amostrados no swap por iteração
    sample_candidates: int = 128, # amostra de candidatos por cluster na construção
    tol_improv: float = 1e-6,   # tolerância numérica
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    GRASP + VND com K FIXO (sem apagar clusters)
    --------------------------------------------
    Objetivo: manter exatamente k clusters ativos, respeitar compatibilidade e capacidades,
    otimizar posições dos centros (mediana geográfica) e as atribuições de clientes.

    Diagnóstico (mudanças vs. versão anterior):
      • Antes a construção podia deixar clusters vazios; agora garantimos ≥1 cliente/cluster
        com um passo 'enforce_nonempty' (puxando cliente de outro cluster com menor custo
        incremental e respeitando capacidade).
      • Mantemos os centros livres para mover (mediana geográfica), mas nunca desativamos
        um cluster.
      • Busca local (VND) limitada por vizinhança próxima e amostragem para eficiência.

    Retorna: (assign, centers, obj)
    """
    rng_global = np.random.default_rng(seed)

    # --- dados base
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
            # se centro for NaN, usa baricentro global como referência temporária
            if np.isnan(centers[j]).any():
                centers[j] = pts.mean(axis=0)

            # escolhe cliente para mover de outro cluster para j
            best_i, best_src, best_delta = -1, -1, float("inf")
            for i in range(n):
                src = assign[i]
                if src == -1 or src == j:
                    continue
                if not compat[i, j]:
                    continue
                # capacidade pós-move
                if not _capacity_ok(loads1[j] + d1[i], loads2[j] + d2[i], c1[j], c2[j]):
                    continue
                # liberar do cluster src (sempre possível ao remover)
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
                # aplica movimento
                assign[best_i] = j
                loads1[best_src] -= d1[best_i]; loads2[best_src] -= d2[best_i]
                loads1[j] += d1[best_i];       loads2[j] += d2[best_i]
                # atualiza centros de src e j
                centers = update_centroids_geomed(pts, assign, k, old_centers=centers)
            else:
                # não achou candidato viável → mantém como está (cluster vazio permanece, mas tentaremos de novo após VND)
                pass

        return assign, centers

    # -------------- CONSTRUÇÃO GULOSA (com RCL/alpha) --------------
    def greedy_construct(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        assign = np.full(n, -1, dtype=int)
        centers = np.full((k, 2), np.nan)
        used = np.zeros(n, dtype=bool)

        # ordem aleatória dos clusters para diversidade
        order_clusters = rng.permutation(k)

        def _seed_score(i, jj):
            return (d1[i] / max(c1[jj], 1e-9)) + (d2[i] / max(c2[jj], 1e-9))

        for jj in order_clusters:
            viable = np.where((~used) & compat[:, jj] & (d1 <= c1[jj]) & (d2 <= c2[jj]))[0]
            if viable.size == 0:
                continue

            # RCL de sementes
            scores = np.array([_seed_score(i, jj) for i in viable])
            order = np.argsort(-scores)
            rcl_len = max(1, int(np.ceil(alpha * len(order))))
            s_idx = viable[order[:rcl_len]]
            seed_i = rng.choice(s_idx)

            # inicia cluster jj com a semente
            assign[seed_i] = jj
            used[seed_i] = True
            load1 = d1[seed_i]
            load2 = d2[seed_i]
            members = [seed_i]
            center = pts[seed_i]

            # expansão gulosa (amostrada)
            while True:
                pool = np.where((~used) & compat[:, jj] &
                                (d1 + load1 <= c1[jj] + 1e-12) &
                                (d2 + load2 <= c2[jj] + 1e-12))[0]
                if pool.size == 0:
                    break
                m = min(sample_candidates, pool.size)
                cand = rng.choice(pool, size=m, replace=False)

                # escolhe o mais próximo do centro corrente
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

            # centro robusto final
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

        # ajuste de centros e garantia de não-vazio
        centers = update_centroids_geomed(pts, assign, k, old_centers=centers)
        assign, centers = enforce_nonempty(assign, centers)
        centers = update_centroids_geomed(pts, assign, k, old_centers=centers)
        return assign, centers

    # -------------- VND (Relocate + Swap) --------------
    def vnd_refine(assign: np.ndarray, centers: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        loads1, loads2 = _recompute_loads(assign)

        it = 0
        improved_global = True
        while improved_global and it < vnd_max_iter:
            it += 1
            improved_global = False

            # ---------- RELOCATE ----------
            for i in range(n):
                j = assign[i]
                if j < 0 or np.isnan(centers[j]).any():
                    continue

                # top-k centros mais próximos ao cliente i
                # (calcular distâncias rápidas só aos centros válidos)
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
                    # aplica move
                    loads1[j] -= d1[i]; loads2[j] -= d2[i]
                    loads1[best_l] += d1[i]; loads2[best_l] += d2[i]
                    assign[i] = best_l
                    improved_global = True

            if improved_global:
                centers = update_centroids_geomed(pts, assign, k, old_centers=centers)

            # ---------- SWAP (amostrado) ----------
            idx_assigned = np.where(assign >= 0)[0]
            if idx_assigned.size >= 2:
                m = min(swap_pairs, (idx_assigned.size * (idx_assigned.size - 1)) // 2)
                for _ in range(m):
                    i, t = rng.choice(idx_assigned, size=2, replace=False)
                    j1, j2 = assign[i], assign[t]
                    if j1 == j2 or j1 < 0 or j2 < 0:
                        continue
                    if (not compat[i, j2]) or (not compat[t, j1]):
                        continue
                    # capacidade pós-swap
                    if not _capacity_ok(loads1[j1] - d1[i] + d1[t], loads2[j1] - d2[i] + d2[t], c1[j1], c2[j1]):
                        continue
                    if not _capacity_ok(loads1[j2] - d1[t] + d1[i], loads2[j2] - d2[t] + d2[i], c1[j2], c2[j2]):
                        continue
                    if np.isnan(centers[j1]).any() or np.isnan(centers[j2]).any():
                        continue
                    cur = haversine(pts[i], centers[j1]) + haversine(pts[t], centers[j2])
                    new = haversine(pts[i], centers[j2]) + haversine(pts[t], centers[j1])
                    if cur - new > tol_improv:
                        # aplica swap
                        assign[i], assign[t] = j2, j1
                        loads1[j1] = loads1[j1] - d1[i] + d1[t]
                        loads2[j1] = loads2[j1] - d2[i] + d2[t]
                        loads1[j2] = loads1[j2] - d1[t] + d1[i]
                        loads2[j2] = loads2[j2] - d2[t] + d2[i]
                        improved_global = True

            if improved_global:
                centers = update_centroids_geomed(pts, assign, k, old_centers=centers)

            # reforça K fixo (nenhum vazio) após cada rodada
            if improved_global:
                assign, centers = enforce_nonempty(assign, centers)
                centers = update_centroids_geomed(pts, assign, k, old_centers=centers)

        return assign, centers

    # -------------- GRASP MULTI-START --------------
    best_assign, best_centers, best_obj = None, None, float("inf")
    for s in range(multi_starts):
        rng = np.random.default_rng(int(seed + 101*s))

        # Construção gulosa que já força todos os clusters ativos
        assign, centers = greedy_construct(rng)

        # VND (refino eficiente)
        assign, centers = vnd_refine(assign, centers, rng)

        # Garantia final de k não vazios
        assign, centers = enforce_nonempty(assign, centers)
        centers = update_centroids_geomed(pts, assign, k, old_centers=centers)

        # Custo
        obj = _cost(assign, centers)
        if obj < best_obj - tol_improv:
            best_obj = obj
            best_assign = assign.copy()
            best_centers = centers.copy()

    print(f"\n[GRASP-VND-Greedy(K-FIXO)] custo={best_obj:.2f} | alocados={(best_assign>=0).sum()}/{n} | clusters-ativos={sum(np.array([np.sum(best_assign==j)>0 for j in range(k)]))}/{k}")
    return best_assign, best_centers, best_obj

def greedy_teitz_bart_sa_fast(
    clients: pd.DataFrame,
    clusters: pd.DataFrame,
    compat: np.ndarray,
    *,
    seed: int = 42,
    # SA
    sa_iters: int = 1000,
    sa_T0: float = 1.0,
    sa_Tf: float = 1e-3,
    reheats: int = 2,
    # Multi-start
    n_starts: int = 8,
    # Vizinhança
    neighbor_base: int | None = None,   # se None => int(sqrt(n))
    # Tabu
    tabu_horizon: int = 100,
    # Paralelo
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Greedy + refino local (relocate/swap) + Simulated Annealing (paralelo e adaptativo) + 
    REFINO GULOSO DE COMPACTAÇÃO (aplicado pós-inicial e pré-retorno).
    - Respeita skills (compat), 2 demandas e capacidades.
    - Usa mediana geográfica para centros.
    - Multi-start em paralelo (joblib se disponível, senão fallback serial).
    Retorna: (assign, centers, custo)
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

    # ---------- helpers ----------
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
        """
        Inicialização ágil (variante do seu GreedyFAST).
        - Escolhe um 'semente' compatível por cluster
        - Vai adicionando clientes mais próximos ao centro corrente respeitando a capacidade
        """
        assign = np.full(n, -1, dtype=int)
        centers = np.full((k, 2), np.nan)
        used = np.zeros(n, dtype=bool)

        # ordem de clusters aleatória (diversificação multi-start)
        order_clusters = rng.permutation(k)

        for jj in order_clusters:
            # escolhe semente compatível e viável
            viable = np.where((compat[:, jj]) & (d1 <= c1[jj]) & (d2 <= c2[jj]) & (~used))[0]
            if viable.size == 0:
                continue
            seed_i = rng.choice(viable)
            assign[seed_i] = jj
            used[seed_i] = True

            L1, L2 = d1[seed_i], d2[seed_i]
            members = [seed_i]
            ctr = pts[seed_i]

            # lista de candidatos ordena por distância ao centro corrente (amostrada)
            improved = True
            while improved:
                improved = False
                best_i, best_dist = -1, float("inf")
                pool = np.where(~used & compat[:, jj])[0]
                if pool.size == 0:
                    break
                sample_sz = min(max(64, int(np.sqrt(pool.size))), pool.size)
                cand = rng.choice(pool, size=sample_sz, replace=False)
                # avalia candidatos
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

        # não alocados ficam -1
        return assign, update_centroids_geomed(pts, assign, k)

    def _local_refine(assign: np.ndarray, centers: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refino local barato (relocate/swap limitados).
        """
        loads1, loads2 = _recompute_loads(assign)
        improved = True
        it = 0
        while improved and it < 3:  # poucas passagens (rápido)
            it += 1
            improved = False

            # relocate aleatório (amostrado)
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

            # swaps aleatórios (poucos)
            idx_assigned = np.where(assign >= 0)[0]
            if idx_assigned.size < 2:
                break
            pairs = rng.choice(idx_assigned, size=(min(len(idx_assigned)//2, 32), 2), replace=False)
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

    # ---------- NOVO: refino guloso de compactação ----------
    def _greedy_compact_refine(assign: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Heurística rápida:
        - Marca 'clientes ruins' (dist > mediana do cluster atual).
        - Para cada ruim, escolhe o cluster viável mais próximo (cap/skills) e realoca de forma gulosa,
          processando em ordem crescente de distância candidata.
        - Atualiza cargas on-the-fly; recalcula centros ao final (mediana geográfica).
        """
        # cargas atuais
        loads1, loads2 = _recompute_loads(assign)

        # distâncias atuais e limiares (mediana) por cluster
        # construímos bad_list: clientes ruins
        bad_clients = []
        for j in range(k):
            idx = np.where(assign == j)[0]
            if idx.size == 0 or np.isnan(centers[j]).any():
                continue
            dists = np.array([haversine(pts[i], centers[j]) for i in idx], dtype=float)
            med = np.median(dists)
            # ruins = dist > mediana
            bad = idx[dists > med + 1e-12]
            if bad.size > 0:
                bad_clients.extend(bad.tolist())

        if not bad_clients:
            return assign, centers

        bad_clients = np.array(bad_clients, dtype=int)

        # Para cada cliente ruim, encontre o cluster viável mais próximo no estado atual
        # (mask de compatibilidade e capacidade)
        candidate_moves = []
        for i in bad_clients:
            j0 = assign[i]
            if j0 < 0 or np.isnan(centers[j0]).any():
                continue
            # mascara de compatibilidade
            compat_i = compat[i, :].copy()
            # capacidade viável
            cap_ok = (loads1 + d1[i] <= c1 + 1e-9) & (loads2 + d2[i] <= c2 + 1e-9)
            # não considerar ficar no mesmo cluster (só realocação)
            mask = compat_i & cap_ok
            mask[j0] = False
            feasible_js = np.where(mask)[0]
            if feasible_js.size == 0:
                continue
            # distâncias para centros viáveis (ignora centros NaN)
            dlist = []
            for l in feasible_js:
                if np.isnan(centers[l]).any():
                    continue
                dlist.append((l, haversine(pts[i], centers[l])))
            if not dlist:
                continue
            # melhor target deste cliente
            l_best, d_best = min(dlist, key=lambda t: t[1])
            candidate_moves.append((i, l_best, d_best))

        if not candidate_moves:
            return assign, centers

        # ordenar globalmente por distância (mais perto primeiro)
        candidate_moves.sort(key=lambda t: t[2])

        moved = np.zeros(n, dtype=bool)
        for (i, l) in [(t[0], t[1]) for t in candidate_moves]:
            if moved[i]:
                continue
            j = assign[i]
            if j == l or j < 0:
                continue
            # checar se ainda cabe em l e se centros ainda válidos
            if np.isnan(centers[l]).any():
                continue
            if not compat[i, l]:
                continue
            if _capacity_ok(loads1[l] + d1[i], loads2[l] + d2[i], c1[l], c2[l]):
                # aplica realocação
                assign[i] = l
                loads1[j] -= d1[i]; loads2[j] -= d2[i]
                loads1[l] += d1[i]; loads2[l] += d2[i]
                moved[i] = True

        # atualiza centros após realocações
        centers = update_centroids_geomed(pts, assign, k, old_centers=centers)
        return assign, centers

    def _sa_single_run(run_seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
        rng = np.random.default_rng(run_seed)

        # 1) Greedy inicial
        assign, centers = _greedy_init(rng)

        # ---- NOVO: compactação imediata da solução inicial
        assign, centers = _greedy_compact_refine(assign, centers)

        best_assign = assign.copy()
        best_centers = centers.copy()
        best_cost = _cost(assign, centers)

        # 2) breve refino local (barato)
        assign, centers = _local_refine(assign, centers, rng)
        cur_cost = _cost(assign, centers)
        if cur_cost < best_cost:
            best_cost, best_assign, best_centers = cur_cost, assign.copy(), centers.copy()

        # 3) SA com vizinhança adaptativa + tabu leve
        T = sa_T0
        total_steps = sa_iters
        base_neighbors = neighbor_base if neighbor_base is not None else max(64, int(np.sqrt(max(1, n))))
        tabu = set()
        tabu_queue: list[tuple] = []

        def _push_tabu(key: tuple):
            tabu.add(key)
            tabu_queue.append(key)
            if len(tabu_queue) > tabu_horizon:
                old = tabu_queue.pop(0)
                if old in tabu:
                    tabu.remove(old)

        # (re)calcular cargas após refinos
        loads1, loads2 = _recompute_loads(assign)
        reheats_left = max(0, int(reheats))

        for step in range(total_steps):
            # progress (0..1)
            prog = step / max(1, total_steps - 1)
            # temperatura exponencial
            T = sa_T0 * (sa_Tf / sa_T0) ** prog
            # vizinhança adaptativa
            neigh_size = max(16, int(base_neighbors * max(0.1, T / sa_T0)))

            accepted = False

            # ======  vizinhança: mistura move + swap  ======
            for _ in range(neigh_size):
                if rng.random() < 0.6:
                    # MOVE: escolhe cliente e tenta outro cluster
                    idx_assigned = np.where(assign >= 0)[0]
                    if idx_assigned.size == 0:
                        continue
                    i = int(rng.choice(idx_assigned))
                    j = int(assign[i])

                    # escolhe cluster alvo
                    l = int(rng.integers(0, k))
                    if l == j or not compat[i, l]:
                        continue
                    key = ("mv", i, j, l)
                    if key in tabu:
                        continue

                    # checar capacidade
                    if not _capacity_ok(loads1[l] + d1[i], loads2[l] + d2[i], c1[l], c2[l]):
                        continue

                    # delta custo (aprox) com centros atuais
                    if np.isnan(centers[j]).any() or np.isnan(centers[l]).any():
                        continue
                    cur = haversine(pts[i], centers[j])
                    new = haversine(pts[i], centers[l])
                    delta = new - cur

                    # aceitação SA
                    if delta < 0 or rng.random() < np.exp(-delta / max(1e-9, T)):
                        # aplica
                        assign[i] = l
                        loads1[j] -= d1[i]; loads2[j] -= d2[i]
                        loads1[l] += d1[i]; loads2[l] += d2[i]
                        _push_tabu(key)
                        accepted = True
                        break
                else:
                    # SWAP: escolhe dois clientes
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

                    # capacidade pós-swap
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
                        # aplica swap
                        assign[i], assign[t] = j2, j1
                        loads1[j1] = loads1[j1] - d1[i] + d1[t]
                        loads2[j1] = loads2[j1] - d2[i] + d2[t]
                        loads1[j2] = loads1[j2] - d1[t] + d1[i]
                        loads2[j2] = loads2[j2] - d2[t] + d2[i]
                        _push_tabu(key)
                        accepted = True
                        break

            # se aceitou algum vizinho → atualiza centros periodicamente
            if accepted and (step % 20 == 0 or step == total_steps - 1):
                centers = update_centroids_geomed(pts, assign, k)

            # melhor global?
            if step % 20 == 0 or step == total_steps - 1:
                cur_cost = _cost(assign, centers)
                if cur_cost < best_cost - 1e-6:
                    best_cost = cur_cost
                    best_assign = assign.copy()
                    best_centers = centers.copy()

            # reheating leve
            if (reheats_left > 0) and (step > 0) and (step % max(1, total_steps // (reheats + 1)) == 0):
                reheats_left -= 1
                T = max(T, sa_T0 * 0.5)  # reaquece parcialmente

        # refino final rápido (local)
        assign, centers = _local_refine(best_assign.copy(), best_centers.copy(), rng)

        # ---- NOVO: compactação final para remover outliers remanescentes
        assign, centers = _greedy_compact_refine(assign, centers)

        final_cost = _cost(assign, centers)
        if final_cost < best_cost:
            best_cost = final_cost
            best_assign = assign
            best_centers = centers

        return best_assign, best_centers, best_cost

    # ---------- multi-start ----------
    seeds = [int(seed + 101 * s) for s in range(n_starts)]
    if _JOBLIB and n_starts > 1:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_sa_single_run)(s) for s in seeds
        )
    else:
        results = [_sa_single_run(s) for s in seeds]

    # escolhe melhor
    best = min(results, key=lambda t: t[2])
    return best

def greedy_teitz_bart_sa_fast_border_polish(
    clients: pd.DataFrame,
    clusters: pd.DataFrame,
    compat: np.ndarray,
    *,
    seed: int = 42,
    # SA
    sa_iters: int = 1000,
    sa_T0: float = 1.0,
    sa_Tf: float = 1e-3,
    reheats: int = 2,
    # Multi-start
    n_starts: int = 8,
    # Vizinhança
    neighbor_base: int | None = None,   # se None => int(sqrt(n))
    # Tabu
    tabu_horizon: int = 100,
    # Paralelo
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Greedy + refino local (relocate/swap) + Simulated Annealing (paralelo e adaptativo)
    + Border Polish (outlier move + swap dirigido) em três níveis:
      - SOFT pós-Greedy (limpeza rápida sem perder diversidade),
      - SOFT periódico durante SA (reparo criterioso),
      - STRONG final (aparar últimas impurezas).
    Respeita compat (skills), 2 demandas e capacidades. Centros por mediana geográfica.
    Retorna: (assign, centers, custo)
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

    # ---------- helpers ----------
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
        """
        Inicialização ágil (variante GreedyFAST).
        - Escolhe 'semente' compatível por cluster.
        - Adiciona clientes mais próximos ao centro corrente respeitando capacidade.
        """
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
                pool = np.where(~used & compat[:, jj])[0]
                if pool.size == 0:
                    break
                sample_sz = min(max(64, int(np.sqrt(pool.size))), pool.size)
                cand = rng.choice(pool, size=sample_sz, replace=False)
                best_i, best_dist = -1, float("inf")
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
        """
        Refino local barato (relocate/swap amostrados).
        """
        loads1, loads2 = _recompute_loads(assign)
        improved = True
        it = 0
        while improved and it < 3:
            it += 1
            improved = False

            # relocate aleatório (amostrado)
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

            # swaps aleatórios (poucos)
            idx_assigned = np.where(assign >= 0)[0]
            if idx_assigned.size < 2:
                break
            pairs = rng.choice(idx_assigned, size=(min(len(idx_assigned)//2, 32), 2), replace=False)
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

    # ---------- Border Polish (outlier move + swap dirigido) ----------
    def _outlier_border_polish(
        assign: np.ndarray,
        centers: np.ndarray,
        *,
        passes: int = 2,
        k_near_centers: int = 5,
        q_quantile: float = 0.85,
        max_swap_candidates: int = 32,
        eps_improve: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Limpeza de impurezas nos clusters:
        - marca outliers por quantil de distância ao centro (q_quantile),
        - tenta mover para k centros viáveis mais próximos,
        - se não couber, tenta swap 1–1 dirigido com membros do cluster-alvo,
        - poucas passadas; centros atualizados ao fim de cada passada.
        """
        loads1, loads2 = _recompute_loads(assign)

        def _cluster_stats(j):
            idx = np.where(assign == j)[0]
            if idx.size == 0 or np.isnan(centers[j]).any():
                return idx, None, None
            dists = np.array([haversine(pts[i], centers[j]) for i in idx], dtype=float)
            thr = np.quantile(dists, q_quantile)
            return idx, dists, thr

        for _ in range(max(1, passes)):
            # 1) lista global de outliers (ordenada por distância decrescente)
            cand = []
            for j in range(k):
                idx, dists, thr = _cluster_stats(j)
                if dists is None:
                    continue
                bad_mask = dists > thr + 1e-12
                if not np.any(bad_mask):
                    continue
                bad_idx = idx[bad_mask]
                for i in bad_idx:
                    cand.append((i, j))
            if not cand:
                break

            def _excesso(pair):
                i, j = pair
                return haversine(pts[i], centers[j])

            cand.sort(key=_excesso, reverse=True)
            changed = False

            for i, j in cand:
                if assign[i] != j or j < 0 or np.isnan(centers[j]).any():
                    continue

                # 2) centros mais próximos do cliente i (compatíveis)
                dist_to_centers = [
                    (l, haversine(pts[i], centers[l]))
                    for l in range(k)
                    if l != j and not np.isnan(centers[l]).any() and compat[i, l]
                ]
                if not dist_to_centers:
                    continue
                dist_to_centers.sort(key=lambda t: t[1])
                neigh = [l for (l, _) in dist_to_centers[:k_near_centers]]

                cur = haversine(pts[i], centers[j])

                # 3) MOVE guloso
                moved = False
                for l in neigh:
                    if _capacity_ok(loads1[l] + d1[i], loads2[l] + d2[i], c1[l], c2[l]):
                        new = haversine(pts[i], centers[l])
                        if new + eps_improve < cur:
                            assign[i] = l
                            loads1[j] -= d1[i]; loads2[j] -= d2[i]
                            loads1[l] += d1[i]; loads2[l] += d2[i]
                            changed = moved = True
                            break
                if moved:
                    continue

                # 4) SWAP dirigido (com cluster alvo mais promissor)
                for l in neigh:
                    idx_l = np.where(assign == l)[0]
                    if idx_l.size == 0:
                        continue
                    d_l = np.array([haversine(pts[t], centers[l]) for t in idx_l], dtype=float)
                    order = np.argsort(d_l)[:max_swap_candidates]
                    best = None
                    for t in idx_l[order]:
                        j1, j2 = j, l
                        if not compat[t, j1]:
                            continue
                        if not _capacity_ok(loads1[j1] - d1[i] + d1[t], loads2[j1] - d2[i] + d2[t], c1[j1], c2[j1]):
                            continue
                        if not _capacity_ok(loads1[j2] - d1[t] + d1[i], loads2[j2] - d2[t] + d2[i], c1[j2], c2[j2]):
                            continue
                        cur_pair = cur + haversine(pts[t], centers[l])
                        new_pair = haversine(pts[i], centers[l]) + haversine(pts[t], centers[j])
                        delta = new_pair - cur_pair
                        if delta < -eps_improve and (best is None or delta < best[0]):
                            best = (delta, t)
                    if best is not None:
                        _, t = best
                        assign[i], assign[t] = l, j
                        loads1[j] = loads1[j] - d1[i] + d1[t]
                        loads2[j] = loads2[j] - d2[i] + d2[t]
                        loads1[l] = loads1[l] - d1[t] + d1[i]
                        loads2[l] = loads2[l] - d2[t] + d2[i]
                        changed = True
                        break  # próximo outlier

            if not changed:
                break
            centers = update_centroids_geomed(pts, assign, k, old_centers=centers)

        return assign, centers

    def _sa_single_run(run_seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
        rng = np.random.default_rng(run_seed)

        # 1) Greedy inicial
        assign, centers = _greedy_init(rng)

        # 1.1) Border polish "SOFT" para limpar raios óbvios sem perder diversidade
        assign, centers = _outlier_border_polish(
            assign, centers,
            passes=1,            # leve
            k_near_centers=4,    # pequeno raio de busca
            q_quantile=0.92,     # conservador (90–95%)
            max_swap_candidates=16
        )

        best_assign = assign.copy()
        best_centers = centers.copy()
        best_cost = _cost(assign, centers)

        # 2) Refino local rápido
        assign, centers = _local_refine(assign, centers, rng)
        cur_cost = _cost(assign, centers)
        if cur_cost < best_cost:
            best_cost, best_assign, best_centers = cur_cost, assign.copy(), centers.copy()

        # 3) SA com vizinhança adaptativa + tabu leve
        T = sa_T0
        total_steps = sa_iters
        base_neighbors = neighbor_base if neighbor_base is not None else max(64, int(np.sqrt(max(1, n))))
        tabu = set()
        tabu_queue: list[tuple] = []

        def _push_tabu(key: tuple):
            tabu.add(key)
            tabu_queue.append(key)
            if len(tabu_queue) > tabu_horizon:
                old = tabu_queue.pop(0)
                if old in tabu:
                    tabu.remove(old)

        loads1, loads2 = _recompute_loads(assign)
        reheats_left = max(0, int(reheats))

        # controle de aceitações recentes para reparo periódico
        accepted_recent = 0
        REPAIR_PERIOD = max(200, total_steps // 4)  # reparo a cada ~1/4 das iterações (mín. 200)

        for step in range(total_steps):
            prog = step / max(1, total_steps - 1)
            T = sa_T0 * (sa_Tf / sa_T0) ** prog
            neigh_size = max(16, int(base_neighbors * max(0.1, T / sa_T0)))

            accepted = False

            for _ in range(neigh_size):
                if rng.random() < 0.6:
                    # MOVE
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
                    # SWAP
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

            # manutenção de centros e melhor global
            if accepted and (step % 20 == 0 or step == total_steps - 1):
                centers = update_centroids_geomed(pts, assign, k)
            if step % 20 == 0 or step == total_steps - 1:
                cur_cost = _cost(assign, centers)
                if cur_cost < best_cost - 1e-6:
                    best_cost = cur_cost
                    best_assign = assign.copy()
                    best_centers = centers.copy()

            # contagem de aceitações para reparo periódico
            accepted_recent = accepted_recent + 1 if accepted else max(0, accepted_recent - 1)

            # ---- Reparo periódico (SOFT) quando o SA esfria e poucas aceitações recentes
            if (step > 0) and (step % REPAIR_PERIOD == 0) and (T < 0.3 * sa_T0) and (accepted_recent < 3):
                assign, centers = _outlier_border_polish(
                    assign, centers,
                    passes=1,
                    k_near_centers=4,
                    q_quantile=0.93,       # ainda mais conservador durante o SA
                    max_swap_candidates=16
                )
                # recalc cargas após reparo
                loads1, loads2 = _recompute_loads(assign)
                # atualiza best se melhorou
                cur_cost = _cost(assign, centers)
                if cur_cost < best_cost - 1e-6:
                    best_cost = cur_cost
                    best_assign = assign.copy()
                    best_centers = centers.copy()

            # reheating leve
            if (reheats_left > 0) and (step > 0) and (step % max(1, total_steps // (reheats + 1)) == 0):
                reheats_left -= 1
                T = max(T, sa_T0 * 0.5)

        # refino local final sobre o melhor encontrado
        assign, centers = _local_refine(best_assign.copy(), best_centers.copy(), rng)

        # 4) Border polish "STRONG" final
        assign, centers = _outlier_border_polish(
            assign, centers,
            passes=2,
            k_near_centers=5,
            q_quantile=0.85,
            max_swap_candidates=24
        )

        final_cost = _cost(assign, centers)
        if final_cost < best_cost:
            best_cost = final_cost
            best_assign = assign
            best_centers = centers

        return best_assign, best_centers, best_cost

    # ---------- multi-start ----------
    seeds = [int(seed + 101 * s) for s in range(n_starts)]
    if _JOBLIB and n_starts > 1:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_sa_single_run)(s) for s in seeds
        )
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
    # Hyperparâmetros de velocidade/qualidade
    relocate_sample_frac: float = 0.35,  # fração dos clientes amostrada por iteração no relocate
    knn_neighbors: int = 100,             # vizinhos por cliente no swap limitado
    mip_topk: int = 3,                   # top-K centros por cliente no MIP
    candidate_topk: int = 5              # top-K centros no relocate (TB)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Refino com clusters fixos (versão rápida):
      1) Teitz & Bart (relocate amostrado + swap por k-NN) com custos pré-computados.
      2) Mini-MIP apenas para não alocados, restrito aos top-K centros viáveis.

    Melhoria de performance:
      - Matriz de distâncias pré-computada cliente x centro.
      - Relocate avalia só top-K centros mais próximos (compatíveis).
      - Swap restrito a pares (i,t) vizinhos (k-NN em coordenadas geográficas).
      - MIP com top-K centros e time limit adaptativo.

    Retorna: (assign_final, centers_fixed, custo_total)
    """
    rng = np.random.default_rng(seed)
    pts = stack_latlon(clients).astype(float)
    n, k = len(clients), len(clusters)

    # --- demandas & capacidades ---
    d1 = clients["demanda1"].to_numpy(float)
    d2 = clients["demanda2"].to_numpy(float)
    c1 = clusters["capacidade1"].to_numpy(float)
    c2 = clusters["capacidade2"].to_numpy(float)

    # ===========================
    # 1) MATRIZ DE DISTÂNCIAS (n x k) – haversine vetorizado
    # ===========================
    def _haversine_matrix(pts_xy: np.ndarray, centers_xy: np.ndarray) -> np.ndarray:
        # pts: (n,2), centers: (k,2) -> dist: (n,k)
        R = 6371.0088
        lat1 = np.radians(pts_xy[:, 0])[:, None]      # (n,1)
        lon1 = np.radians(pts_xy[:, 1])[:, None]
        lat2 = np.radians(centers_xy[:, 0])[None, :]  # (1,k)
        lon2 = np.radians(centers_xy[:, 1])[None, :]

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
        return 2.0 * R * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))

    dist_ck = _haversine_matrix(pts, centers_fixed)  # (n,k)

    # Pré-ordenação: para cada cliente, os índices dos centros do mais próximo ao mais distante
    centers_rank_per_client = np.argsort(dist_ck, axis=1)  # (n,k)

    # ===========================
    # 2) ESTRUTURA KNN ENTRE CLIENTES (para swaps locais)
    # ===========================
    try:
        from scipy.spatial import cKDTree as KDTree
        _KD_OK = True
        tree = KDTree(pts)
    except Exception:
        _KD_OK = False
        tree = None  # fallback sem KDTree

    def _knn_indices(i: int, m: int) -> np.ndarray:
        if _KD_OK:
            d, idx = tree.query(pts[i], k=min(m+1, n))  # inclui o próprio i
            idx = np.atleast_1d(idx)
            # remove o próprio i
            idx = idx[idx != i]
            return idx[:m]
        else:
            # Fallback: distância euclidiana no plano (aprox), pega os m mais próximos
            vec = pts - pts[i]
            sq = np.einsum('ij,ij->i', vec, vec)
            order = np.argsort(sq)
            order = order[order != i]
            return order[:m]

    # ===========================
    # 3) TEITZ & BART RÁPIDO (clusters fixos)
    # ===========================
    assign = assign_init.copy()
    # cargas atuais por cluster
    loads1 = np.zeros(k, dtype=float)
    loads2 = np.zeros(k, dtype=float)
    for j in range(k):
        idx = np.where(assign == j)[0]
        if idx.size:
            loads1[j] = d1[idx].sum()
            loads2[j] = d2[idx].sum()

    def _capacity_ok(l1, l2, C1, C2) -> bool:
        return (l1 <= C1 + 1e-9) and (l2 <= C2 + 1e-9)

    # parâmetros auxiliares
    sample_size = max(1, int(np.ceil(relocate_sample_frac * n)))
    candidate_topk = int(max(1, min(candidate_topk, k)))
    knn_neighbors = int(max(1, min(knn_neighbors, n-1)))

    it = 0
    improved = True
    while improved and it < max_iter_tb:
        it += 1
        improved = False

        # ---------- RELOCATE (amostrado) ----------
        idx_assigned = np.where(assign >= 0)[0]
        if idx_assigned.size > 0:
            sample = rng.choice(idx_assigned, size=min(sample_size, idx_assigned.size), replace=False)
            for i in sample:
                j = assign[i]
                if j < 0:
                    continue
                cur_cost = dist_ck[i, j]

                # candidatos: top-K centros mais próximos e compatíveis
                cand_centers = centers_rank_per_client[i][:candidate_topk]
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
                    # aplica move
                    assign[i] = best_l
                    loads1[j] -= d1[i]; loads2[j] -= d2[i]
                    loads1[best_l] += d1[i]; loads2[best_l] += d2[i]
                    improved = True

        # ---------- SWAP (limitado a k-NN) ----------
        idx_assigned = np.where(assign >= 0)[0]
        if idx_assigned.size >= 2:
            # amostra também para não explodir custo
            sample = rng.choice(idx_assigned, size=min(sample_size, idx_assigned.size), replace=False)
            for i in sample:
                j1 = assign[i]
                if j1 < 0:
                    continue

                # pega vizinhos do i
                neigh = _knn_indices(i, knn_neighbors)
                for t in neigh:
                    if t <= i:
                        continue  # evita pares repetidos
                    j2 = assign[t]
                    if j2 < 0 or j2 == j1:
                        continue

                    # compatibilidade cruzada
                    if (not compat[i, j2]) or (not compat[t, j1]):
                        continue

                    # capacidade pós-swap
                    if not _capacity_ok(loads1[j1] - d1[i] + d1[t], loads2[j1] - d2[i] + d2[t], c1[j1], c2[j1]):
                        continue
                    if not _capacity_ok(loads1[j2] - d1[t] + d1[i], loads2[j2] - d2[t] + d2[i], c1[j2], c2[j2]):
                        continue

                    # custo rápido via matriz pré-computada
                    cur = dist_ck[i, j1] + dist_ck[t, j2]
                    new = dist_ck[i, j2] + dist_ck[t, j1]
                    if new + 1e-9 < cur:
                        # aplica swap
                        assign[i], assign[t] = j2, j1
                        loads1[j1] = loads1[j1] - d1[i] + d1[t]
                        loads2[j1] = loads2[j1] - d2[i] + d2[t]
                        loads1[j2] = loads1[j2] - d1[t] + d1[i]
                        loads2[j2] = loads2[j2] - d2[t] + d2[i]
                        improved = True

    # ===========================
    # 4) MINI-MIP PARA NÃO ALOCADOS (top-K centros & time limit curto)
    # ===========================
    unassigned = np.where(assign < 0)[0]
    if unassigned.size > 0:
        try:
            from ortools.linear_solver import pywraplp
            solver = pywraplp.Solver.CreateSolver("CBC")

            # para cada i, reduzir domínios aos top-K mais próximos compatíveis
            cand_per_i = {}
            for i in unassigned:
                ranked = centers_rank_per_client[i]
                # filtra compatibilidade
                feasible = [j for j in ranked if compat[i, j]][:max(1, mip_topk)]
                # se vazio (sem compat), ainda assim deixa vazio => ficará unassigned via u_i
                cand_per_i[i] = feasible

            # variáveis somente nos candidatos
            x = {}
            u = {}
            for i in unassigned:
                u[i] = solver.IntVar(0, 1, f"u_{i}")
                for j in cand_per_i[i]:
                    x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

            # cada i: soma x + u = 1
            for i in unassigned:
                vars_i = [x[i, j] for j in cand_per_i[i]] if cand_per_i[i] else []
                solver.Add(sum(vars_i) + u[i] == 1)

            # capacidades incrementais
            for j in range(k):
                lhs1 = sum(d1[i] * x[i, j] for i in unassigned if j in cand_per_i[i])
                lhs2 = sum(d2[i] * x[i, j] for i in unassigned if j in cand_per_i[i])
                solver.Add(lhs1 + loads1[j] <= c1[j])
                solver.Add(lhs2 + loads2[j] <= c2[j])

            # custos
            obj_terms = []
            for i in unassigned:
                for j in cand_per_i[i]:
                    obj_terms.append(dist_ck[i, j] * x[i, j])
                obj_terms.append(penalty_unassigned * u[i])

            solver.Minimize(solver.Sum(obj_terms))

            # time limit adaptativo
            tl = int(1000 * min(max(1, mip_time_limit), 5))  # máx 5s
            solver.SetTimeLimit(tl)

            status = solver.Solve()

            # aplica solução (mesmo que não ótima)
            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                for i in unassigned:
                    placed = False
                    for j in cand_per_i[i]:
                        if x[i, j].solution_value() > 0.5:
                            assign[i] = j
                            loads1[j] += d1[i]; loads2[j] += d2[i]
                            placed = True
                            break
                    # se u[i]=1, permanece -1
            else:
                # fallback: greedy
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
            # OR-Tools indisponível -> greedy como fallback
            for i in unassigned:
                ranked = centers_rank_per_client[i]
                # considera apenas compatíveis top-K
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
                # senão fica -1 (penalizado no custo)

    # ===========================
    # 5) CUSTO FINAL
    # ===========================
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
    # --- parâmetros SA ---
    sa_iters: int = 800,
    sa_T0: float = 1.0,
    sa_Tf: float = 1e-3,
    reheats: int = 2,
    n_starts: int = 8,
    neighbor_base: int | None = None,
    tabu_horizon: int = 50,
    n_jobs: int = -1,
    # --- parâmetros MIP rápido ---
    penalty_unassigned: float = 1000.0,
    max_iter_tb: int = 1000,
    mip_time_limit: int = 10,
    relocate_sample_frac: float = 0.35,
    knn_neighbors: int = 20,
    mip_topk: int = 10,
    candidate_topk: int = 10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Pipeline híbrido:
      1) Gera solução inicial via Greedy + Teitz & Bart + SA (com refino apenas inicial).
      2) Alimenta solução no refinador FixedClusters+TB+MIP (rápido).
    Retorna: (assign_final, centers_final, custo_final)
    """
    # -----------------------------
    # ETAPA 1: Greedy+T&B+SA FAST
    # -----------------------------
    assign_gtsa, centers_gtsa, obj_gtsa = greedy_teitz_bart_sa_fast_border_polish(
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

    # -----------------------------
    # ETAPA 2: FixedClusters+TB+MIP FAST
    # -----------------------------
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

def gerar_dados_clientes_clusters(
    num_clientes=1500,
    num_clusters=15,
    demanda1_range=(1, 5),
    demanda2_range=(0, 10),
    cap1_range=(50, 120),
    cap2_range=(300, 600),
    gerar_skills=True,
    max_skills=5,
    pasta_saida="."
):
    """
    Gera automaticamente arquivos CSV de clientes e clusters.

    Parâmetros:
    - num_clientes: quantidade de clientes
    - num_clusters: quantidade de clusters
    - demanda1_range: tupla (min, max) da demanda1
    - demanda2_range: tupla (min, max) da demanda2
    - cap1_range: tupla (min, max) da capacidade1 dos clusters
    - cap2_range: tupla (min, max) da capacidade2 dos clusters
    - gerar_skills: se True, gera skills aleatórias
    - max_skills: número máximo de skills possíveis
    - pasta_saida: diretório onde salvar os arquivos
    """

    # --- Gerar clientes ---
    clientes = []
    for i in range(1, num_clientes + 1):
        lat = random.uniform(-23.3, -25.3)   # Grande SP
        lon = random.uniform(-49.9, -46.3)

        demanda1 = random.randint(*demanda1_range)
        demanda2 = random.randint(*demanda2_range)

        if gerar_skills and max_skills > 0:
            num_skills_cliente = random.randint(0, max_skills)
            skills = ";".join([f"s{random.randint(1, max_skills)}" for _ in range(num_skills_cliente)])
        else:
            skills = ""

        clientes.append([f"c{i}", f"Cliente {i}", lat, lon, demanda1, demanda2, skills])

    clientes_df = pd.DataFrame(clientes, columns=["id", "nome", "lat", "lon", "demanda1", "demanda2", "skills"])

    # --- Gerar clusters ---
    clusters = []
    for i in range(1, num_clusters + 1):
        cap1 = random.randint(*cap1_range)
        cap2 = round(random.uniform(*cap2_range), 1)

        if gerar_skills and max_skills > 0:
            num_skills_cluster = random.randint(0, 1)
            skills_requeridas = ";".join([f"s{random.randint(1, max_skills)}" for _ in range(num_skills_cluster)])
        else:
            skills_requeridas = ""

        clusters.append([f"k{i}", cap1, cap2, skills_requeridas])

    clusters_df = pd.DataFrame(clusters, columns=["id", "capacidade1", "capacidade2", "skills_requeridas"])

    # --- Salvar arquivos ---
    clientes_path = os.path.join(pasta_saida, "clientes_sp.csv")
    clusters_path = os.path.join(pasta_saida, f"clusters_15.csv")

    clientes_df.to_csv(clientes_path, index=False)
    clusters_df.to_csv(clusters_path, index=False)

    print(f"Arquivos gerados:\n- {clientes_path}\n- {clusters_path}")


if __name__=="__main__":
    # ==============================
    # Dados de entrada
    # ==============================
    gerar_dados_clientes_clusters(
        num_clientes=1000,
        num_clusters=50,
        demanda1_range=(1, 1),
        demanda2_range=(0, 0),
        cap1_range=(20, 51),
        cap2_range=(900000, 9999990),
        gerar_skills=True,
        max_skills=5
    )
    clients = parse_clients_csv("clientes_sp.csv")
    clusters = parse_clusters_csv("clusters_15.csv")
    compat = build_compat_matrix(clients["skills"].tolist(), clusters["skills_requeridas"].tolist())

    resultados = {}

    # ==============================
    # Algoritmo 1: Heurístico
    # ==============================
    t0 = time.perf_counter()
    assign_km, centers_km, obj_km = heuristic_kmeans_capacitado(
        clients, clusters, compat, max_iter=200, seed=42, ls_max_iter=20, repair_rounds=3
    )
    t1 = time.perf_counter()
    resultados["Heurístico Kmeans"] = {
        "obj": obj_km,
        "alocados": (assign_km >= 0).sum(),
        "centros": len(np.unique(assign_km[assign_km >= 0])),
        "tempo": t1 - t0
    }
    make_map(clients, centers_km, assign_km, "Heurístico Kmeans", "mapa_heuristico.html")

    t0 = time.perf_counter()
    assign_km, centers_km, obj_km = heuristic_kmeans_capacitado_revivendoclusters(
        clients, clusters, compat, max_iter=200, seed=42, ls_max_iter=20, repair_rounds=3
    )
    t1 = time.perf_counter()
    resultados["Heurístico Kmeans 2"] = {
        "obj": obj_km,
        "alocados": (assign_km >= 0).sum(),
        "centros": len(np.unique(assign_km[assign_km >= 0])),
        "tempo": t1 - t0
    }
    make_map(clients, centers_km, assign_km, "Heurístico Kmeans 2", "mapa_heuristico_reaproveitando cluster.html")

        # Algoritmo 2: Meta-heurística
    t0 = time.perf_counter()
    assign_meta, centers_meta, obj_meta = solve_grasp_vnd_greedy_fixedK(
        clients, clusters, compat,
        alpha=0.3,
        multi_starts=1,
        seed=42,
        repair_rounds=2,
        vnd_max_iter=200
    )
    t1 = time.perf_counter()
    resultados["Meta-heurística"] = {
        "obj": obj_meta,
        "alocados": (assign_meta >= 0).sum(),
        "centros": len(np.unique(assign_meta[assign_meta >= 0])),
        "tempo": t1 - t0
    }
    make_map(clients, centers_meta, assign_meta, "Meta-heurística", "mapa_meta.html")

    t0 = time.perf_counter()
    assign_gtsa, centers_gtsa, obj_gtsa = greedy_teitz_bart_sa_fast(
        clients, clusters, compat,
        seed=42,
        sa_iters=800,
        sa_T0=1.0,
        sa_Tf=1e-3,
        reheats=2,
        n_starts=4,
        n_jobs=-1
    )


    t1 = time.perf_counter()
    resultados["Greedy+T&B+SA (Fast) -"] = {
        "obj": obj_gtsa,
        "alocados": (assign_gtsa >= 0).sum(),
        "centros": len(np.unique(assign_gtsa[assign_gtsa >= 0])),
        "tempo": t1 - t0
    }
    make_map(clients, centers_gtsa, assign_gtsa, "Greedy+T&B+SA (Fast) - ", "mapa_greedy_tb_sa_fast.html")

    t0 = time.perf_counter()
    assign_gtsa, centers_gtsa, obj_gtsa = greedy_teitz_bart_sa_fast_border_polish(
        clients, clusters, compat,
        seed=42,
        sa_iters=800,       # iterações SA
        sa_T0=1.0,          # temperatura inicial
        sa_Tf=1e-3,         # temperatura final
        reheats=2,          # número de reaquecimentos
        n_starts=4,         # multi-start (quantas rodadas paralelas)
        n_jobs=-1           # usa todos os núcleos disponíveis
    )
    t1 = time.perf_counter()
    resultados["Greedy+T&B+SA (Fast)_POLISHHH"] = {
        "obj": obj_gtsa,
        "alocados": (assign_gtsa >= 0).sum(),
        "centros": len(np.unique(assign_gtsa[assign_gtsa >= 0])),
        "tempo": t1 - t0
    }
    make_map(clients, centers_gtsa, assign_gtsa, "Greedy+T&B+SA (Fast)_ POLISH", "mapa_polish.html")


    
    # ==============================
    t0 = time.perf_counter()
    assign_fixed, centers_fixed, obj_fixed = refine_fixed_clusters_tb_mip_fast(
        clients, clusters, compat,
        assign_gtsa, centers_gtsa,  # warm start da solução refinada
        penalty_unassigned=1000.0,
        max_iter_tb=50,
        mip_time_limit=10,
        seed=42
    )
    t1 = time.perf_counter()
    resultados["FixedClusters+TB+MIP (OTIMIZADO)"] = {
        "obj": obj_fixed,
        "alocados": (assign_fixed >= 0).sum(),
        "centros": len(np.unique(assign_fixed[assign_fixed >= 0])),
        "tempo": t1 - t0
    }
    make_map(clients, centers_fixed, assign_fixed, "FixedClusters+TB+MIP (OTIMIZADO)", "mapa_fixed_tb_mip_base.html")

    t0 = time.perf_counter()
    assign_hybrid, centers_hybrid, obj_hybrid = hybrid_greedy_sa_tb_mip_fast(
        clients, clusters, compat,
        penalty_unassigned=1000.0,
        max_iter_tb=50,
        mip_time_limit=10,
        seed=42
    )
    t1 = time.perf_counter()
    


    resultados["Hybrid Greedy+SA+FixedTB+MIP (OTIMIZADO)"] = {
        "obj": obj_hybrid,
        "alocados": (assign_hybrid >= 0).sum(),
        "centros": len(np.unique(assign_hybrid[assign_hybrid >= 0])),
        "tempo": t1 - t0
    }

    make_map(
        clients, centers_hybrid, assign_hybrid,
        "Hybrid Greedy+SA+FixedTB+MIP (OTIMIZADO)",
        "mapa_hybrid_greedy_tb_sa_mip_fast.html"
    )


    # ==============================
    # Comparativo Final
    # ==============================
    print("\n=== COMPARATIVO FINAL ===")
    print(f"{'Algoritmo':30} | {'Obj':>10} | {'Alocados':>10} | {'Centros':>8} | {'Tempo (s)':>10}")
    print("-"*85)
    for nome, res in resultados.items():
        print(f"{nome:30} | {res['obj']:10.2f} | {res['alocados']:10d} | {res['centros']:8d} | {res['tempo']:10.3f}")