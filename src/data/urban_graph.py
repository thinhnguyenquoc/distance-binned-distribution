"""
Spatial Urban Graph Construction (G^urban).

Constructs the urban spatial graph from tract centroid coordinates (lon, lat).
Crucial requirement: G^urban uses ONLY observable spatial geography, NEVER OD flows.

Supports:
1. k-NN graph: connects each node to its k geographically nearest neighbors.
2. Radius graph: connects nodes within a geographic distance threshold d_max (km).
3. Adaptive Radius graph: radius normalized to the city's empirical spatial diameter / extent.
"""

import math
import numpy as np
import torch


def haversine_distance_matrix(lon_lat: np.ndarray) -> np.ndarray:
    """Computes pairwise Haversine distances in kilometers."""
    R = 6371.0
    lons = np.radians(lon_lat[:, 0])
    lats = np.radians(lon_lat[:, 1])

    dlon = lons[:, None] - lons[None, :]
    dlat = lats[:, None] - lats[None, :]

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return R * c


def build_knn_graph(lon_lat: np.ndarray, k: int = 10, include_self_loop: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs a k-nearest neighbor spatial graph."""
    N = len(lon_lat)
    k = min(k, N - 1)
    dist_mat = haversine_distance_matrix(lon_lat)

    rows, cols, dists = [], [], []
    for i in range(N):
        indices = np.argsort(dist_mat[i])
        neighbors = indices[1 : k + 1]

        if include_self_loop:
            rows.append(i)
            cols.append(i)
            dists.append(0.0)

        for nbr in neighbors:
            rows.append(i)
            cols.append(nbr)
            dists.append(dist_mat[i, nbr])

    edge_dict = {}
    for r, c, d in zip(rows, cols, dists):
        edge_dict[(r, c)] = d
        edge_dict[(c, r)] = d

    e_rows = [k[0] for k in edge_dict.keys()]
    e_cols = [k[1] for k in edge_dict.keys()]
    e_dists = list(edge_dict.values())

    edge_index = torch.tensor([e_rows, e_cols], dtype=torch.long)
    edge_dist = torch.tensor(e_dists, dtype=torch.float32)
    return edge_index, edge_dist


def build_radius_graph(
    lon_lat: np.ndarray,
    radius_km: float = 5.0,
    include_self_loop: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs a radius-based spatial graph connecting nodes within radius_km."""
    N = len(lon_lat)
    dist_mat = haversine_distance_matrix(lon_lat)

    rows, cols, dists = [], [], []
    for i in range(N):
        if include_self_loop:
            rows.append(i)
            cols.append(i)
            dists.append(0.0)

        within_radius = np.where((dist_mat[i] <= radius_km) & (dist_mat[i] > 0))[0]
        if len(within_radius) == 0:
            closest = np.argsort(dist_mat[i])[1]
            within_radius = [closest]

        for nbr in within_radius:
            rows.append(i)
            cols.append(nbr)
            dists.append(dist_mat[i, nbr])

    edge_dict = {}
    for r, c, d in zip(rows, cols, dists):
        edge_dict[(r, c)] = d
        edge_dict[(c, r)] = d

    e_rows = [k[0] for k in edge_dict.keys()]
    e_cols = [k[1] for k in edge_dict.keys()]
    e_dists = list(edge_dict.values())

    edge_index = torch.tensor([e_rows, e_cols], dtype=torch.long)
    edge_dist = torch.tensor(e_dists, dtype=torch.float32)
    return edge_index, edge_dist


def build_adaptive_radius_graph(
    lon_lat: np.ndarray,
    scale_fraction: float = 0.15,
    min_radius_km: float = 2.0,
    include_self_loop: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Constructs a spatial radius graph where radius_km is normalized to the city's
    empirical spatial diameter (max distance * scale_fraction).
    """
    dist_mat = haversine_distance_matrix(lon_lat)
    diameter = float(np.max(dist_mat))
    adaptive_radius = max(min_radius_km, diameter * scale_fraction)
    ei, ed = build_radius_graph(lon_lat, radius_km=adaptive_radius, include_self_loop=include_self_loop)
    return ei, ed, adaptive_radius


if __name__ == "__main__":
    coords = np.array([
        [-84.3880, 33.7490],
        [-84.3900, 33.7500],
        [-84.4000, 33.7600],
        [-84.5000, 33.8000],
    ])
    ei, ed, r = build_adaptive_radius_graph(coords, scale_fraction=0.2)
    print(f"Adaptive radius: {r:.2f} km | Edges: {ei.shape[1]}")
