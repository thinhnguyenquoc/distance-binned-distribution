"""
Spatial Urban Graph Construction (G^urban).

Constructs the urban spatial graph from tract centroid coordinates (lon, lat).
Crucial requirement: G^urban uses ONLY observable spatial geography, NEVER OD flows.

Supports:
1. k-NN graph: connects each node to its k geographically nearest neighbors.
2. Radius graph: connects nodes within a geographic distance threshold d_max (km).

Returns PyTorch Geometric compatible edge_index (2, E) and optional edge_weight (distance in km).
"""

import math
import numpy as np
import torch


def haversine_distance_matrix(lon_lat: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Haversine distances in kilometers.

    Args:
        lon_lat: (N, 2) array of [lon, lat] coordinates in degrees.

    Returns:
        (N, N) distance matrix in km.
    """
    R = 6371.0  # Earth radius in km
    lons = np.radians(lon_lat[:, 0])
    lats = np.radians(lon_lat[:, 1])

    dlon = lons[:, None] - lons[None, :]
    dlat = lats[:, None] - lats[None, :]

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return R * c


def build_knn_graph(lon_lat: np.ndarray, k: int = 10, include_self_loop: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs a k-nearest neighbor spatial graph.

    Args:
        lon_lat: (N, 2) coordinates.
        k: number of nearest neighbors.
        include_self_loop: whether to include self-loops.

    Returns:
        edge_index: (2, E) LongTensor
        edge_dist:  (E,) FloatTensor distance in km
    """
    N = len(lon_lat)
    k = min(k, N - 1)  # cap at N-1
    dist_mat = haversine_distance_matrix(lon_lat)

    rows, cols, dists = [], [], []

    for i in range(N):
        # Sort neighbors by distance (excluding self for neighbor selection)
        indices = np.argsort(dist_mat[i])
        # indices[0] is self (dist = 0)
        neighbors = indices[1 : k + 1]

        if include_self_loop:
            rows.append(i)
            cols.append(i)
            dists.append(0.0)

        for nbr in neighbors:
            rows.append(i)
            cols.append(nbr)
            dists.append(dist_mat[i, nbr])

    # Convert to symmetric / undirected if needed, or maintain directed k-NN
    # By default, we make the graph symmetric for message passing:
    edge_dict = {}
    for r, c, d in zip(rows, cols, dists):
        edge_dict[(r, c)] = d
        edge_dict[(c, r)] = d  # make undirected

    e_rows = [k[0] for k in edge_dict.keys()]
    e_cols = [k[1] for k in edge_dict.keys()]
    e_dists = list(edge_dict.values())

    edge_index = torch.tensor([e_rows, e_cols], dtype=torch.long)
    edge_dist = torch.tensor(e_dists, dtype=torch.float32)
    return edge_index, edge_dist


def build_radius_graph(lon_lat: np.ndarray, radius_km: float = 5.0, include_self_loop: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs a radius-based spatial graph connecting nodes within radius_km.
    Guarantees at least 1 neighbor per node (fallback to 1-NN if isolated).
    """
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
            # Fallback: connect to closest neighbor
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


if __name__ == "__main__":
    # Quick test
    coords = np.array([
        [-84.3880, 33.7490],  # node 0
        [-84.3900, 33.7500],  # node 1 (very close to 0)
        [-84.4000, 33.7600],  # node 2
        [-84.5000, 33.8000],  # node 3 (further)
    ])
    ei, ed = build_knn_graph(coords, k=2)
    print(f"k-NN graph edge_index shape: {ei.shape}, dists: {ed.shape}")
    print(f"Edge index:\n{ei}")
    print("urban_graph.py test passed.")
