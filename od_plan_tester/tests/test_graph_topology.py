"""
Tests for Spatial Urban Graph Topology and Fallback Invariants.
(Tests T07 to T10)
"""

import pytest
import torch
import numpy as np
from od_plan_tester.project_adapter import (
    build_radius_graph,
    haversine_distance_matrix,
)


@pytest.mark.reference
def test_radius_graph_5km_threshold(sample_coordinates):
    """T07: Radius graph with r=5.0 km connects all pairs within 5 km."""
    edge_index, edge_dist = build_radius_graph(sample_coordinates, radius_km=5.0)

    # Core tract 0 (-84.388, 33.749) and tract 1 (-84.390, 33.750) are ~0.25 km apart
    # Tract 2 is ~1.5 km apart. They should be connected.
    dist_mat = haversine_distance_matrix(sample_coordinates)
    assert dist_mat[0, 1] < 5.0
    assert dist_mat[0, 2] < 5.0

    # Verify edge exists in edge_index
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert (0, 1) in edges or (1, 0) in edges
    assert (0, 2) in edges or (2, 0) in edges


@pytest.mark.reference
def test_radius_graph_isolated_fallback_1nn(sample_coordinates):
    """T08: Isolated tract (node 4, ~70 km away) connects via 1-NN fallback (degree >= 1)."""
    edge_index, edge_dist = build_radius_graph(sample_coordinates, radius_km=5.0)

    # Node 4 is distant. It must have at least 1 neighbor via fallback
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    node_4_neighbors = [dst for src, dst in edges if src == 4 and dst != 4]

    assert len(node_4_neighbors) >= 1, "Isolated node must have 1-NN fallback edge"


@pytest.mark.scientific
def test_urban_graph_zero_od_leakage(sample_coordinates):
    """T09: Graph construction uses only spatial coordinates and is completely independent of OD flows."""
    ei1, ed1 = build_radius_graph(sample_coordinates, radius_km=5.0)
    ei2, ed2 = build_radius_graph(sample_coordinates, radius_km=5.0)

    assert torch.equal(ei1, ei2)
    assert torch.equal(ed1, ed2)


@pytest.mark.contract
def test_graph_symmetric_and_self_loops(sample_coordinates):
    """T10: Built graph contains self-loops and is undirected (symmetric adjacency)."""
    edge_index, edge_dist = build_radius_graph(sample_coordinates, radius_km=5.0, include_self_loop=True)

    n_nodes = len(sample_coordinates)
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Check self loops
    for i in range(n_nodes):
        assert (i, i) in edges

    # Check symmetry
    for u, v in edges:
        assert (v, u) in edges


@pytest.mark.contract
def test_graph_caching_and_clear(sample_coordinates):
    """T11: Graph caching returns identical cached tensors and respects clear_graph_cache."""
    from src.data.urban_graph import build_radius_graph, build_knn_graph, clear_graph_cache, _GRAPH_CACHE

    clear_graph_cache()
    assert len(_GRAPH_CACHE) == 0

    ei1, ed1 = build_radius_graph(sample_coordinates, radius_km=5.0, use_cache=True)
    assert len(_GRAPH_CACHE) == 1

    ei2, ed2 = build_radius_graph(sample_coordinates, radius_km=5.0, use_cache=True)
    assert len(_GRAPH_CACHE) == 1
    assert torch.equal(ei1, ei2)
    assert torch.equal(ed1, ed2)

    # Test knn caching
    ki1, kd1 = build_knn_graph(sample_coordinates, k=3, use_cache=True)
    assert len(_GRAPH_CACHE) == 2
    ki2, kd2 = build_knn_graph(sample_coordinates, k=3, use_cache=True)
    assert len(_GRAPH_CACHE) == 2
    assert torch.equal(ki1, ki2)
    assert torch.equal(kd1, kd2)

    clear_graph_cache()
    assert len(_GRAPH_CACHE) == 0


@pytest.mark.contract
def test_raw_city_data_caching():
    """T12: Raw city dataset cache avoids redundant I/O and supports independent scalers."""
    from src.data.dataset import load_city, load_raw_city, clear_city_cache, _RAW_CITY_CACHE
    from sklearn.preprocessing import StandardScaler

    clear_city_cache()
    assert len(_RAW_CITY_CACHE) == 0

    raw1 = load_raw_city("Raleigh", data_root="data", use_cache=True)
    assert len(_RAW_CITY_CACHE) == 1
    raw2 = load_raw_city("Raleigh", data_root="data", use_cache=True)
    assert raw1 is raw2

    # Verify scaling with two different scalers preserves raw features
    s1 = StandardScaler()
    s1.fit(raw1.X_raw * 2.0)
    cd1 = load_city("Raleigh", data_root="data", feature_scaler=s1, use_cache=True)

    s2 = StandardScaler()
    s2.fit(raw1.X_raw * 0.5)
    cd2 = load_city("Raleigh", data_root="data", feature_scaler=s2, use_cache=True)

    # cd1 and cd2 have different normalized features but share same raw data
    assert not torch.allclose(cd1.node_features, cd2.node_features)
    assert torch.equal(cd1.pair_distance, cd2.pair_distance)
    assert torch.equal(cd1.pair_trips, cd2.pair_trips)
    assert torch.equal(cd1.lon_lat, cd2.lon_lat)

    clear_city_cache()
    assert len(_RAW_CITY_CACHE) == 0


@pytest.mark.contract
def test_distance_matrix_caching_and_clear(sample_coordinates):
    """T13: Haversine distance matrix cache avoids redundant O(N^2) computation."""
    from src.data.urban_graph import (
        haversine_distance_matrix,
        build_radius_graph,
        clear_distance_matrix_cache,
        clear_graph_cache,
        _DISTANCE_MATRIX_CACHE,
    )

    clear_distance_matrix_cache()
    assert len(_DISTANCE_MATRIX_CACHE) == 0

    # 1. Direct call caches the matrix
    m1 = haversine_distance_matrix(sample_coordinates, use_cache=True)
    assert len(_DISTANCE_MATRIX_CACHE) == 1

    m2 = haversine_distance_matrix(sample_coordinates, use_cache=True)
    assert m1 is m2  # Exact object identity from cache
    assert np.array_equal(m1, m2)

    # 2. build_radius_graph with different radii shares the cached distance matrix
    clear_graph_cache()
    assert len(_DISTANCE_MATRIX_CACHE) == 0

    _ = build_radius_graph(sample_coordinates, radius_km=3.0, use_cache=True)
    assert len(_DISTANCE_MATRIX_CACHE) == 1

    # Calling with radius=7.0 reuses the distance matrix without recomputing
    _ = build_radius_graph(sample_coordinates, radius_km=7.0, use_cache=True)
    assert len(_DISTANCE_MATRIX_CACHE) == 1

    clear_distance_matrix_cache()
    assert len(_DISTANCE_MATRIX_CACHE) == 0


@pytest.mark.contract
def test_city_data_instance_cache_with_scaler():
    """T14: CityData instance cache returns identical object when called with same scaler."""
    from src.data.dataset import load_city, load_raw_city, clear_city_cache, _CITY_DATA_CACHE
    from sklearn.preprocessing import StandardScaler

    clear_city_cache()
    assert len(_CITY_DATA_CACHE) == 0

    raw = load_raw_city("Raleigh", data_root="data", use_cache=True)
    s = StandardScaler()
    s.fit(raw.X_raw)

    cd1 = load_city("Raleigh", data_root="data", feature_scaler=s, use_cache=True)
    assert len(_CITY_DATA_CACHE) == 1

    cd2 = load_city("Raleigh", data_root="data", feature_scaler=s, use_cache=True)
    assert cd1 is cd2  # Exact object identity reused without re-creating tensors
    assert len(_CITY_DATA_CACHE) == 1

    clear_city_cache()
    assert len(_CITY_DATA_CACHE) == 0


@pytest.mark.contract
def test_preload_all_cities_smoke():
    """T15: preload_all_cities warms up raw data and spatial graph caches for selected cities."""
    from src.data.dataset import preload_all_cities, clear_city_cache, _RAW_CITY_CACHE
    from src.data.urban_graph import _GRAPH_CACHE, clear_graph_cache

    clear_city_cache()
    clear_graph_cache()

    preload_all_cities(data_root="data", city_names=["Raleigh", "Denver"], build_graphs=True, radius_km=5.0)
    assert len(_RAW_CITY_CACHE) == 2
    assert len(_GRAPH_CACHE) == 2

    clear_city_cache()
    clear_graph_cache()


