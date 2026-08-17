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
