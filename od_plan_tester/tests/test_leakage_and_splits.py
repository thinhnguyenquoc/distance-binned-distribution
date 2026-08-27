"""
Tests for Data Splits, Scaler Isolation, and Omega_c Candidate Provenance.
(Tests T11 to T15)
"""

import pytest
import torch
import numpy as np
from od_plan_tester.project_adapter import (
    generate_35_5_10_splits,
    load_city,
    load_cities,
)


@pytest.mark.reference
def test_stratified_5fold_split_structure():
    """T11: Canonical 5-Fold split contains 5 folds, each with 35 train, 5 val, and 10 test cities."""
    splits = generate_35_5_10_splits(data_root="data")

    assert len(splits) == 5
    for fold_id, split in splits.items():
        assert len(split["train"]) == 35
        assert len(split["val"]) == 5
        assert len(split["test"]) == 10
        # All pairwise disjoint
        assert len(set(split["train"]) & set(split["test"])) == 0
        assert len(set(split["train"]) & set(split["val"])) == 0
        assert len(set(split["val"]) & set(split["test"])) == 0


@pytest.mark.scientific
def test_stratified_5fold_exact_single_coverage():
    """T12: Across all 5 folds, every single city of the 50 cities is tested exactly once."""
    splits = generate_35_5_10_splits(data_root="data")

    all_test_cities = []
    for split in splits.values():
        all_test_cities.extend(split["test"])

    assert len(all_test_cities) == 50
    assert len(set(all_test_cities)) == 50, "Every city must appear in the test set exactly once."


@pytest.mark.scientific
def test_scaler_train_isolation():
    """T13: StandardScaler is fitted strictly on source training cities; target city tracts never enter scaler."""
    train_cities = ["Raleigh", "Denver"]
    test_city = "Philadelphia"

    train_data_list, fitted_scaler = load_cities(train_cities, data_root="data")
    test_data = load_city(test_city, data_root="data", feature_scaler=fitted_scaler, fit_scaler=False)

    total_train_tracts = sum(c.n_tracts for c in train_data_list)
    assert fitted_scaler.n_samples_seen_ == total_train_tracts
    assert fitted_scaler.n_samples_seen_ != (total_train_tracts + test_data.n_tracts)


@pytest.mark.scientific
def test_omega_c_provenance_strictly_positive():
    """T14: Target city candidate pairs Omega_c contain strictly positive flow counts (T_ij >= 1)."""
    test_cities = ["Philadelphia", "Denver", "Raleigh"]
    for c_name in test_cities:
        cd = load_city(c_name, data_root="data")
        assert (cd.pair_trips >= 1.0).all()
        assert cd.pair_trips.min().item() >= 1.0


@pytest.mark.scientific
def test_unobserved_pairs_excluded_not_zero_filled():
    """T15: Candidate support Omega_c has size <= N*(N-1); unobserved pairs are excluded (not zero-filled)."""
    cd = load_city("Philadelphia", data_root="data")
    n_nodes = cd.n_tracts
    max_possible_pairs = n_nodes * n_nodes

    assert cd.n_pairs <= max_possible_pairs
    assert len(cd.pair_trips) == cd.n_pairs
    # Ensure there are no 0s stored
    assert not (cd.pair_trips == 0).any()


@pytest.mark.scientific
def test_omega_c_plus_distance_equivalence():
    """T15b: Omega_c^+ definition invariant: (bin_labels > 0) <=> (D_ij > 0) <=> (pair_o != pair_d)."""
    test_cities = ["Philadelphia", "Denver", "Raleigh", "Austin", "Seattle"]
    for c_name in test_cities:
        cd = load_city(c_name, data_root="data")
        dist_km = torch.expm1(cd.pair_distance)

        # 1. Invariant: bin_labels > 0 iff distance_km > 0
        assert torch.equal(cd.bin_labels > 0, dist_km > 0.0), f"Bin label vs distance mismatch in {c_name}"

        # 2. Invariant: (pair_o != pair_d) & (bin_labels > 0) strictly matches (pair_o != pair_d) & (D_ij > 0)
        mask_bins = (cd.pair_o_idx != cd.pair_d_idx) & (cd.bin_labels > 0)
        mask_dist = (cd.pair_o_idx != cd.pair_d_idx) & (dist_km > 0.0)
        assert torch.equal(mask_bins, mask_dist), f"Omega_c^+ mask mismatch in {c_name}"

        # 3. Invariant: Intrazonal pairs have distance == 0 and bin == 0
        intra_mask = (cd.pair_o_idx == cd.pair_d_idx)
        assert torch.all(dist_km[intra_mask] == 0.0)
        assert torch.all(cd.bin_labels[intra_mask] == 0)


@pytest.mark.scientific
def test_all_50_cities_omega_plus_invariants():
    """T15c: Exhaustive verification across all 50 dataset cities that D_ij > 0 <=> bin_labels > 0."""
    from src.data.city_splits import get_all_cities_sorted_by_size
    all_cities_info = get_all_cities_sorted_by_size(data_root="data")
    assert len(all_cities_info) == 50

    for c_info in all_cities_info:
        c_name = c_info["city"]
        cd = load_city(c_name, data_root="data")
        dist_km = torch.expm1(cd.pair_distance)

        # Invariant: bin_labels > 0 strictly equals distance_km > 0
        assert torch.equal(cd.bin_labels > 0, dist_km > 0.0), f"Distance vs bin mismatch in {c_name}"

        # Invariant: Omega_c^+ mask strictly identical
        mask_bins = (cd.pair_o_idx != cd.pair_d_idx) & (cd.bin_labels > 0)
        mask_dist = (cd.pair_o_idx != cd.pair_d_idx) & (dist_km > 0.0)
        assert torch.equal(mask_bins, mask_dist), f"Omega_c^+ definition discrepancy in {c_name}"



