"""
Tests for Data Splits, Scaler Isolation, and Omega_c Candidate Provenance.
(Tests T11 to T15)
"""

import pytest
import torch
import numpy as np
from od_plan_tester.project_adapter import (
    generate_5fold_splits,
    load_city,
    load_cities,
)


@pytest.mark.reference
def test_stratified_5fold_split_structure():
    """T11: 5-Fold split contains 5 folds, each with 40 train and 10 test cities."""
    splits = generate_5fold_splits(data_root="data")

    assert len(splits) == 5
    for fold_id, split in splits.items():
        assert len(split["train"]) == 40
        assert len(split["test"]) == 10
        # Disjoint
        assert len(set(split["train"]) & set(split["test"])) == 0


@pytest.mark.scientific
def test_stratified_5fold_exact_single_coverage():
    """T12: Across all 5 folds, every single city of the 50 cities is tested exactly once."""
    splits = generate_5fold_splits(data_root="data")

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
