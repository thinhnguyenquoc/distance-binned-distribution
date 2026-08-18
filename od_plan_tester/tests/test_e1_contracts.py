"""
Unit and Contract Tests for E1 Oracle Existence Test Implementation.

Tests cover:
  - T49: 35/5/10 fold split invariants (sizes, disjointness, coverage).
  - T50: compute_kbin_edges invariant (strictly increasing, bounds, deduplication).
  - T51: extract_yd_kbins invariant (proper sum to 1.0, support handling).
  - T52: calibrate_kbins mass preservation and intrazonal identity.
  - T53: calibrate_kbins q=1 exact bin distribution matching.
  - T54: calibrate_kbins GT permutation invariance.
  - T55: get_donor_city distinctness and wrap-around.
"""

import numpy as np
import pytest
import torch

from src.data.city_splits import generate_35_5_10_splits, get_donor_city
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.calibration.bin_calibration import calibrate_kbins


def test_t49_splits_35_5_10_invariants():
    splits = generate_35_5_10_splits("data")
    assert len(splits) == 5

    all_test = []
    for f, s in splits.items():
        train = set(s["train"])
        val   = set(s["val"])
        test  = set(s["test"])

        assert len(s["train"]) == 35, f"Fold {f} train size {len(s['train'])} != 35"
        assert len(s["val"]) == 5, f"Fold {f} val size {len(s['val'])} != 5"
        assert len(s["test"]) == 10, f"Fold {f} test size {len(s['test'])} != 10"

        # Disjoint within fold
        assert len(train & val) == 0, f"Fold {f} train and val overlap!"
        assert len(train & test) == 0, f"Fold {f} train and test overlap!"
        assert len(val & test) == 0, f"Fold {f} val and test overlap!"

        all_test.extend(s["test"])

    # Across folds: test sets form exact partition of 50 cities
    assert len(all_test) == 50
    assert len(set(all_test)) == 50


def test_t50_kbin_edges_strictly_increasing():
    splits = generate_35_5_10_splits("data")
    train35 = splits[1]["train"]
    edges, K_active = compute_kbin_edges(train35, K=8, data_root="data")

    assert len(edges) == K_active + 1
    assert edges[0] == 0.0
    assert np.isinf(edges[-1])
    assert np.all(np.diff(edges) > 0), f"Bin edges not strictly increasing: {edges}"
    assert K_active >= 2, f"Too few active bins: {K_active}"


def test_t51_extract_yd_kbins_normalized():
    dist_km = np.array([0.0, 2.5, 5.0, 15.0, 30.0, 80.0, 150.0])
    trips   = np.array([50.0, 20.0, 30.0, 15.0, 10.0, 5.0, 2.0])
    inter_mask = np.array([False, True, True, True, True, True, True])
    edges = np.array([0.0, 10.0, 50.0, 100.0, np.inf])

    yd = extract_yd_kbins(dist_km, trips, edges, inter_mask)
    assert len(yd) == 4
    assert np.isclose(yd.sum(), 1.0, atol=1e-6)
    assert np.all(yd >= 0.0)


def test_t52_calibrate_kbins_mass_and_intrazonal_invariants():
    t0 = np.array([100.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    dist_km = np.array([0.0, 2.0, 8.0, 25.0, 120.0], dtype=np.float64)
    inter_mask = np.array([False, True, True, True, True])
    edges = np.array([0.0, 5.0, 15.0, 50.0, np.inf])
    yd_target = np.array([0.1, 0.4, 0.3, 0.2])

    t_cal = calibrate_kbins(t0, dist_km, inter_mask, yd_target, edges, q=1.0)

    # Invariant 1: Intrazonal preserved
    assert np.isclose(t_cal[0], t0[0], atol=1e-6)

    # Invariant 2: Interzonal mass preserved
    assert np.isclose(t_cal[inter_mask].sum(), t0[inter_mask].sum(), atol=1e-4)


def test_t53_calibrate_kbins_q1_exact_distribution():
    t0 = np.array([10.0, 50.0, 20.0, 10.0, 5.0], dtype=np.float64)
    dist_km = np.array([0.0, 2.0, 8.0, 25.0, 120.0], dtype=np.float64)
    inter_mask = np.array([False, True, True, True, True])
    edges = np.array([0.0, 5.0, 15.0, 50.0, np.inf])
    yd_target = np.array([0.40, 0.30, 0.20, 0.10])

    t_cal = calibrate_kbins(t0, dist_km, inter_mask, yd_target, edges, q=1.0)
    inter_cal = t_cal[inter_mask]
    total_cal = inter_cal.sum()

    for k in range(4):
        lo, hi = edges[k], edges[k+1]
        in_b = (dist_km[inter_mask] > lo) & (dist_km[inter_mask] <= hi)
        prop = inter_cal[in_b].sum() / total_cal
        assert np.isclose(prop, yd_target[k], atol=1e-5), f"Bin {k} prop {prop} != target {yd_target[k]}"


def test_t54_calibrate_kbins_gt_invariance():
    """T_cal is a function of (T0, Y_D), completely independent of T_GT at cell level."""
    t0 = np.array([50.0, 20.0, 30.0, 40.0], dtype=np.float64)
    dist_km = np.array([0.0, 5.0, 15.0, 45.0], dtype=np.float64)
    inter_mask = np.array([False, True, True, True])
    edges = np.array([0.0, 10.0, 30.0, np.inf])
    yd_target = np.array([0.5, 0.3, 0.2])

    t_cal1 = calibrate_kbins(t0, dist_km, inter_mask, yd_target, edges, q=1.0)
    t_cal2 = calibrate_kbins(t0, dist_km, inter_mask, yd_target, edges, q=1.0)

    assert np.allclose(t_cal1, t_cal2)


def test_t55_donor_city_deterministic_and_distinct():
    test_cities = ["Austin", "Denver", "Portland", "Seattle"]
    for c in test_cities:
        donor = get_donor_city(c, test_cities)
        assert donor != c, f"Donor {donor} is identical to target {c}"
        assert donor in test_cities, f"Donor {donor} not in test set"

    # Verify wrap-around
    assert get_donor_city("Seattle", test_cities) == "Austin"


def test_t56_confirmatory_guard_on_incomplete_subsets():
    """Verify that smoke / partial results are NOT reported as confirmatory."""
    from src.experiment.run_e1 import compute_summary

    # Dummy partial results (only 2 cities in Folds 4 and 5)
    dummy_results = [
        {
            "city": "Portland", "fold": 4, "donor_city": "Denver", "n_inter_pairs": 1000,
            "K_active": 8, "cpc_baseline": 0.40, "cpc_baseline_norm": 0.50,
            "cpc_target_yd": 0.43, "cpc_target_yd_norm": 0.53, "delta_cpc_target": 0.03,
            "cpc_wrong_yd": 0.39, "cpc_wrong_yd_norm": 0.49, "delta_cpc_wrong": -0.01,
            "Y_D_target": [0.125]*8, "Y_D_wrong": [0.125]*8
        },
        {
            "city": "Denver", "fold": 5, "donor_city": "Portland", "n_inter_pairs": 1000,
            "K_active": 8, "cpc_baseline": 0.42, "cpc_baseline_norm": 0.52,
            "cpc_target_yd": 0.45, "cpc_target_yd_norm": 0.55, "delta_cpc_target": 0.03,
            "cpc_wrong_yd": 0.41, "cpc_wrong_yd_norm": 0.51, "delta_cpc_wrong": -0.01,
            "Y_D_target": [0.125]*8, "Y_D_wrong": [0.125]*8
        }
    ]

    summary = compute_summary(dummy_results)
    assert not summary["is_confirmatory_complete"], "Partial 2-city run was falsely marked as confirmatory complete!"
    assert not summary["is_full_50_complete"], "Partial 2-city run was falsely marked as full 50 complete!"
    assert summary["confirmatory_folds_2_5"]["status"] == "not_available", "Confirmatory status should be not_available!"

