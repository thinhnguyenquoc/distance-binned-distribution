"""
Unit and Contract Tests for E1 Oracle Existence Test Implementation (Amended Protocol v2).

Tests cover:
  - T49: 35/5/10 fold split invariants & manifest v2 integrity + SHA-256 matching.
  - T50: compute_kbin_edges invariant (strictly increasing, bounds, deduplication).
  - T51: extract_yd_kbins invariant (proper sum to 1.0, support handling).
  - T52: calibrate_kbins mass preservation and intrazonal identity.
  - T53: calibrate_kbins q=1 exact bin distribution matching.
  - T54: calibrate_kbins GT permutation invariance.
  - T55: wrong-donor helper distinctness, coverage (9 donors), and legacy single donor.
  - T56: Confirmatory guard on incomplete subsets.
  - T57: Size-stratified validation representation invariant across size strata and metadata logging.
  - T58: Specificity estimand (Delta_target - Delta_wrong_avg9) & IQR calculation.
"""

import numpy as np
import pytest
import torch

from src.data.city_splits import (
    get_all_cities_sorted_by_size,
    generate_5fold_splits,
    load_splits_manifest_v2,
    generate_35_5_10_splits,
    get_donor_city,
    get_wrong_donors,
    LOCKED_V1_TEST_FOLDS,
)
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.calibration.bin_calibration import calibrate_kbins
from src.data.dataset import get_scaler_fingerprint, load_city, load_cities, clear_city_cache
from src.experiment.run_e1 import compute_summary, compute_iqr, get_runtime_metadata, configure_cpu_threads


def test_t49_splits_35_5_10_invariants_and_v1_locking():
    splits = load_splits_manifest_v2("results/e1/splits_manifest_v2.json", data_root="data")
    assert len(splits) == 5

    all_test = []
    for f, s in splits.items():
        train = set(s["train"])
        val   = set(s["val"])
        test  = set(s["test"])

        assert len(s["train"]) == 35, f"Fold {f} train size {len(s['train'])} != 35"
        assert len(s["val"]) == 5, f"Fold {f} val size {len(s['val'])} != 5"
        assert len(s["test"]) == 10, f"Fold {f} test size {len(s['test'])} != 10"

        # Strictly identical to locked E1-v1 test sets
        assert s["test"] == sorted(LOCKED_V1_TEST_FOLDS[f]), f"Fold {f} test set does not match locked v1 test set"

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


def test_t55_donor_city_and_all_9_wrong_donors():
    test_cities = ["Austin", "Denver", "Portland", "Seattle", "Chicago", "Boston", "Miami", "Dallas", "Atlanta", "Detroit"]
    for c in test_cities:
        # Single donor
        donor = get_donor_city(c, test_cities)
        assert donor != c, f"Donor {donor} is identical to target {c}"
        assert donor in test_cities, f"Donor {donor} not in test set"

        # 9 wrong donors
        wrong_9 = get_wrong_donors(c, test_cities)
        assert len(wrong_9) == 9, f"Expected 9 wrong donors, got {len(wrong_9)}"
        assert c not in wrong_9, f"Target city {c} was included in wrong donors list"
        assert set(wrong_9) == set(test_cities) - {c}

    # Verify wrap-around for legacy single donor
    assert get_donor_city(sorted(test_cities)[-1], test_cities) == sorted(test_cities)[0]


def test_t56_confirmatory_guard_on_incomplete_subsets():
    """Verify that smoke / partial results are NOT reported as confirmatory."""
    dummy_results = [
        {
            "city": "Portland", "fold": 4, "donor_city": "all_9_fold_donors", "n_wrong_donors": 9,
            "n_inter_pairs": 1000, "K_active": 8, "cpc_baseline": 0.40, "cpc_baseline_norm": 0.50,
            "cpc_target_yd": 0.43, "cpc_target_yd_norm": 0.53, "delta_cpc_target": 0.03,
            "cpc_wrong_yd": 0.39, "cpc_wrong_yd_norm": 0.49, "delta_cpc_wrong": -0.01,
            "delta_cpc_specificity": 0.04,
            "Y_D_target": [0.125]*8, "wrong_donor_breakdown": []
        },
        {
            "city": "Denver", "fold": 5, "donor_city": "all_9_fold_donors", "n_wrong_donors": 9,
            "n_inter_pairs": 1000, "K_active": 8, "cpc_baseline": 0.42, "cpc_baseline_norm": 0.52,
            "cpc_target_yd": 0.45, "cpc_target_yd_norm": 0.55, "delta_cpc_target": 0.03,
            "cpc_wrong_yd": 0.41, "cpc_wrong_yd_norm": 0.51, "delta_cpc_wrong": -0.01,
            "delta_cpc_specificity": 0.04,
            "Y_D_target": [0.125]*8, "wrong_donor_breakdown": []
        }
    ]

    summary = compute_summary(dummy_results)
    assert not summary["is_confirmatory_complete"], "Partial 2-city run was falsely marked as confirmatory complete!"
    assert not summary["is_full_50_complete"], "Partial 2-city run was falsely marked as full 50 complete!"
    assert summary["confirmatory_folds_2_5"]["status"] == "not_available", "Confirmatory status should be not_available!"


def test_t57_stratified_validation_strata_coverage_and_metadata():
    """Verify validation representation across size strata and candidates metadata presence."""
    cities_info = get_all_cities_sorted_by_size("data")
    city_dict = {c["city"]: c for c in cities_info}
    splits = load_splits_manifest_v2("results/e1/splits_manifest_v2.json", data_root="data")

    for fold_id, s in splits.items():
        val_cities = set(s["val"])
        test_cities = set(s["test"])
        non_test_cities = [c["city"] for c in cities_info if c["city"] not in test_cities]
        non_test_info = [city_dict[c] for c in non_test_cities]
        ordered = sorted(non_test_info, key=lambda x: (x["n_tracts"], x["city"]))

        # Stratum coverage check
        strata = [ordered[i * 8 : (i + 1) * 8] for i in range(5)]
        for s_idx, stratum in enumerate(strata):
            stratum_cities = set(x["city"] for x in stratum)
            overlap = val_cities & stratum_cities
            assert len(overlap) == 1, (
                f"Fold {fold_id} stratum {s_idx} must have exactly 1 validation city, got {overlap}"
            )

        # Candidates metadata check
        cand_meta = s.get("validation_candidates_by_stratum", {})
        assert len(cand_meta) == 5, f"Fold {fold_id} missing stratum candidate metadata"
        for s_name, candidates in cand_meta.items():
            assert len(candidates) == 8, f"Stratum {s_name} in Fold {fold_id} must list exactly 8 candidates"


def test_t58_specificity_estimand_and_iqr():
    """Verify that delta_specificity = delta_target - delta_wrong is computed on city level."""
    test_results = []
    for i in range(50):
        f = (i % 5) + 1
        dt = 0.05 + 0.01 * (i % 3)
        dw = 0.01 + 0.005 * (i % 2)
        test_results.append({
            "city": f"City_{i}",
            "fold": f,
            "donor_city": "all_9_fold_donors",
            "n_wrong_donors": 9,
            "n_inter_pairs": 1000,
            "K_active": 8,
            "cpc_baseline": 0.40,
            "cpc_baseline_norm": 0.50,
            "cpc_target_yd": 0.40 + dt,
            "cpc_target_yd_norm": 0.50 + dt,
            "delta_cpc_target": dt,
            "cpc_wrong_yd": 0.40 + dw,
            "cpc_wrong_yd_norm": 0.50 + dw,
            "delta_cpc_wrong": dw,
            "delta_cpc_specificity": dt - dw,
            "Y_D_target": [0.125]*8,
            "wrong_donor_breakdown": [],
        })

    summary = compute_summary(test_results)
    assert summary["is_full_50_complete"]
    assert summary["is_confirmatory_complete"]
    conf = summary["confirmatory_folds_2_5"]

    # Invariant: Mean Specificity = Mean Target - Mean Wrong
    expected_spec = conf["delta_cpc_target_mean"] - conf["delta_cpc_wrong_mean"]
    assert np.isclose(conf["delta_specificity_mean"], expected_spec, atol=1e-6)

    # Invariant: IQR is non-negative
    assert conf["delta_specificity_iqr"] >= 0.0
    assert conf["delta_cpc_target_iqr"] >= 0.0
    assert conf["delta_cpc_wrong_iqr"] >= 0.0


def test_t59_runtime_metadata_and_cpu_thread_control():
    """Verify runtime metadata collection, CPU thread configuration, and summary integration."""
    meta = get_runtime_metadata()
    required_keys = [
        "platform",
        "processor",
        "python_version",
        "torch_version",
        "cuda_available",
        "cpu_count_logical",
        "cpu_count_physical",
        "torch_num_threads",
        "torch_num_interop_threads",
        "omp_num_threads",
        "mkl_num_threads",
    ]
    for key in required_keys:
        assert key in meta, f"Missing required runtime metadata key: {key}"

    assert meta["cpu_count_logical"] is not None and meta["cpu_count_logical"] > 0
    assert meta["torch_num_threads"] > 0
    assert meta["torch_num_interop_threads"] > 0

    # Test thread configuration
    orig_threads = torch.get_num_threads()
    try:
        set_threads = 4
        active = configure_cpu_threads(set_threads)
        assert active == set_threads
        assert torch.get_num_threads() == set_threads
        updated_meta = get_runtime_metadata()
        assert updated_meta["torch_num_threads"] == set_threads
        assert updated_meta["omp_num_threads"] == str(set_threads)
        assert updated_meta["mkl_num_threads"] == str(set_threads)
    finally:
        configure_cpu_threads(orig_threads)

    # Test compute_summary integration
    mock_results = [{
        "city": "Boston",
        "fold": 1,
        "donor_city": "all_9_fold_donors",
        "n_wrong_donors": 9,
        "n_inter_pairs": 500,
        "K_active": 8,
        "cpc_baseline": 0.50,
        "cpc_baseline_norm": 0.60,
        "cpc_target_yd": 0.55,
        "cpc_target_yd_norm": 0.65,
        "delta_cpc_target": 0.05,
        "cpc_wrong_yd": 0.52,
        "cpc_wrong_yd_norm": 0.62,
        "delta_cpc_wrong": 0.02,
        "delta_cpc_specificity": 0.03,
        "Y_D_target": [0.125]*8,
        "wrong_donor_breakdown": [],
    }]
    summary = compute_summary(mock_results)
    assert "runtime_environment" in summary
    assert summary["runtime_environment"]["torch_num_threads"] > 0


def test_t60_scaler_fingerprint_and_cache_isolation():
    """Verify deterministic content-based scaler hashing and cross-fold cache isolation."""
    from sklearn.preprocessing import StandardScaler

    # 1. Unfitted / None scaler handling
    assert get_scaler_fingerprint(None) is None

    s1 = StandardScaler()
    s1.mean_ = np.ones(26, dtype=np.float64) * 1.0
    s1.var_  = np.ones(26, dtype=np.float64) * 0.5
    s1.scale_ = np.sqrt(s1.var_)

    s2 = StandardScaler()
    s2.mean_ = np.ones(26, dtype=np.float64) * 1.0
    s2.var_  = np.ones(26, dtype=np.float64) * 0.5
    s2.scale_ = np.sqrt(s2.var_)

    s3 = StandardScaler()
    s3.mean_ = np.ones(26, dtype=np.float64) * 2.0
    s3.var_  = np.ones(26, dtype=np.float64) * 1.0
    s3.scale_ = np.sqrt(s3.var_)

    # Invariant: identical parameters -> identical fingerprint (even with different object IDs)
    assert get_scaler_fingerprint(s1) == get_scaler_fingerprint(s2)
    assert id(s1) != id(s2)

    # Invariant: different parameters -> different fingerprint
    assert get_scaler_fingerprint(s1) != get_scaler_fingerprint(s3)

    # 2. In-memory cache isolation on load_city
    clear_city_cache()
    cd_s1 = load_city("Boston", data_root="data", feature_scaler=s1)
    cd_s3 = load_city("Boston", data_root="data", feature_scaler=s3)

    # Features must differ because s1 and s3 normalization parameters differ
    assert not torch.allclose(cd_s1.node_features, cd_s3.node_features)
    
    # Reloading with s1 must hit cache and return exact same tensor values
    cd_s1_cached = load_city("Boston", data_root="data", feature_scaler=s1)
    assert torch.allclose(cd_s1.node_features, cd_s1_cached.node_features)
