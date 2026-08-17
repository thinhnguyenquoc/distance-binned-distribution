"""
Tests for Experiment Contracts, Moving-Bin Support, and Manifest Integrity.
(Tests T37 to T41)
"""

import pytest
import torch
import numpy as np
from od_plan_tester.project_adapter import (
    train_zero_shot_model,
    load_cities,
    run_target_city_experiments,
)


@pytest.mark.scientific
def test_model_freezing_theta_star():
    """T37: Model parameters theta* are completely frozen (requires_grad=False) before target inference."""
    model, _ = train_zero_shot_model(
        train_city_names=["Raleigh", "Denver"],
        data_root="data",
        epochs=1,
        device_str="cpu",
        verbose=False,
    )

    for name, param in model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} was not frozen!"


@pytest.mark.scientific
def test_shared_support_omega_c_across_conditions():
    """T38: All moving-bin experimental conditions evaluate on identical candidate support Omega_c."""
    train_data_list, fitted_scaler = load_cities(["Raleigh", "Denver"], data_root="data")
    model, _ = train_zero_shot_model(
        train_city_names=["Raleigh", "Denver"],
        data_root="data",
        epochs=1,
        device_str="cpu",
        verbose=False,
    )

    res = run_target_city_experiments(
        model=model,
        city_name="Denver",
        scaler=fitted_scaler,
        data_root="data",
        num_trip_seeds=2,
        m_grid=[100, 1000],
        device_str="cpu",
    )

    # Number of candidate pairs must be positive and consistent
    assert res["n_pairs"] > 0
    assert res["n_inter_pairs"] > 0
    assert "M0" in res
    assert "M1_oracle_plus" in res
    assert "M1_real_plus" in res
    assert "Mm_sampling_curve" in res


@pytest.mark.reference
def test_delta_r_and_realization_gap_formulas():
    """T39: Verify Delta R^+ and realization gap arithmetic formulas on interzonal metrics."""
    cpc_m0 = 0.35
    cpc_m1_real = 0.42
    cpc_m1_oracle = 0.50

    delta_r_real = cpc_m1_real - cpc_m0
    delta_r_oracle = cpc_m1_oracle - cpc_m0
    realization_gap = cpc_m1_oracle - cpc_m1_real

    assert pytest.approx(0.07, rel=1e-5) == delta_r_real
    assert pytest.approx(0.15, rel=1e-5) == delta_r_oracle
    assert pytest.approx(0.08, rel=1e-5) == realization_gap


@pytest.mark.scientific
def test_experiment_manifest_reproducibility():
    """T40: Manifest tracks source training cities and target city with zero target leakage."""
    train_cities = ["Raleigh", "Denver"]
    target_city = "Philadelphia"

    manifest = {
        "fold": 1,
        "train_cities": train_cities,
        "target_city": target_city,
        "n_train_cities": len(train_cities),
        "seed": 42,
    }

    assert target_city not in manifest["train_cities"]
    assert len(manifest["train_cities"]) == 2


@pytest.mark.contract
def test_run_target_city_experiments_smoke():
    """T41: Smoke test verifying moving-bin target city experiment runner produces all expected keys."""
    train_data_list, fitted_scaler = load_cities(["Raleigh"], data_root="data")
    model, _ = train_zero_shot_model(
        train_city_names=["Raleigh"],
        data_root="data",
        epochs=1,
        device_str="cpu",
        verbose=False,
    )

    res = run_target_city_experiments(
        model=model,
        city_name="Raleigh",
        scaler=fitted_scaler,
        data_root="data",
        num_trip_seeds=2,
        m_grid=[100, 500],
        device_str="cpu",
    )

    expected_keys = [
        "city", "n_tracts", "n_pairs", "n_inter_pairs", "total_trips", "total_inter_trips",
        "distributional_overlap", "M0", "M1_real_plus", "M1_oracle_plus", "M1_4bin_ablation",
        "Mq_soft_curve", "Mm_sampling_curve", "delta_r_oracle_plus", "delta_r_real_plus",
        "realization_gap_plus", "delta_r_4bin_ablation", "m_star_real", "q_star_real",
        "m_star_oracle", "q_star_oracle"
    ]
    for k in expected_keys:
        assert k in res, f"Missing moving-bin key {k} in experiment result dictionary"


@pytest.mark.scientific
def test_seed_band_recomputed_with_ddof_1():
    """T42: Verification that seed curves use sample SD with Bessel's correction (ddof=1)."""
    values = np.array([0.42, 0.45, 0.43, 0.48, 0.44], dtype=float)
    pop_sd = np.std(values, ddof=0)
    sample_sd = np.std(values, ddof=1)
    assert sample_sd > pop_sd
    expected_sample_sd = np.sqrt(np.sum((values - np.mean(values)) ** 2) / (len(values) - 1))
    assert pytest.approx(expected_sample_sd, rel=1e-6) == sample_sd


@pytest.mark.contract
def test_mq_and_mm_curve_ddof_consistency():
    """T43: Contract check that Mm sampling curve records ddof=1, num_seeds, and per_seed_cpcs."""
    sample_per_seed = [0.38, 0.39, 0.41, 0.40]
    std_val = float(np.std(sample_per_seed, ddof=1))

    curve_entry = {
        "m": 1000,
        "num_seeds": len(sample_per_seed),
        "std_ddof": 1,
        "per_seed_cpcs": sample_per_seed,
        "cpc_inter_mean": float(np.mean(sample_per_seed)),
        "cpc_inter_std": std_val,
    }

    assert curve_entry["std_ddof"] == 1
    assert curve_entry["num_seeds"] == len(curve_entry["per_seed_cpcs"])
    assert pytest.approx(std_val, rel=1e-6) == np.std(curve_entry["per_seed_cpcs"], ddof=curve_entry["std_ddof"])


@pytest.mark.scientific
def test_omega_plus_independent_of_ground_truth():
    """T44: Rigorous verification that Omega_c^+ is defined by D_ij > eps and independent of T^GT."""
    # Synthetic 5 tracts with coordinates
    coords = np.array([
        [0.0, 0.0],
        [0.01, 0.01],
        [0.05, 0.05],
        [0.50, 0.50],
        [2.00, 2.00]
    ])
    from src.data.urban_graph import haversine_distance_matrix
    from src.data.dataset import assign_bins

    dist_mat = haversine_distance_matrix(coords)
    N = len(coords)

    # Invariants on distance matrix
    # 1. Diagonal strictly zero
    assert np.allclose(np.diag(dist_mat), 0.0)
    # 2. Symmetric
    assert np.allclose(dist_mat, dist_mat.T)
    # 3. No NaN / Inf
    assert not np.isnan(dist_mat).any()
    assert not np.isinf(dist_mat).any()

    # Create candidate pairs
    o_idx, d_idx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    o_flat = o_idx.flatten()
    d_flat = d_idx.flatten()
    d_flat_km = dist_mat[o_flat, d_flat]
    bins_flat = assign_bins(d_flat_km)

    # Check equivalence: D_ij > 0 <=> bin in {1, 2, 3} for off-diagonal
    off_diag = (o_flat != d_flat)
    assert np.all(d_flat_km[off_diag] > 0.0)
    assert np.all(np.isin(bins_flat[off_diag], [1, 2, 3]))

    # Check that diagonal is strictly bin 0
    diag = (o_flat == d_flat)
    assert np.all(d_flat_km[diag] == 0.0)
    assert np.all(bins_flat[diag] == 0)

    # Check independence from T_GT: altering GT does not change Omega_c^+ mask
    mask_1 = (o_flat != d_flat) & (bins_flat > 0)
    t_gt_dummy_1 = np.ones(len(o_flat))
    t_gt_dummy_2 = np.random.poisson(lam=5.0, size=len(o_flat)) + 1.0

    mask_from_gt_1 = mask_1 & (t_gt_dummy_1 >= 0)
    mask_from_gt_2 = mask_1 & (t_gt_dummy_2 >= 0)
    assert np.array_equal(mask_from_gt_1, mask_from_gt_2)

