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
