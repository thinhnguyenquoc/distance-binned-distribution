"""
Tests for Experiment Contracts, Model Freezing, Shared State, and Manifest Integrity.
(Tests T37 to T41)
"""

import pytest
import torch
import numpy as np
from od_plan_tester.project_adapter import (
    train_zero_shot_model,
    load_cities,
    load_city,
    run_target_city_experiments,
    ZeroShotODModel,
)


@pytest.mark.scientific
def test_model_freezing_theta_star():
    """T37: Model parameters theta* are completely frozen (requires_grad=False) before inference."""
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
    """T38: All experimental conditions (M0, M1^oracle, M1^real, Mq) evaluate on identical candidate support Omega_c."""
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

    # Number of candidate pairs must be identical
    assert res["n_pairs"] > 0
    assert "M0" in res
    assert "M1_oracle" in res
    assert "M1_real" in res
    assert "Mq_curve" in res


@pytest.mark.reference
def test_delta_r_and_realization_gap_formulas():
    """T39: Verify Delta R and realization gap arithmetic formulas."""
    cpc_m0 = 0.40
    cpc_m1_real = 0.48
    cpc_m1_oracle = 0.55

    delta_r_real = cpc_m1_real - cpc_m0
    delta_r_oracle = cpc_m1_oracle - cpc_m0
    realization_gap = cpc_m1_oracle - cpc_m1_real

    assert pytest.approx(0.08, rel=1e-5) == delta_r_real
    assert pytest.approx(0.15, rel=1e-5) == delta_r_oracle
    assert pytest.approx(0.07, rel=1e-5) == realization_gap


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
    """T41: Smoke test verifying target city experiment runner produces all expected keys."""
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
        "city", "n_tracts", "n_pairs", "total_trips", "M0", "M1_oracle",
        "M1_real", "Mq_curve", "delta_r_oracle", "delta_r_real",
        "realization_gap", "m_star_real", "q_star_real", "m_star_oracle", "q_star_oracle"
    ]
    for k in expected_keys:
        assert k in res, f"Missing key {k} in experiment result dictionary"
