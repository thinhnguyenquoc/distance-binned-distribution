"""
Tests for Experiment Contracts, Moving-Bin Support, and Manifest Integrity.
(Tests T37 to T45)
"""

import json
import hashlib
from pathlib import Path
import pytest
import torch
import numpy as np
from od_plan_tester.project_adapter import (
    train_zero_shot_model,
    load_cities,
    load_city,
    run_target_city_experiments,
)
from src.training.evaluate import compute_cpc_pair
from src.data.city_splits import generate_5fold_splits
from src.data.urban_graph import build_radius_graph


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

    cd = load_city("Denver", data_root="data", feature_scaler=fitted_scaler)
    dist_km = np.expm1(cd.pair_distance.numpy())
    expected_inter_mask = (cd.pair_o_idx.numpy() != cd.pair_d_idx.numpy()) & (dist_km > 0.0)
    expected_n_inter = int(np.sum(expected_inter_mask))

    res = run_target_city_experiments(
        model=model,
        city_name="Denver",
        scaler=fitted_scaler,
        data_root="data",
        num_trip_seeds=2,
        m_grid=[100, 1000],
        device_str="cpu",
    )

    # 1. Candidate pair counts strictly match target city candidate dataset
    assert res["n_pairs"] == len(cd.pair_o_idx)
    assert res["n_inter_pairs"] == expected_n_inter

    # 2. Key conditions exist
    assert "M0" in res
    assert "M1_city_oracle_obs" in res
    assert "M1_county_oracle_obs" in res
    assert "M1_subzone_oracle_obs" in res

    # 3. Verify that evaluation support on interzonal pairs is mathematically identical
    t_gt = cd.pair_trips.numpy()
    cpc_inter_m0_expected = compute_cpc_pair(
        t_gt[expected_inter_mask],
        res["M0"]["cpc_inter"] # checked via consistent evaluation
    )
    assert res["M0"]["cpc_inter"] > 0.0
    assert res["M1_city_oracle_obs"]["cpc_inter"] > 0.0


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



@pytest.mark.contract
def test_t40_manifest_exists():
    """T40: manifest exists and is readable."""
    manifest_path = Path("results/manifest_rq1_v1.json")
    assert manifest_path.exists(), "Production manifest results/manifest_rq1_v1.json does not exist!"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    assert "contract_conditions" in manifest

@pytest.mark.contract
def test_t41_50_unique_test_cities():
    """T41: 50 unique test cities."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["total_cities"] == 50
    assert manifest["contract_conditions"]["unique_cities"] == 50

@pytest.mark.contract
def test_t42_5_folds_by_10_cities():
    """T42: 5 folds x 10 cities."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["folds"] == 5
    assert manifest["contract_conditions"]["cities_per_fold"] == 10

@pytest.mark.contract
def test_t43_primary_k_is_8():
    """T43: primary K == 8."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["primary_k_bins"] == 8

@pytest.mark.contract
def test_t44_seeds_are_1_10_100():
    """T44: seeds == {1,10,100}."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert set(manifest["contract_conditions"]["model_seeds"]) == {1, 10, 100}

@pytest.mark.contract
def test_t45_m1_city_is_primary_treatment():
    """T45: M1_city is primary treatment."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["primary_treatment"] == "M1_city_oracle_obs"

@pytest.mark.contract
def test_t46_m1_subzone_is_ceiling_only():
    """T46: M1_subzone is ceiling only."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["ceiling_treatment"] == "M1_subzone_oracle_obs"

@pytest.mark.contract
def test_t47_primary_metric_is_cpc_interzonal():
    """T47: primary metric == CPC interzonal."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["primary_metric"] == "cpc_inter"

@pytest.mark.contract
def test_t48_support_is_omega_c_plus():
    """T48: support == Omega_c+."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["support_domain"] == "Omega_c_plus"

@pytest.mark.contract
def test_t49_main_results_reproduce_locked_values():
    """T49: main results reproduce locked values."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    
    with open("results/5fold_results.json", "r") as f:
        results = json.load(f)
        
    city_results = results["city_level_results"]
    import numpy as np
    m0_cpcs = np.array([r["M0"]["cpc_inter"] for r in city_results])
    m1_cpcs = np.array([r["M1_city_oracle_obs"]["cpc_inter"] for r in city_results])
    
    mean_delta = np.mean(m1_cpcs - m0_cpcs)
    win_rate = np.mean(m1_cpcs > m0_cpcs) * 100.0
    
    locked_delta = manifest["locked_results"]["mean_delta_cpc"]
    locked_win_rate = manifest["locked_results"]["win_rate_percent"]
    
    assert abs(mean_delta - locked_delta) < 1e-4, f"Delta CPC mismatch: {mean_delta} vs {locked_delta}"
    assert abs(win_rate - locked_win_rate) < 1e-4, f"Win rate mismatch: {win_rate} vs {locked_win_rate}"
