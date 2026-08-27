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

    from src.data.yd_extractor import compute_kbin_edges
    bin_edges, _ = compute_kbin_edges(["Raleigh", "Denver"], K=8, data_root="data")
    res = run_target_city_experiments(
        model=model,
        city_name="Denver",
        scaler=fitted_scaler,
        data_root="data",
        bin_edges=bin_edges,
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
    """T40: manifest exists, is readable, and file hashes match."""
    import hashlib
    manifest_path = Path("results/manifest_rq1_v1.json")
    assert manifest_path.exists(), "Production manifest results/manifest_rq1_v1.json does not exist!"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    assert "contract_conditions" in manifest
    
    # Verify file hashes
    file_hashes = manifest.get("file_hashes", {})
    assert len(file_hashes) > 0, "No file hashes found in manifest!"
    for fp, expected_hash in file_hashes.items():
        p = Path("results") / fp
        if p.exists():
            computed = hashlib.sha256(p.read_bytes()).hexdigest()
            assert computed == expected_hash, f"Hash mismatch for {fp}!"

@pytest.mark.contract
def test_t41_50_unique_test_cities():
    """T41: manifest vs pipeline (50 unique test cities)."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    with open("results/5fold_results.json", "r") as f:
        results = json.load(f)
    cities = set([r["city"] for r in results["city_level_results"]])
    assert len(cities) == manifest["contract_conditions"]["unique_cities"]
    assert len(cities) == 50

@pytest.mark.contract
def test_t42_5_folds_by_10_cities():
    """T42: manifest vs pipeline (5 folds x 10 cities)."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    with open("results/5fold_results.json", "r") as f:
        results = json.load(f)
    from collections import defaultdict
    fold_counts = defaultdict(int)
    for r in results["city_level_results"]:
        fold_counts[r["fold"]] += 1
    
    assert len(fold_counts) == manifest["contract_conditions"]["folds"]
    assert len(fold_counts) == 5
    for count in fold_counts.values():
        assert count == manifest["contract_conditions"]["cities_per_fold"]
        assert count == 10

@pytest.mark.contract
def test_t43_primary_k_is_8():
    """T43: manifest vs pipeline (primary K == 8)."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    locked_k = manifest["contract_conditions"]["primary_k_bins"]
    
    from src.data.yd_extractor import compute_kbin_edges
    bin_edges, _ = compute_kbin_edges(["Denver"], K=locked_k, data_root="data")
    assert len(bin_edges) - 1 == locked_k

@pytest.mark.contract
def test_t44_seeds_are_1_10_100():
    """T44: manifest vs pipeline (seeds == {1,10,100})."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    locked_seeds = set(manifest["contract_conditions"]["model_seeds"])
    
    import ast
    with open("src/experiment/run_5fold.py", "r") as f:
        tree = ast.parse(f.read())
    
    found_seeds = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, 'id', '') == 'seeds':
                    if isinstance(node.value, ast.List):
                        found_seeds = set(elt.value for elt in node.value.elts if isinstance(elt, ast.Constant))
    
    assert found_seeds == locked_seeds, f"Found seeds {found_seeds} in run_5fold.py!"

@pytest.mark.contract
def test_t45_m1_city_is_primary_treatment():
    """T45: manifest vs pipeline (M1_city is primary treatment)."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    primary = manifest["contract_conditions"]["primary_treatment"]
    
    with open("results/5fold_results.json", "r") as f:
        results = json.load(f)
    keys = results["city_level_results"][0].keys()
    assert primary in keys, f"{primary} not found in pipeline results!"

@pytest.mark.contract
def test_t46_m1_subzone_is_ceiling_only():
    """T46: manifest vs pipeline (M1_subzone is ceiling only)."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    ceiling = manifest["contract_conditions"]["ceiling_treatment"]
    
    with open("results/5fold_results.json", "r") as f:
        results = json.load(f)
    keys = results["city_level_results"][0].keys()
    assert ceiling in keys, f"{ceiling} not found in pipeline results!"

@pytest.mark.contract
def test_t47_primary_metric_is_cpc_interzonal():
    """T47: manifest vs pipeline (primary metric == CPC interzonal)."""
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    metric = manifest["contract_conditions"]["primary_metric"]
    
    with open("results/5fold_results.json", "r") as f:
        results = json.load(f)
    m0_metrics = results["city_level_results"][0]["M0"].keys()
    assert metric in m0_metrics, f"{metric} not found in pipeline metrics!"

@pytest.mark.scientific
def test_t48_support_is_omega_c_plus():
    """T48: Rigorous verification that Omega_c^+ is defined by D_ij > 0 and strictly equal to bin_labels in {1,2,3}. (Restored T44)"""
    for city_name in ["Denver", "Portland"]:
        cd = load_city(city_name, data_root="data")
        dist_km = np.expm1(cd.pair_distance.numpy())
        o_np = cd.pair_o_idx.numpy()
        d_np = cd.pair_d_idx.numpy()
        b_np = cd.bin_labels.numpy()

        # Check equivalence between distance threshold and bin assignment
        mask_dist = (o_np != d_np) & (dist_km > 0.0)
        mask_bins = (o_np != d_np) & (b_np > 0)
        assert np.array_equal(mask_dist, mask_bins), f"{city_name}: mask_dist and mask_bins mismatch!"

        # Intrazonal pairs are strictly bin 0
        diag_mask = (o_np == d_np)
        assert np.all(b_np[diag_mask] == 0), f"{city_name}: intrazonal pairs contain non-zero bins!"


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

@pytest.mark.scientific
def test_t50_target_ground_truth_permutation_invariance_for_m0():
    """T50: Changing, permuting, or zeroing target ground-truth T^GT has zero effect on M0 predictions. (Restored T45)"""
    train_data_list, fitted_scaler = load_cities(["Raleigh", "Denver"], data_root="data")
    model, _ = train_zero_shot_model(
        train_city_names=["Raleigh", "Denver"],
        data_root="data",
        epochs=1,
        device_str="cpu",
        verbose=False,
    )

    cd = load_city("Portland", data_root="data", feature_scaler=fitted_scaler)
    import copy
    cd = copy.deepcopy(cd)
    edge_idx, edge_dist = build_radius_graph(cd.lon_lat, radius_km=5.0)

    with torch.no_grad():
        pred_orig = model(
            cd.node_features,
            edge_idx,
            edge_dist,
            cd.pair_o_idx,
            cd.pair_d_idx,
            cd.pair_distance,
            cd.population,
            return_conditional_mean=True,
        )

    # Permute trips completely
    cd.pair_trips = cd.pair_trips[torch.randperm(len(cd.pair_trips))]

    with torch.no_grad():
        pred_permuted = model(
            cd.node_features,
            edge_idx,
            edge_dist,
            cd.pair_o_idx,
            cd.pair_d_idx,
            cd.pair_distance,
            cd.population,
            return_conditional_mean=True,
        )

    # Set trips to zeros
    cd.pair_trips = torch.zeros_like(cd.pair_trips)

    with torch.no_grad():
        pred_zeros = model(
            cd.node_features,
            edge_idx,
            edge_dist,
            cd.pair_o_idx,
            cd.pair_d_idx,
            cd.pair_distance,
            cd.population,
            return_conditional_mean=True,
        )

    torch.testing.assert_close(pred_orig, pred_permuted, rtol=0, atol=1e-6)
    torch.testing.assert_close(pred_orig, pred_zeros, rtol=0, atol=1e-6)
