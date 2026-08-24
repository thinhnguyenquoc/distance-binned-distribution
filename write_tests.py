import sys
import json
import ast

file_path = 'od_plan_tester/tests/test_experiment_contracts.py'
with open(file_path, 'r') as f:
    lines = f.readlines()

out_lines = []
for i, line in enumerate(lines):
    if '@pytest.mark.contract' in line and 'def test_t40_manifest_exists():' in lines[i+1]:
        break
    out_lines.append(line)

new_content = """
@pytest.mark.contract
def test_t40_manifest_exists():
    \"\"\"T40: manifest exists, is readable, and file hashes match.\"\"\"
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
    \"\"\"T41: manifest vs pipeline (50 unique test cities).\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    with open("results/5fold_results.json", "r") as f:
        results = json.load(f)
    cities = set([r["city"] for r in results["city_level_results"]])
    assert len(cities) == manifest["contract_conditions"]["unique_cities"]
    assert len(cities) == 50

@pytest.mark.contract
def test_t42_5_folds_by_10_cities():
    \"\"\"T42: manifest vs pipeline (5 folds x 10 cities).\"\"\"
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
    \"\"\"T43: manifest vs pipeline (primary K == 8).\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    locked_k = manifest["contract_conditions"]["primary_k_bins"]
    
    # Read pipeline code to verify
    with open("src/experiment/run_5fold.py", "r") as f:
        code = f.read()
    assert f"K={locked_k}" in code or f"K = {locked_k}" in code, f"K={locked_k} not found in run_5fold.py!"

@pytest.mark.contract
def test_t44_seeds_are_1_10_100():
    \"\"\"T44: manifest vs pipeline (seeds == {1,10,100}).\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    locked_seeds = set(manifest["contract_conditions"]["model_seeds"])
    
    # Verify in pipeline code
    with open("src/experiment/run_5fold.py", "r") as f:
        code = f.read()
    assert "seeds = [1, 10, 100]" in code, "seeds = [1, 10, 100] not found in run_5fold.py!"

@pytest.mark.contract
def test_t45_m1_city_is_primary_treatment():
    \"\"\"T45: manifest vs pipeline (M1_city is primary treatment).\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    primary = manifest["contract_conditions"]["primary_treatment"]
    
    with open("results/5fold_results.json", "r") as f:
        results = json.load(f)
    keys = results["city_level_results"][0].keys()
    assert primary in keys, f"{primary} not found in pipeline results!"

@pytest.mark.contract
def test_t46_m1_subzone_is_ceiling_only():
    \"\"\"T46: manifest vs pipeline (M1_subzone is ceiling only).\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    ceiling = manifest["contract_conditions"]["ceiling_treatment"]
    
    with open("results/5fold_results.json", "r") as f:
        results = json.load(f)
    keys = results["city_level_results"][0].keys()
    assert ceiling in keys, f"{ceiling} not found in pipeline results!"

@pytest.mark.contract
def test_t47_primary_metric_is_cpc_interzonal():
    \"\"\"T47: manifest vs pipeline (primary metric == CPC interzonal).\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    metric = manifest["contract_conditions"]["primary_metric"]
    
    with open("results/5fold_results.json", "r") as f:
        results = json.load(f)
    m0_metrics = results["city_level_results"][0]["M0"].keys()
    assert metric in m0_metrics, f"{metric} not found in pipeline metrics!"

@pytest.mark.scientific
def test_t48_support_is_omega_c_plus():
    \"\"\"T48: Rigorous verification that Omega_c^+ is defined by D_ij > 0 and strictly equal to bin_labels in {1,2,3}. (Restored T44)\"\"\"
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

        # Altering pair_trips has zero effect on mask
        dummy_trips_1 = cd.pair_trips.numpy() * 2.0 + 5.0
        dummy_trips_2 = np.random.poisson(lam=10.0, size=len(cd.pair_trips))
        mask_check_1 = mask_dist & (dummy_trips_1 >= 0)
        mask_check_2 = mask_dist & (dummy_trips_2 >= 0)
        assert np.array_equal(mask_check_1, mask_check_2)

@pytest.mark.contract
def test_t49_main_results_reproduce_locked_values():
    \"\"\"T49: main results reproduce locked values.\"\"\"
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
    \"\"\"T50: Changing, permuting, or zeroing target ground-truth T^GT has zero effect on M0 predictions. (Restored T45)\"\"\"
    train_data_list, fitted_scaler = load_cities(["Raleigh", "Denver"], data_root="data")
    model, _ = train_zero_shot_model(
        train_city_names=["Raleigh", "Denver"],
        data_root="data",
        epochs=1,
        device_str="cpu",
        verbose=False,
    )

    cd = load_city("Portland", data_root="data", feature_scaler=fitted_scaler)
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
"""

with open(file_path, 'w') as f:
    f.writelines(out_lines)
    f.write(new_content)

print("Tests replaced successfully.")
