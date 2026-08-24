import sys
import json
from pathlib import Path

file_path = 'od_plan_tester/tests/test_experiment_contracts.py'
with open(file_path, 'r') as f:
    lines = f.readlines()

out_lines = []
for i, line in enumerate(lines):
    if '@pytest.mark.skip(reason="Obsolete manifest logic for Q3 scope")' in line or 'def test_experiment_manifest_reproducibility():' in line:
        break
    out_lines.append(line)

new_content = """
@pytest.mark.contract
def test_t40_manifest_exists():
    \"\"\"T40: manifest exists and is readable.\"\"\"
    manifest_path = Path("results/manifest_rq1_v1.json")
    assert manifest_path.exists(), "Production manifest results/manifest_rq1_v1.json does not exist!"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    assert "contract_conditions" in manifest

@pytest.mark.contract
def test_t41_50_unique_test_cities():
    \"\"\"T41: 50 unique test cities.\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["total_cities"] == 50
    assert manifest["contract_conditions"]["unique_cities"] == 50

@pytest.mark.contract
def test_t42_5_folds_by_10_cities():
    \"\"\"T42: 5 folds x 10 cities.\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["folds"] == 5
    assert manifest["contract_conditions"]["cities_per_fold"] == 10

@pytest.mark.contract
def test_t43_primary_k_is_8():
    \"\"\"T43: primary K == 8.\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["primary_k_bins"] == 8

@pytest.mark.contract
def test_t44_seeds_are_1_10_100():
    \"\"\"T44: seeds == {1,10,100}.\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert set(manifest["contract_conditions"]["model_seeds"]) == {1, 10, 100}

@pytest.mark.contract
def test_t45_m1_city_is_primary_treatment():
    \"\"\"T45: M1_city is primary treatment.\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["primary_treatment"] == "M1_city_oracle_obs"

@pytest.mark.contract
def test_t46_m1_subzone_is_ceiling_only():
    \"\"\"T46: M1_subzone is ceiling only.\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["ceiling_treatment"] == "M1_subzone_oracle_obs"

@pytest.mark.contract
def test_t47_primary_metric_is_cpc_interzonal():
    \"\"\"T47: primary metric == CPC interzonal.\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["primary_metric"] == "cpc_inter"

@pytest.mark.contract
def test_t48_support_is_omega_c_plus():
    \"\"\"T48: support == Omega_c+.\"\"\"
    with open("results/manifest_rq1_v1.json", "r") as f:
        manifest = json.load(f)
    assert manifest["contract_conditions"]["support_domain"] == "Omega_c_plus"

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
"""
with open(file_path, 'w') as f:
    f.writelines(out_lines)
    f.write(new_content)
