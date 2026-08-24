import sys

file_path = 'od_plan_tester/tests/test_e1_contracts.py'
with open(file_path, 'r') as f:
    content = f.read()

old_str = """    # Invariant: Mean Specificity = Mean Target - Mean Wrong
    expected_spec = conf["delta_cpc_target_mean"] - conf["delta_cpc_wrong_mean"]
    assert np.isclose(conf["delta_specificity_mean"], expected_spec, atol=1e-6)

    # Invariant: IQR is non-negative
    assert conf["delta_specificity_iqr"] >= 0.0"""

new_str = """    # Invariant: Mean Specificity = Mean Target - Mean Wrong
    full = summary["full_50_cities"]
    expected_spec = full["delta_cpc_target_mean"] - full["delta_cpc_wrong_mean"]
    assert np.isclose(full["delta_specificity_mean"], expected_spec, atol=1e-6)

    # Invariant: IQR is non-negative
    assert full["delta_specificity_iqr"] >= 0.0"""

content = content.replace(old_str, new_str)
with open(file_path, 'w') as f:
    f.write(content)
