import sys

file_path = 'od_plan_tester/tests/test_e1_contracts.py'
with open(file_path, 'r') as f:
    content = f.read()

# Fix T60 variable from conf to full
old = """    # Invariant: Mean Specificity = Mean Target - Mean Wrong
    expected_spec = conf["delta_cpc_target_mean"] - conf["delta_cpc_wrong_mean"]
    assert abs(conf["delta_cpc_specificity_mean"] - expected_spec) < 1e-7
"""
new = """    # Invariant: Mean Specificity = Mean Target - Mean Wrong
    full = summary["full_50_cities"]
    expected_spec = full["delta_cpc_target_mean"] - full["delta_cpc_wrong_mean"]
    assert abs(full["delta_cpc_specificity_mean"] - expected_spec) < 1e-7
"""

content = content.replace(old, new)
with open(file_path, 'w') as f:
    f.write(content)
