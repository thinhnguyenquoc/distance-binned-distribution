import sys

file_path = 'od_plan_tester/tests/test_experiment_contracts.py'
with open(file_path, 'r') as f:
    content = f.read()

# Replace cd.pair_trips permutation with deepcopy
old = "cd = load_city(\"Portland\", data_root=\"data\", feature_scaler=fitted_scaler)"
new = "cd = load_city(\"Portland\", data_root=\"data\", feature_scaler=fitted_scaler)\n    import copy\n    cd = copy.deepcopy(cd)"

content = content.replace(old, new)

with open(file_path, 'w') as f:
    f.write(content)
