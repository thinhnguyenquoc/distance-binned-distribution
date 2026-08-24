import sys

file_path = 'od_plan_tester/tests/test_experiment_contracts.py'
with open(file_path, 'r') as f:
    content = f.read()

old_import = "from src.experiment.run_experiment import compute_kbin_edges"
new_import = "from src.data.yd_extractor import compute_kbin_edges"

content = content.replace(old_import, new_import)
with open(file_path, 'w') as f:
    f.write(content)
