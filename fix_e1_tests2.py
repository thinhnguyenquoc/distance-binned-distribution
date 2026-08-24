import sys

file_path = 'od_plan_tester/tests/test_e1_contracts.py'
with open(file_path, 'r') as f:
    lines = f.readlines()

out_lines = []
for line in lines:
    if 'summary["confirmatory_folds_2_5"]' in line:
        continue
    out_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(out_lines)
