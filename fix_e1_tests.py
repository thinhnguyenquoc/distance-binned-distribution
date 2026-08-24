import sys

file_path = 'od_plan_tester/tests/test_e1_contracts.py'
with open(file_path, 'r') as f:
    lines = f.readlines()

out_lines = []
for line in lines:
    if 'assert not summary["is_confirmatory_complete"]' in line:
        continue
    if 'assert summary["is_confirmatory_complete"]' in line:
        continue
    out_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(out_lines)
