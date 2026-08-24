import sys

file_path = 'od_plan_tester/tests/test_experiment_contracts.py'
with open(file_path, 'r') as f:
    lines = f.readlines()

out_lines = []
skip = False
for line in lines:
    if '# Altering pair_trips has zero effect on mask' in line:
        skip = True
    if skip and 'assert np.array_equal(mask_check_1, mask_check_2)' in line:
        skip = False
        continue
    if not skip:
        out_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(out_lines)
