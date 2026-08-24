import sys

file_path = 'od_plan_tester/tests/test_e1_contracts.py'
with open(file_path, 'r') as f:
    content = f.read()

content = content.replace('conf["delta_cpc_target_iqr"]', 'summary["delta_cpc_target_iqr"]')
content = content.replace('conf["delta_cpc_wrong_iqr"]', 'summary["delta_cpc_wrong_iqr"]')

with open(file_path, 'w') as f:
    f.write(content)
