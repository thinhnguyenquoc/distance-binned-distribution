import sys
import re

file_path = 'od_plan_tester/tests/test_e1_contracts.py'
with open(file_path, 'r') as f:
    content = f.read()

for i in range(60, 48, -1):
    old_str = f"test_t{i}_"
    new_str = f"test_t{i+2}_"
    content = content.replace(old_str, new_str)
    
    old_str_doc = f"T{i}:"
    new_str_doc = f"T{i+2}:"
    content = content.replace(old_str_doc, new_str_doc)

with open(file_path, 'w') as f:
    f.write(content)
