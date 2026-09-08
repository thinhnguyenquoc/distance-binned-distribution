import sys, re
from pathlib import Path

content = Path('paper/full_paper_vi.md').read_text(encoding='utf-8')
lines = content.splitlines()

print('=== AUDIT MATH FORMULAS ===')
paren_matches = []
for idx, line in enumerate(lines, 1):
    m = re.findall(r'\\\((.+?)\\\)', line)
    if m:
        paren_matches.append((idx, m))
print(f'Lines with \\( ... \\): {len(paren_matches)}')
for idx, m in paren_matches:
    print(f'  L{idx}: {m}')

block_pattern = re.compile(r'\$\$\$', re.DOTALL)
block_maths = list(block_pattern.finditer(content))
print(f'Total block math: {len(block_maths)}')
for i, bm in enumerate(block_maths, 1):
    expr = bm.group(1).strip()
    if expr.count('{') != expr.count('}'):
        print(f'  [ERR] Block {i}: brace mismatch!')

inline_pattern = re.compile(r'(?<!\\)\(.+?)(?<!\s)(?<!\\)\$')
inline_maths = list(inline_pattern.finditer(content))
print(f'Total inline math: {len(inline_maths)}')
for i, im in enumerate(inline_maths, 1):
    expr = im.group(1)
    if expr.count('{') != expr.count('}'):
        print(f'  [ERR] Inline {i}: brace mismatch in {expr}')

print('
=== AUDIT TABLES ===')
tables = []
cur = []
start = 0
for idx, line in enumerate(lines, 1):
    if line.strip().startswith('|'):
        if not cur:
            start = idx
        cur.append((idx, line))
    else:
        if cur:
            tables.append((start, cur))
            cur = []

for t_start, t_lines in tables:
    h = t_lines[0][1]
    cols = len([c for c in h.strip().split('|')[1:-1]])
    errs = []
    for l_idx, l_str in t_lines:
        c = len([x for x in l_str.strip().split('|')[1:-1]])
        if c != cols:
            errs.append((l_idx, c))
    print(f'Table at L{t_start}: {len(t_lines)} rows, {cols} cols, errs={len(errs)}')
    for e in errs:
        print(f'  L{e[0]}: has {e[1]} cols')
