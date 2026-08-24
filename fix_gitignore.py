import sys

with open('.gitignore', 'r') as f:
    lines = f.readlines()

out = []
for line in lines:
    if line.strip() == 'results/':
        out.append('# results/\n')
        out.append('!results/*.json\n')
    else:
        out.append(line)

with open('.gitignore', 'w') as f:
    f.writelines(out)
