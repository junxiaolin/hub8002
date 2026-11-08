"""Reproduction check: compare the regenerated results/ with the versions committed in git.

Run after `make all` (or after `make reproduce`, which clears everything first). For every committed
results/*.json and results/*.md:
  - json: values compared with a relative tolerance (default 1e-6). Simulation files whose rows arrive in
    parallel order (net_sim*.json) are compared on their summary and on their rows sorted by key.
    Optimiser outputs (partition_selection.json) use a looser tolerance (1e-3): L-BFGS-B stops at
    slightly different points across runs.
  - md: text must be identical.
Prints one line per file and exits with status 1 if anything differs beyond tolerance.
Usage: python -I scripts/check_reproduction.py [git ref, default HEAD]
"""
import json
import math
import subprocess
import sys

from hub8002 import ROOT

LOOSE = {'partition_selection.json': 1e-3}
UNORDERED_ROWS = {'net_sim.json', 'net_sim_bench50.json'}
SKIP_KEYS = {'n_perm'}


def committed(ref, path):
    out = subprocess.run(['git', 'show', f'{ref}:{path}'], cwd=ROOT, capture_output=True)
    return out.stdout if out.returncode == 0 else None


def diff(x, y, tol, path='', out=None):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in set(x) | set(y):
            if k in SKIP_KEYS:
                continue
            if k not in x or k not in y:
                out.append(f'{path}/{k}: missing')
            else:
                diff(x[k], y[k], tol, f'{path}/{k}', out)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append(f'{path}: length {len(x)} vs {len(y)}')
        else:
            for i, (a, b) in enumerate(zip(x, y)):
                diff(a, b, tol, f'{path}[{i}]', out)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        if not (x == y or (x is not None and y is not None and math.isclose(x, y, rel_tol=tol, abs_tol=tol * 1e-3))):
            out.append(f'{path}: {x} vs {y}')
    elif x != y:
        out.append(f'{path}: {str(x)[:40]} vs {str(y)[:40]}')
    return out


def row_key(r):
    return json.dumps({k: v for k, v in r.items() if k in ('team', 'rep', 'world', 'm')}, sort_keys=True)


def main(ref='HEAD'):
    files = subprocess.run(['git', 'ls-tree', '-r', '--name-only', ref, 'results/'], cwd=ROOT,
                           capture_output=True, text=True).stdout.split()
    bad = 0
    for f in files:
        name = f.split('/')[-1]
        if name.startswith('SUMMARY'):
            continue                      # hand-written
        new_path = ROOT / f
        old = committed(ref, f)
        if not new_path.exists():
            print(f'{name:30s} MISSING (not regenerated)')
            bad += 1
            continue
        if name.endswith('.md'):
            same = old.decode() == new_path.read_text()
            print(f"{name:30s} {'identical' if same else 'DIFFERENT'}")
            bad += not same
            continue
        a, b = json.loads(old), json.load(open(new_path))
        if name in UNORDERED_ROWS:
            a['rows'] = sorted(a['rows'], key=row_key)
            b['rows'] = sorted(b['rows'], key=row_key)
        out = diff(a, b, LOOSE.get(name, 1e-6), '', [])
        print(f"{name:30s} {'identical (within tolerance)' if not out else f'{len(out)} DIFFERENCES, e.g. ' + out[0]}")
        bad += bool(out)
    print(f'\n{len(files)} committed result files checked; {bad} differ.')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main(*sys.argv[1:2])
