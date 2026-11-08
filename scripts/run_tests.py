"""Minimal test runner (used when pytest is unavailable): runs every test_* function in tests/."""
import importlib.util
import sys
import time
import traceback
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'tests'
fails = 0
for f in sorted(root.glob('test_*.py')):
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in sorted(n for n in dir(mod) if n.startswith('test_')):
        t = time.time()
        try:
            getattr(mod, name)()
            print(f'PASS {f.stem}::{name} ({time.time() - t:.1f}s)')
        except Exception:
            fails += 1
            print(f'FAIL {f.stem}::{name}')
            traceback.print_exc()
print('failures:', fails)
sys.exit(1 if fails else 0)
