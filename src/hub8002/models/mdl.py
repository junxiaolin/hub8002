"""Kirkley's MDL hub method (PRE 109, 034310, 2024).

`official_hubs` calls hubs() from the official repository (third_party/kirkley_hubs/functions.py);
`mdl_hubs` is the pilot's dependency-free port of the weighted (multigraph) encodings. They are
checked against each other in tests/test_mdl.py.
"""
import importlib.util
from collections import Counter
from math import lgamma, log

from .. import ROOT

_OFFICIAL = None


def _official():
    global _OFFICIAL
    if _OFFICIAL is None:
        path = ROOT / 'third_party' / 'kirkley_hubs' / 'functions.py'
        spec = importlib.util.spec_from_file_location('kirkley_functions', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _OFFICIAL = mod
    return _OFFICIAL


def official_hubs(degrees, weighted=True):
    """Official hubs() on a degree list -> {'ER': [...], 'CM': [...], 'AVG': [...], 'LOUBAR': [...]}."""
    res = _official().hubs(list(degrees), len(degrees), degree_list=True, weighted=weighted)
    return {k: sorted(int(i) for i in v['hub_nodes']) for k, v in res.items()}


def logchoose(n, k):
    if k <= 0 or n <= 0:
        return 0.0
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def logmultiset(n, k):
    return logchoose(n + k - 1, k)


def mdl_hubs(degrees):
    """Weighted ER and CM encodings -> {'ER': (hub indices, compression ratio), 'CM': (...)}."""
    degrees = [int(d) for d in degrees]
    n = len(degrees)
    m = sum(degrees)
    if m == 0:
        return {'ER': ([], 1.0), 'CM': ([], 1.0)}
    order = sorted(range(n), key=lambda i: -degrees[i])
    counts = Counter(degrees)
    er0 = logmultiset(n * n, m)
    cm0 = logmultiset(n, m) + sum(logmultiset(n, k) for k in degrees)
    best = {'ER': (0, er0), 'CM': (0, cm0)}
    h = mh = 0
    combs = 0.0
    for k in sorted(counts, reverse=True):
        nk = counts[k]
        h += nk
        mh += nk * k
        combs += nk * logmultiset(n, k)
        base = log(n) + log(m) + logchoose(n, h)
        rest = logmultiset((n - h) * n, m - mh)
        dl_er = base + logmultiset(h * n, mh) + rest
        dl_cm = base + logmultiset(h, mh) + combs + rest
        if dl_er < best['ER'][1]:
            best['ER'] = (h, dl_er)
        if dl_cm < best['CM'][1]:
            best['CM'] = (h, dl_cm)
    denom = max(er0, cm0)
    return {code: (sorted(order[:hh]), dl / denom) for code, (hh, dl) in best.items()}
