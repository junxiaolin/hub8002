"""v2 simulation (E1 + E2 for value cores): does the rule name the truly most valuable player, and are
its probabilities honest?

Each replicate copies a real 2015/16 squad (rotating over the 20 teams): real exposure over the first
m matches (m = 5, 10, 19, 38) in the relevant possession phase, position groups, league rates. True
rates r_i = lam_g * u_i with u_i drawn from the calibrated spread: 'gamma' world (as the model assumes)
or 'lognormal' world (same mean and CV, heavier right tail: a misspecified check). Values are drawn as
quasi-Poisson. The truth is the player with the largest true value over an average player of his
position, VOA_i = T_i (r_i - lam_g).
Methods (each names one top player): ours (largest posterior mean VOA); raw VOA (no shrinkage);
most total value (no minutes, no position); most per 90 (no position, no shrinkage); largest ratio to
the position average (v1-style). For ours also: calibration of p_top, coverage and size of the 90%
core set, and how often a "clear core" (single name) is right.
Baseline 'bench50' (2026-11-08): the truth and every value-based method compare with the replacement
level (the league's bench players of the position, results/bench_rates.json) instead of the average;
writes results/v2_sim_bench50.json.
Usage: python -I -m hub8002.experiments.v2_sim [reps] [workers] [average|bench50]
"""
import json
import sys
import time
from itertools import product
from multiprocessing import Pool

import numpy as np

from ..models.core_model import value_cores
from ..sim.generator import EXPOSURE_KEY
from ..sim.squads import squad_structures
from .common import CFG, RESULTS, constants, dump

LAYERS = ('att', 'prog', 'def')
M_VALUES = (5, 10, 19, 38)
WORLDS = ('gamma', 'lognormal')
METHODS = ('ours', 'raw_voa', 'total', 'per90', 'ratio')
SQUADS = None


def _init():
    global SQUADS
    SQUADS = squad_structures()


def draw_u(rng, al, world):
    if world == 'gamma':
        return rng.gamma(al, 1.0 / al)
    s2 = np.log1p(1.0 / al)                      # lognormal with mean 1 and variance 1/alpha
    return rng.lognormal(-s2 / 2, np.sqrt(s2))


def run_chunk(args):
    layer, m, world, reps, seed, baseline = args
    rates, phi, _ = constants()
    alpha = json.load(open(RESULTS / 'alpha_voa.json'))[layer]
    vbase = None if baseline == 'average' else json.load(open(RESULTS / 'bench_rates.json'))['node'][baseline][layer]
    rng = np.random.default_rng(seed)
    teams = sorted(SQUADS)
    hit = {k: 0 for k in METHODS}
    n = 0
    cal = []          # (p_top of our pick, correct)
    cover = size = clear = clear_ok = 0
    for r in range(reps):
        sq = SQUADS[teams[(seed + r) % len(teams)]]
        mins = sq['minutes'][:m].sum(axis=0)
        T = sq[EXPOSURE_KEY[layer]][:m].sum(axis=0) / 90.0
        keep = (mins >= CFG['min_minutes']) & (T > 0)
        if keep.sum() < 3:
            continue
        T, g = T[keep], sq['group'][keep]
        lam = np.array([rates[layer][x] for x in g])
        lamb = lam if vbase is None else np.array([vbase[x] for x in g])   # whom we compare with
        al = np.array([alpha[x] for x in g])
        rt = lam * draw_u(rng, al, world)
        k = phi[layer] * rng.poisson(T * rt / phi[layer])
        truth = int(np.argmax(T * (rt - lamb)))
        f = value_cores(k, T, g, rates[layer], alpha, phi=phi[layer], pos_prob=0.0, max_set=len(k),
                        n_draws=2000, seed=int(rng.integers(1 << 30)), voa_base=vbase)
        picks = {'ours': f.top, 'raw_voa': int(np.argmax(k - T * lamb)), 'total': int(np.argmax(k)),
                 'per90': int(np.argmax(k / T)), 'ratio': int(np.argmax(k / (T * lam)))}
        for key, p in picks.items():
            hit[key] += p == truth
        cal.append((float(f.p_top[f.top]), f.top == truth))
        cover += truth in f.core_set
        size += len(f.core_set)
        if len(f.core_set) == 1:
            clear += 1
            clear_ok += f.core_set[0] == truth
        n += 1
    return (layer, m, world), n, hit, cal, cover, size, clear, clear_ok


def main(reps=400, workers=20, chunk=50, baseline='average'):
    tasks = [(l, m, w, chunk, CFG['seed'] * 31 + i * 1000 + j, baseline)
             for i, (l, m, w) in enumerate(product(LAYERS, M_VALUES, WORLDS)) for j in range(reps // chunk)]
    agg = {}
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        for key, n, hit, cal, cover, size, clear, clear_ok in pool.imap_unordered(run_chunk, tasks):
            a = agg.setdefault(key, {'n': 0, 'hit': {k: 0 for k in METHODS}, 'cal': [], 'cover': 0, 'size': 0,
                                     'clear': 0, 'clear_ok': 0})
            a['n'] += n
            for k in METHODS:
                a['hit'][k] += hit[k]
            a['cal'] += cal
            a['cover'] += cover
            a['size'] += size
            a['clear'] += clear
            a['clear_ok'] += clear_ok
    rows = []
    for (layer, m, world), a in sorted(agg.items()):
        cal = np.array(a['cal'])
        bins = np.digitize(cal[:, 0], [0.5, 0.7, 0.9])
        calib = [{'bin': ['<0.5', '0.5-0.7', '0.7-0.9', '>=0.9'][b], 'n': int((bins == b).sum()),
                  'mean_p': float(cal[bins == b, 0].mean()) if (bins == b).any() else None,
                  'accuracy': float(cal[bins == b, 1].mean()) if (bins == b).any() else None} for b in range(4)]
        rows.append({'layer': layer, 'm': m, 'world': world, 'n': a['n'],
                     'accuracy': {k: a['hit'][k] / a['n'] for k in METHODS},
                     'mean_p_top': float(cal[:, 0].mean()), 'coverage90': a['cover'] / a['n'],
                     'mean_set_size': a['size'] / a['n'], 'clear_rate': a['clear'] / a['n'],
                     'clear_precision': a['clear_ok'] / a['clear'] if a['clear'] else None, 'calibration': calib})
    dump({'reps': reps, 'baseline': baseline, 'rows': rows},
         'v2_sim.json' if baseline == 'average' else f'v2_sim_{baseline}.json')
    print(f'done in {time.time() - t0:.0f}s')
    for r in rows:
        acc = ' '.join(f"{k}={v:.2f}" for k, v in r['accuracy'].items())
        print(f"{r['layer']} m={r['m']:2d} {r['world']:9s} {acc} | p_top {r['mean_p_top']:.2f} cover {r['coverage90']:.2f} "
              f"size {r['mean_set_size']:.2f} clear {r['clear_rate']:.2f} prec {r['clear_precision']}")


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[0]) if a else 400, int(a[1]) if len(a) > 1 else 20, baseline=a[2] if len(a) > 2 else 'average')
