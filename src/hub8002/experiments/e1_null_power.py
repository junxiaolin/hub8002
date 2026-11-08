"""E1: do the methods name the wrong players? (simulation, organising layer)

Each replicate copies a real 2015/16 squad (rotating over the 20 teams) and regenerates pass counts.
Factors: individual differences (none = the plan's original generator / measured alpha),
match-level dispersion (phi = 1 / measured), rotation (equal / real / real with 30% extra rest),
planted hub (none, or one regular MID or FWD with rho = 1.25, 1.5, 2).
Metrics: false-positive rate (>= 1 hub reported when none planted), detection (planted hub in the
reported set), exact (reported set == {planted hub}).
Usage: python -I -m hub8002.experiments.e1_null_power [reps] [workers]
"""
import sys
import time
from itertools import product
from multiprocessing import Pool

import numpy as np

from ..models.baselines import pagerank_scores
from ..sim.generator import rotation_minutes, simulate_team
from ..sim.squads import squad_structures
from .common import ALL_METHODS, CFG, constants, dump, run_methods

LAYER = 'org'
SQUADS = None


def _init():
    global SQUADS
    SQUADS = squad_structures()


def conditions():
    hubs = [('none', None, 1.0)] + [(f'{g}', g, r) for g in ('MID', 'FWD') for r in (1.25, 1.5, 2.0)]
    return [dict(het=h, phi=p, rotation=rot, hub=hb[0], hub_group=hb[1], rho=hb[2])
            for h, p, rot, hb in product(('none', 'measured'), ('1', 'measured'), ('equal', 'real', 'rest'), hubs)]


def run_chunk(args):
    ci, cond, reps, seed = args
    rates, phi_m, alpha_m = constants()
    rng = np.random.default_rng(seed)
    teams = sorted(SQUADS)
    phi = 1.0 if cond['phi'] == '1' else phi_m[LAYER]
    alpha_gen = None if cond['het'] == 'none' else alpha_m[LAYER]
    tally = {m: {'any': 0, 'detect': 0, 'exact': 0, 'size': 0} for m in ALL_METHODS}
    for r in range(reps):
        sq = SQUADS[teams[(seed + r) % len(teams)]]
        T = rotation_minutes(sq['minutes'], cond['rotation'], rng)
        keep = T > 0
        group = sq['group'][keep]
        T = T[keep]
        C = sq['copresence'][np.ix_(keep, keep)]
        hubs = []
        if cond['hub_group']:
            Treal = sq['minutes'].sum(axis=0)[keep] / 90.0
            pool = [i for i in range(len(T)) if group[i] == cond['hub_group']
                    and Treal[i] >= np.median(Treal[group == cond['hub_group']])]
            hubs = [int(rng.choice(pool))]
        k, E = simulate_team(T, group, rates[LAYER], hubs, cond['rho'], phi, rng, alpha=alpha_gen, copresence=C)
        pr_d = pagerank_scores(list(range(len(k))), [(i, j, E[i, j]) for i in range(len(k)) for j in range(len(k))])
        pr = [pr_d[i] for i in range(len(k))]
        res, _ = run_methods(ALL_METHODS, k, T, group, LAYER, pr=pr)
        for m, h in res.items():
            t = tally[m]
            t['any'] += len(h) > 0
            t['size'] += len(h)
            if hubs:
                t['detect'] += hubs[0] in h
                t['exact'] += h == hubs
    return ci, reps, tally


def main(reps=500, workers=12, chunk=50):
    conds = conditions()
    chunk = min(chunk, reps)
    tasks = [(ci, c, chunk, CFG['seed'] * 1000 + ci * 100 + j) for ci, c in enumerate(conds)
             for j in range(reps // chunk)]
    agg = {ci: {'n': 0, 'tally': {m: {'any': 0, 'detect': 0, 'exact': 0, 'size': 0} for m in ALL_METHODS}}
           for ci in range(len(conds))}
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        for done, (ci, n, tally) in enumerate(pool.imap_unordered(run_chunk, tasks), 1):
            agg[ci]['n'] += n
            for m in ALL_METHODS:
                for key in tally[m]:
                    agg[ci]['tally'][m][key] += tally[m][key]
            if done % 40 == 0:
                print(f'{done}/{len(tasks)} chunks, {time.time() - t0:.0f}s', flush=True)
    rows = []
    for ci, c in enumerate(conds):
        n = agg[ci]['n']
        for m, t in agg[ci]['tally'].items():
            rows.append({**c, 'method': m, 'n': n, 'any_rate': t['any'] / n, 'mean_size': t['size'] / n,
                         'detect_rate': t['detect'] / n if c['hub_group'] else None,
                         'exact_rate': t['exact'] / n if c['hub_group'] else None})
    dump({'layer': LAYER, 'reps': reps, 'rows': rows}, 'e1_null_power.json')
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
