"""E2: how many matches are needed to confirm a hub? (simulation, both layers)

Same generator as E1 with measured phi and measured individual differences; exposure = the real
minutes of a 2015/16 squad over its first m matches (m = 3, 5, 8, 12, 19, 38). Candidates: outfield
players with >= 90 minutes within those m matches. Output: detection / false-positive rates and the
"matches needed" table (smallest m with detection >= 80% and false positives <= 5%).
Usage: python -I -m hub8002.experiments.e2_matches_needed [reps] [workers]
"""
import sys
import time
from itertools import product
from multiprocessing import Pool

import numpy as np

from ..sim.generator import simulate_team
from ..sim.squads import squad_structures
from .common import CFG, LAYERS, constants, dump, run_methods

M_VALUES = (3, 5, 8, 12, 19, 38)
METHODS = ('bayes_nb', 'bayes_p', 'mdl_cm_min')
SQUADS = None


def _init():
    global SQUADS
    SQUADS = squad_structures()


RHOS = (1.25, 1.5, 2.0, 3.0)


def conditions():
    """het = 'measured' (main) or 'none' (players of a group identical: the plan's original assumption)."""
    hubs = [('none', None, 1.0)] + [(g, g, r) for g in ('MID', 'FWD') for r in RHOS]
    return [dict(het=het, layer=l, m=m, hub=h[0], hub_group=h[1], rho=h[2])
            for het, l, m, h in product(('measured', 'none'), LAYERS, M_VALUES, hubs)]


def run_chunk(args):
    ci, cond, reps, seed = args
    rates, phi_m, alpha_m = constants()
    layer = cond['layer']
    rng = np.random.default_rng(seed)
    teams = sorted(SQUADS)
    tally = {m: {'any': 0, 'detect': 0, 'exact': 0} for m in METHODS}
    for r in range(reps):
        sq = SQUADS[teams[(seed + r) % len(teams)]]
        M = sq['minutes'][:cond['m']]
        T = M.sum(axis=0) / 90.0
        keep = T * 90 >= CFG['min_minutes']
        T, group = T[keep], sq['group'][keep]
        hubs = []
        if cond['hub_group']:
            pool = [i for i in range(len(T)) if group[i] == cond['hub_group']
                    and T[i] >= np.median(T[group == cond['hub_group']])]
            hubs = [int(rng.choice(pool))]
        k, _ = simulate_team(T, group, rates[layer], hubs, cond['rho'], phi_m[layer], rng,
                             alpha=alpha_m[layer] if cond['het'] == 'measured' else None)
        res, _ = run_methods(METHODS, k, T, group, layer)
        for m, h in res.items():
            tally[m]['any'] += len(h) > 0
            if hubs:
                tally[m]['detect'] += hubs[0] in h
                tally[m]['exact'] += h == hubs
    return ci, reps, tally


def main(reps=300, workers=12, chunk=50):
    conds = conditions()
    chunk = min(chunk, reps)
    tasks = [(ci, c, chunk, CFG['seed'] * 7000 + ci * 100 + j) for ci, c in enumerate(conds) for j in range(reps // chunk)]
    agg = {ci: [0, {m: {'any': 0, 'detect': 0, 'exact': 0} for m in METHODS}] for ci in range(len(conds))}
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        for ci, n, tally in pool.imap_unordered(run_chunk, tasks):
            agg[ci][0] += n
            for m in METHODS:
                for key in tally[m]:
                    agg[ci][1][m][key] += tally[m][key]
    rows = []
    for ci, c in enumerate(conds):
        n, t = agg[ci]
        for m in METHODS:
            rows.append({**c, 'method': m, 'n': n, 'any_rate': t[m]['any'] / n,
                         'detect_rate': t[m]['detect'] / n if c['hub_group'] else None,
                         'exact_rate': t[m]['exact'] / n if c['hub_group'] else None})
    # matches-needed table
    table = {}
    for het in ('measured', 'none'):
        for m in METHODS:
            for layer, g, rho in product(LAYERS, ('MID', 'FWD'), RHOS):
                need = None
                for mv in M_VALUES:
                    sel = [r for r in rows if r['het'] == het and r['method'] == m and r['layer'] == layer and r['m'] == mv]
                    fp = next(r['any_rate'] for r in sel if r['hub'] == 'none')
                    det = next(r['detect_rate'] for r in sel if r['hub'] == g and r['rho'] == rho)
                    if det >= 0.8 and fp <= 0.05:
                        need = mv
                        break
                table[f'{het}|{m}|{layer}|{g}|{rho}'] = need
    dump({'reps': reps, 'm_values': M_VALUES, 'rows': rows, 'matches_needed': table}, 'e2_matches_needed.json')
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
