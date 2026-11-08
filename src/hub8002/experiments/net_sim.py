"""Simulation check of the network VOA (linking core) estimator.

Worlds are built from the 20 real 2015/16 squads: real players, positions, chain starts and receiving
patterns. Each player's true row (whom he passes to, shooting, losing the ball) is drawn around the
average player of his position, Dirichlet(alpha_g * pbar_i) with the calibrated alpha; true VOA follows
exactly from the true rows. For m matches we simulate chains match by match; per-match rows vary around
the true rows (Dirichlet with concentration c) so that the data are over-dispersed like the real ones
(base world c = 17, measured phi 2.1 as in the real data; 'overdispersed' world c = 6, phi 4.0, while the
model still assumes the calibrated phi 2.16). True VOA is computed for the chains actually started in
the m matches. Methods: ours (posterior), plug-in VOA (raw rows), most touches, best retention (lowest
chain-end rate among players with >= 30% of the top touches), PageRank of the pass network.
Baseline 'bench50' (2026-11-08): truth and estimates compare with the replacement level (the league's
bench players of the position, results/bench_rates.json); writes results/net_sim_bench50.json.
Writes results/net_sim.json.  Usage: python -I -m hub8002.experiments.net_sim [reps] [workers] [average|bench50]
"""
import json
import sys
import zlib
from multiprocessing import Pool

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from ..models.net_model import NetUnit, chain_values, net_cores, replacement_rows
from .common import RESULTS, dump
from .net_common import calib, league_teams, unit_for

MS = (5, 10, 19, 38)
WORLDS = {'base': 17.0, 'overdispersed': 6.0}
SEASON = 38


def simulate(true_rows, N0, g, m, c, rng, a=0.6):
    """Counts (C, S, X, E, N0) from m matches of chains; each shot's xG ~ Gamma(a, a / g)."""
    n = len(N0)
    C, S, E, starts = np.zeros((n, n)), np.zeros(n), np.zeros(n), np.zeros(n)
    for _ in range(m):
        G = rng.gamma(c * true_rows)
        rows = G / G.sum(1, keepdims=True)
        cum = np.cumsum(rows, axis=1)
        k = rng.poisson(N0 / SEASON)
        starts += k
        st = np.repeat(np.arange(n), k)
        while len(st):
            out = (cum[st] < rng.random(len(st))[:, None]).sum(1)
            out = np.minimum(out, n + 1)
            pas = out < n
            np.add.at(C, (st[pas], out[pas]), 1)
            np.add.at(S, st[out == n], 1)
            np.add.at(E, st[out == n + 1], 1)
            st = out[pas]
    X = np.where(S > 0, rng.gamma(np.maximum(a * S, 1e-12), g / a), 0.0)
    return C, S, X, E, starts


def pagerank_top(C, cand):
    G = nx.DiGraph()
    n = len(C)
    G.add_nodes_from(range(n))
    for i, j in zip(*np.nonzero(C)):
        G.add_edge(int(i), int(j), weight=float(C[i, j]))
    pr = nx.pagerank(G, weight='weight')
    ci = np.flatnonzero(cand)
    return int(ci[np.argmax([pr[i] for i in ci])])


def task(args):
    team, rep, n_draws, baseline = args
    rng = np.random.default_rng([8002, rep, zlib.crc32(team.encode())])
    u0, cand, _, _ = unit_for(team, '2015/16')
    q, gbar, alpha, phi, a = calib()
    qb = q if baseline == 'average' else json.load(open(RESULTS / 'bench_rates.json'))['net'][baseline]
    n = len(u0.ids)
    pbar = replacement_rows(u0, q)
    gbar_i = np.array([gbar[x] for x in u0.group])
    g = (u0.X + 5.0 * gbar_i) / (u0.S + 5.0)
    al = np.array([alpha[x] for x in u0.group])
    G = rng.gamma(al[:, None] * pbar)
    true_rows = G / G.sum(1, keepdims=True)
    ci = np.flatnonzero(cand)
    out = []
    for world, c in WORLDS.items():
        for m in MS:
            C, S, X, E, N0 = simulate(true_rows, u0.N0, g, m, c, rng, a)
            tv, *_ = chain_values(true_rows[None, :, :n], true_rows[None, :, n], g, N0, qb, u0.group, gbar_i, 'net')
            tv = tv[0]
            truth = int(ci[np.argmax(tv[ci])])
            u = NetUnit(u0.ids, u0.group, C, S, X, E, N0)
            f = net_cores(u, q, gbar, alpha, cand, phi=phi, n_draws=n_draws, seed=rep, shot_shape=a, q_base=qb)
            p = net_cores(u, q, gbar, alpha, cand, phi=phi, plugin=True, shot_shape=a, q_base=qb)
            touches = u.touches
            t_c = touches[ci]
            end = np.where(touches > 0, E / np.maximum(touches, 1), 1.0)
            regular = ci[t_c >= 0.3 * t_c.max()]
            # 90% credible set before the P(VOA > 0) filter
            order = np.argsort(-f.p_top)
            k = int(np.searchsorted(np.cumsum(f.p_top[order]), 0.9) + 1)
            cred = set(order[:k].tolist())
            out.append({
                'team': team, 'rep': rep, 'world': world, 'm': m, 'truth': truth,
                'picks': {'ours': f.top, 'plugin': p.top, 'touches': int(ci[np.argmax(t_c)]),
                          'retention': int(regular[np.argmin(end[regular])]), 'pagerank': pagerank_top(C, cand)},
                'p_top_pick': float(f.p_top[f.top]), 'p_top_truth': float(f.p_top[truth]),
                'p_top_all': f.p_top[ci].round(4).tolist(), 'is_truth': (ci == truth).tolist(),
                'core_set': [int(i) for i in f.core_set], 'cred_cover': truth in cred, 'cred_size': len(cred),
                'spearman_ours': float(spearmanr(f.voa_mean[ci], tv[ci])[0]),
                'spearman_plugin': float(spearmanr(p.voa_mean[ci], tv[ci])[0]),
                'interval_cover': float(np.mean((f.voa_lo[ci] <= tv[ci]) & (tv[ci] <= f.voa_hi[ci])))})
    return out


def summarise(rows):
    s = {}
    for world in WORLDS:
        for m in MS:
            r = [x for x in rows if x['world'] == world and x['m'] == m]
            key = f'{world}|{m}'
            s[key] = {'n': len(r),
                      'hit': {k: float(np.mean([x['picks'][k] == x['truth'] for x in r]))
                              for k in ('ours', 'plugin', 'touches', 'retention', 'pagerank')},
                      'mean_p_top_pick': float(np.mean([x['p_top_pick'] for x in r])),
                      'cred_cover': float(np.mean([x['cred_cover'] for x in r])),
                      'cred_size': float(np.mean([x['cred_size'] for x in r])),
                      'clear_share': float(np.mean([len(x['core_set']) == 1 for x in r])),
                      'clear_correct': float(np.mean([x['core_set'][0] == x['truth'] for x in r if len(x['core_set']) == 1]))
                      if any(len(x['core_set']) == 1 for x in r) else None,
                      'core_set_size': float(np.mean([len(x['core_set']) for x in r if x['core_set']] or [0])),
                      'no_clear_core': float(np.mean([not x['core_set'] for x in r])),
                      'spearman_ours': float(np.nanmean([x['spearman_ours'] for x in r])),
                      'spearman_plugin': float(np.nanmean([x['spearman_plugin'] for x in r])),
                      'interval_cover_90': float(np.mean([x['interval_cover'] for x in r]))}
            # reliability of P(#1) over all candidates
            p = np.concatenate([x['p_top_all'] for x in r])
            t = np.concatenate([x['is_truth'] for x in r])
            bins = np.array([0, .05, .2, .4, .6, .8, .95, 1.0001])
            idx = np.digitize(p, bins) - 1
            s[key]['reliability'] = [{'bin': f'{bins[b]:.2f}-{min(bins[b + 1], 1):.2f}', 'n': int((idx == b).sum()),
                                      'mean_p': float(p[idx == b].mean()), 'freq': float(t[idx == b].mean())}
                                     for b in range(len(bins) - 1) if (idx == b).any()]
    return s


def main(reps=100, workers=12, n_draws=1000, baseline='average'):
    tasks = [(t, r, n_draws, baseline) for t in league_teams() for r in range(reps)]
    rows = []
    with Pool(workers) as pool:
        for out in pool.imap_unordered(task, tasks, chunksize=2):
            rows.extend(out)
    s = summarise(rows)
    dump({'summary': s, 'baseline': baseline,
          'rows': [{k: v for k, v in x.items() if k not in ('p_top_all', 'is_truth')} for x in rows]},
         'net_sim.json' if baseline == 'average' else f'net_sim_{baseline}.json')
    for k, v in s.items():
        print(k, {m: round(h, 2) for m, h in v['hit'].items()}, 'p_pick', round(v['mean_p_top_pick'], 2),
              'cover', round(v['cred_cover'], 2), 'clear', round(v['clear_share'], 2), '->', v['clear_correct'] and round(v['clear_correct'], 2),
              'rho', round(v['spearman_ours'], 2), '/', round(v['spearman_plugin'], 2), 'int90', round(v['interval_cover_90'], 2))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[0]) if a else 100, int(a[1]) if len(a) > 1 else 12, baseline=a[2] if len(a) > 2 else 'average')
