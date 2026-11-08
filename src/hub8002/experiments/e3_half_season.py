"""E3: whose hubs best predict the second half of the season? (real data)

Units: the 20 teams of 2015/16 plus Barcelona seasons with >= 20 matches. Each method picks a hub
set on the first half (H1, by date). Fair comparison: whatever the set, the same common model
(position base rate x minutes x two-level multiplier, negative binomial with the calibrated
individual spread and phi) is fitted on H1 and predicts each player's H2 count from his H2 minutes:
    log p(k2 | S, k1) = log p(k1, k2 | S) - log p(k1 | S)
(H1 and H2 counts enter as separate observations sharing lambda_g and rho). Evaluated on players
who are candidates (>= 90 min) in both halves. References: no hub (S = {}), and a saturated model
where every player has his own rate (upper reference; it does not answer "is there a hub").
Metrics: gain in log predictive density over "no hub" (mean over units, paired bootstrap CI),
Jaccard agreement of H1 and H2 hub sets, share of units with a hub.
Usage: python -I -m hub8002.experiments.e3_half_season [workers]
"""
import sys
from multiprocessing import Pool

import numpy as np

from .. import load_table
from ..models.bayes_hub import log_ml_het_set, log_pred_individual
from .common import ALL_METHODS, CFG, GROUPS, LAYERS, constants, dump, run_methods, unit_data

PP = ED = None


def _init():
    global PP, ED
    PP = load_table('player_period')
    ED = load_table('edges')


def common_log_pred(k1, T1, k2, T2, group, hub_mask, layer):
    rates, phi, alpha = constants()
    kk = np.concatenate([k1, k2])
    TT = np.concatenate([T1, T2])
    gg = np.concatenate([group, group])
    hm = np.concatenate([hub_mask, hub_mask])
    a = CFG['model']
    kw = dict(a=a['prior_shape'], rho_max=a['rho_max'], phi=phi[layer])
    return (log_ml_het_set(kk, TT, gg, rates[layer], alpha[layer], hm, **kw)
            - log_ml_het_set(k1, T1, group, rates[layer], alpha[layer], hub_mask, **kw))


def run_unit(args):
    team, season = args
    rates, phi, alpha = constants()
    out = {'team': team, 'season': season, 'layers': {}}
    for layer in LAYERS:
        d1 = unit_data(PP, team, f'{season} H1', layer, ED)
        d2 = unit_data(PP, team, f'{season} H2', layer, ED)
        if len(d1[0]) < 3 or len(d2[0]) < 3:
            continue
        k1, T1, g1, n1, id1, pr1 = d1
        k2a, T2a, g2a, n2a, id2a, pr2 = d2
        res1, _ = run_methods(ALL_METHODS, k1, T1, g1, layer, pr=pr1)
        res2, _ = run_methods(ALL_METHODS, k2a, T2a, g2a, layer, pr=pr2)
        # evaluation players: candidates in both halves (groups from the season, identical in both halves)
        pos2 = {p: i for i, p in enumerate(id2a)}
        both = [i for i, p in enumerate(id1) if p in pos2]
        j2 = [pos2[id1[i]] for i in both]
        k1e, T1e, ge = k1[both], T1[both], g1[both]
        k2e, T2e = k2a[j2], T2a[j2]
        base = common_log_pred(k1e, T1e, k2e, T2e, ge, np.zeros(len(both), bool), layer)
        sat = log_pred_individual(k1e, T1e, k2e, T2e, ge, rates[layer], alpha[layer], phi=phi[layer])
        L = {'n_eval': len(both), 'saturated_gain': sat - base, 'methods': {}}
        for m in ALL_METHODS:
            set1 = {id1[i] for i in res1[m]}
            set2 = {id2a[i] for i in res2[m]}
            mask = np.array([id1[i] in set1 for i in both])
            gain = 0.0 if not mask.any() else common_log_pred(k1e, T1e, k2e, T2e, ge, mask, layer) - base
            union = set1 | set2
            L['methods'][m] = {'gain': gain, 'h1': [n1[i] for i in res1[m]], 'h2': [n2a[i] for i in res2[m]],
                               'jaccard': len(set1 & set2) / len(union) if union else None,
                               'any_h1': bool(set1)}
        out['layers'][layer] = L
    return out


def main(workers=12):
    matches = load_table('matches')
    units = [(t, CFG['league_season']) for t in sorted(set(matches[matches.season == CFG['league_season']].home))]
    barca = matches[(matches.home == CFG['focus_team']) | (matches.away == CFG['focus_team'])]
    n_by = barca.groupby('season').size()
    units += [(CFG['focus_team'], s) for s, n in n_by.items() if n >= CFG['e3_min_matches'] and s != CFG['league_season']]
    with Pool(workers, initializer=_init) as pool:
        res = pool.map(run_unit, units)
    rng = np.random.default_rng(CFG['seed'])
    summary = {}
    for layer in LAYERS:
        rows = [r['layers'][layer] for r in res if layer in r['layers']]
        summary[layer] = {'n_units': len(rows),
                          'saturated_gain_mean': float(np.mean([r['saturated_gain'] for r in rows]))}
        for m in ALL_METHODS:
            g = np.array([r['methods'][m]['gain'] for r in rows])
            boot = [rng.choice(g, len(g)).mean() for _ in range(2000)]
            jac = [r['methods'][m]['jaccard'] for r in rows if r['methods'][m]['jaccard'] is not None]
            flagged = np.array([r['methods'][m]['any_h1'] for r in rows])
            n_named = np.array([len(r['methods'][m]['h1']) for r in rows])
            summary[layer][m] = {'gain_mean': float(g.mean()), 'ci95': [float(np.percentile(boot, 2.5)),
                                                                         float(np.percentile(boot, 97.5))],
                                 'gain_when_flagged': float(g[flagged].mean()) if flagged.any() else None,
                                 'gain_per_named_player': float(g.sum() / n_named.sum()) if n_named.sum() else None,
                                 'mean_named': float(n_named.mean()),
                                 'jaccard_mean': float(np.mean(jac)) if jac else None,
                                 'share_units_with_hub': float(np.mean([r['methods'][m]['any_h1'] for r in rows]))}
    dump({'units': res, 'summary': summary}, 'e3_half_season.json')
    for layer in LAYERS:
        print(layer, 'saturated', round(summary[layer]['saturated_gain_mean'], 1))
        for m in ALL_METHODS:
            s = summary[layer][m]
            print(f"  {m:18s} gain {s['gain_mean']:8.2f} CI {s['ci95'][0]:7.2f} {s['ci95'][1]:7.2f}  "
                  f"jaccard {s['jaccard_mean'] if s['jaccard_mean'] is None else round(s['jaccard_mean'], 2)}  "
                  f"share {s['share_units_with_hub']:.2f}")


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:2]))
