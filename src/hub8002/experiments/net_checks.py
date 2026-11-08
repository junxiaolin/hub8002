"""Checks of the network extension (linking core) on real data.

half : first half -> second half of each season (20 teams of 2015/16 + Barcelona seasons with >= 20
       matches). (1) Does the first-half posterior predict what each player does with the ball in the
       second half (pass to GK / DEF / MID / FWD, shoot, lose it) better than his raw first-half shares
       or the positional average? Metric: log score per touch, as gain over the positional average.
       (2) Rank correlation of first-half VOA with second-half (plug-in) VOA. (3) Same top core in both halves?
sens : stability of top core and core set under other choices (alpha, phi, set threshold, P(VOA>0)
       threshold, minimum minutes, xG-per-shot prior, how the average player shares passes, six roles).
Writes results/net_half.json and results/net_sensitivity.json.
Usage: python -I -m hub8002.experiments.net_checks [half|sens|all]
"""
import sys

import numpy as np
from scipy.stats import spearmanr

from ..models.net_model import DEST, _counts, category_counts, net_cores, replacement_rows
from .common import dump
from .net_common import calib, unit_for
from .net_real import fit
from .v2_checks import half_units
from .v2_real import units


def _agg(u, rows):
    """(n, n+2) rows over team-mates -> (n, 6) over DEST groups + SHOT, END."""
    n = len(u.ids)
    out = np.zeros((n, len(DEST) + 2))
    for h, name in enumerate(DEST):
        out[:, h] = rows[:, :n][:, u.group == name].sum(1)
    out[:, -2:] = rows[:, n:]
    return out


def half():
    q, gbar, alpha, phi, a = calib()
    rows, agree = [], []
    for team, season in half_units():
        u1, c1, n1, _ = unit_for(team, f'{season} H1')
        u2, c2, n2, _ = unit_for(team, f'{season} H2')
        pos2 = {p: k for k, p in enumerate(u2.ids)}
        t2 = u2.touches
        both = [i for i in np.flatnonzero(c1) if u1.ids[i] in pos2 and t2[pos2[u1.ids[i]]] >= 30]
        if len(both) < 3:
            continue
        j2 = np.array([pos2[u1.ids[i]] for i in both])
        y = category_counts(u2)[j2]                                   # second-half outcomes
        pbar = replacement_rows(u1, q)
        A = np.array([alpha[x] for x in u1.group])[:, None] * pbar + _counts(u1) / phi
        preds = {'ours': _agg(u1, A / A.sum(1, keepdims=True))[both],
                 'raw': None,
                 'position_average': np.array([[q[x][k] for k in DEST + ('SHOT', 'END')] for x in u1.group[both]])}
        raw = category_counts(u1)[both] + 0.5
        preds['raw'] = raw / raw.sum(1, keepdims=True)
        score = {k: float(np.sum(y * np.log(np.maximum(p, 1e-12))) / y.sum()) for k, p in preds.items()}
        # VOA ranking: first half (ours, plug-in, touches) vs second half plug-in
        f1 = net_cores(u1, q, gbar, alpha, c1, phi=phi, n_draws=1000, shot_shape=a)
        p1 = net_cores(u1, q, gbar, alpha, c1, phi=phi, plugin=True, shot_shape=a)
        p2 = net_cores(u2, q, gbar, alpha, c2, phi=phi, plugin=True, shot_shape=a)
        target = p2.voa_mean[j2]
        rho = {'ours': float(spearmanr(f1.voa_mean[both], target)[0]),
               'plugin': float(spearmanr(p1.voa_mean[both], target)[0]),
               'touches': float(spearmanr(u1.touches[both], target)[0])}
        rows.append({'team': team, 'season': season, 'n': len(both), 'log_score': score, 'spearman': rho})
        f2 = net_cores(u2, q, gbar, alpha, c2, phi=phi, n_draws=1000, shot_shape=a)
        s1 = {int(u1.ids[i]) for i in f1.core_set}
        s2 = {int(u2.ids[i]) for i in f2.core_set}
        agree.append({'team': team, 'season': season, 'top_h1': n1[f1.top], 'top_h2': n2[f2.top],
                      'top_same': int(u1.ids[f1.top]) == int(u2.ids[f2.top]),
                      'core_h1': [n1[i] for i in f1.core_set], 'core_h2': [n2[i] for i in f2.core_set],
                      'jaccard': len(s1 & s2) / len(s1 | s2) if s1 | s2 else None})
    gain = {k: float(np.mean([r['log_score'][k] - r['log_score']['position_average'] for r in rows]))
            for k in ('ours', 'raw')}
    summary = {'n_units': len(rows),
               'log_score_gain_per_touch_vs_position_average': gain,
               'share_units_ours_beats_raw': float(np.mean([r['log_score']['ours'] >= r['log_score']['raw'] for r in rows])),
               'share_units_ours_beats_position_average': float(np.mean([r['log_score']['ours'] >= r['log_score']['position_average'] for r in rows])),
               'mean_spearman': {k: float(np.nanmean([r['spearman'][k] for r in rows])) for k in ('ours', 'plugin', 'touches')},
               'top_core_same_both_halves': float(np.mean([x['top_same'] for x in agree])),
               'top_core_same_barca': float(np.mean([x['top_same'] for x in agree if x['team'] == 'Barcelona'])),
               'core_set_jaccard': float(np.mean([x['jaccard'] for x in agree if x['jaccard'] is not None] or [np.nan]))}
    dump({'rows': rows, 'agreement': agree, 'summary': summary}, 'net_half.json')
    print(summary)


VARIANTS = [('base', {}), ('alpha x0.5', {'alpha_mult': 0.5}), ('alpha x2', {'alpha_mult': 2.0}),
            ('phi = 1', {'phi_off': True}), ('core set 80%', {'cred': 0.8}), ('core set 95%', {'cred': 0.95}),
            ('P(VOA>0) >= 0.8', {'pos_prob': 0.8}), ('P(VOA>0) >= 0.95', {'pos_prob': 0.95}),
            ('min minutes 450', {'min_minutes': 450}), ('xG/shot prior 20 shots', {'kappa': 20.0}),
            ('average player shares passes equally', {'repl': 'uniform'})]


def sens():
    barca, _, league = units()
    all_units = barca + [x for x in league if x[0] != 'Barcelona']
    res = {}
    for name, opt in VARIANTS:
        res[name] = {f'{t}|{l}': fit(t, l, opts={**opt, 'n_draws': 1000}, extras=False) for t, l in all_units}
    res['six roles'] = {f'{t}|{l}': fit(t, l, roles=True, opts={'n_draws': 1000}, extras=False) for t, l in all_units}
    base = res['base']
    summary = {}
    for name, r in res.items():
        keys = list(base)
        summary[name] = {'top_same': float(np.mean([r[k]['top'] == base[k]['top'] for k in keys])),
                         'core_set_same': float(np.mean([r[k]['core_set'] == base[k]['core_set'] for k in keys])),
                         'n_with_core': int(sum(bool(r[k]['core_set']) for k in keys)),
                         'top_same_barca': float(np.mean([r[k]['top'] == base[k]['top'] for k in keys if k.startswith('Barcelona')]))}
    dump({'summary': summary, 'units': {n: {k: {'top': v['top'], 'core_set': v['core_set'], 'p_top_max': v['p_top_max']}
                                            for k, v in r.items()} for n, r in res.items()}}, 'net_sensitivity.json')
    for k, v in summary.items():
        print(f'{k:40s}', v)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('half', 'all'):
        half()
    if what in ('sens', 'all'):
        sens()
