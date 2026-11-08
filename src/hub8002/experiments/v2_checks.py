"""v2 checks on real data.

half   (E3): first half -> second half of each season. Does the first-half estimate predict each player's
             second-half value rate? Compared: our posterior mean (shrunk towards the positional average),
             the raw first-half rate per 90, and the positional average alone. Metric: exposure-weighted mean
             squared error, as skill relative to the positional average. Also: is the first-half top core
             still the top core in the second half; overlap of core sets.
seq    (E7): cumulative fits after every match; for the season's top core, the first match from which his
             probability of being the team's #1 stays above 0.5 / 0.9.
sens        : sensitivity of the core sets to the choices of the v2 rule.
Writes results/v2_half.json, results/v2_sequential.json, results/v2_sensitivity.json.
Usage: python -I -m hub8002.experiments.v2_checks [half|seq|sens|all]
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .. import load_table
from ..aggregate import aggregate, player_groups
from ..models.core_model import eb_alpha, value_cores
from .common import CFG, GROUPS, LAYER_SPEC, dump, unit_data
from .v2_real import fit_unit, units, v2_constants

LAYERS = ('att', 'prog', 'def')


# ---------------------------------------------------------------- E3: half seasons
def half_units():
    m = load_table('matches')
    u = [(t, CFG['league_season']) for t in sorted(set(m[m.season == CFG['league_season']].home))]
    b = m[(m.home == 'Barcelona') | (m.away == 'Barcelona')].groupby('season').size()
    u += [('Barcelona', s) for s, n in b.items() if n >= CFG['e3_min_matches'] and s != CFG['league_season']]
    return u


def half():
    pp = load_table('player_period')
    rates, phi, alpha = v2_constants()
    rows, agree = [], []
    for team, season in half_units():
        for layer in LAYERS:
            d1 = unit_data(pp, team, f'{season} H1', layer)
            d2 = unit_data(pp, team, f'{season} H2', layer)
            pos2 = {p: i for i, p in enumerate(d2[4])}
            both = [i for i, p in enumerate(d1[4]) if p in pos2 and d1[1][i] > 0 and d2[1][pos2[p]] > 0]
            if len(both) < 3:
                continue
            j2 = [pos2[d1[4][i]] for i in both]
            k1, T1, g = d1[0][both], d1[1][both], d1[2][both]
            k2, T2 = d2[0][j2], d2[1][j2]
            f1 = value_cores(k1, T1, g, rates[layer], alpha[layer], phi=phi[layer])
            lam = np.array([rates[layer][x] for x in g])
            y = k2 / T2
            preds = {'ours': f1.rate_mean, 'raw_per90': k1 / T1, 'position_average': lam}
            mse = {k: float(np.sum(T2 * (p - y) ** 2) / T2.sum()) for k, p in preds.items()}
            rows.append({'team': team, 'season': season, 'layer': layer, 'n': len(both), 'mse': mse})
            # core agreement between halves (each half fitted on all its own candidates)
            c1 = value_cores(*d1[:3], rates[layer], alpha[layer], phi=phi[layer])
            c2 = value_cores(*d2[:3], rates[layer], alpha[layer], phi=phi[layer])
            s1 = {d1[4][i] for i in c1.core_set}
            s2 = {d2[4][i] for i in c2.core_set}
            agree.append({'team': team, 'season': season, 'layer': layer,
                          'top_same': d1[4][c1.top] == d2[4][c2.top],
                          'jaccard': len(s1 & s2) / len(s1 | s2) if s1 | s2 else None,
                          'h1': [d1[3][i] for i in c1.core_set], 'h2': [d2[3][i] for i in c2.core_set]})
    summary = {}
    for layer in LAYERS:
        r = [x for x in rows if x['layer'] == layer]
        tot = {k: np.mean([x['mse'][k] for x in r]) for k in ('ours', 'raw_per90', 'position_average')}
        wins = np.mean([x['mse']['ours'] <= x['mse']['raw_per90'] for x in r])
        a = [x for x in agree if x['layer'] == layer]
        jac = [x['jaccard'] for x in a if x['jaccard'] is not None]
        summary[layer] = {'n_units': len(r),
                          'skill_vs_position_average': {k: float(1 - tot[k] / tot['position_average']) for k in tot},
                          'share_units_ours_beats_raw': float(wins),
                          'top_core_same_both_halves': float(np.mean([x['top_same'] for x in a])),
                          'core_set_jaccard': float(np.mean(jac)) if jac else None}
    dump({'rows': rows, 'agreement': agree, 'summary': summary}, 'v2_half.json')
    for layer, s in summary.items():
        print(layer, s)


# ---------------------------------------------------------------- E7: sequential
TAB = {}


def _init_seq():
    for n in ('matches', 'appearances', 'passes', 'match_edges', 'player_match_values'):
        TAB[n] = load_table(n)


def seq_unit(args):
    team, season = args
    rates, phi, alpha = v2_constants()
    m = TAB['matches']
    mids = list(m[(m.season == season) & ((m.home == team) | (m.away == team))].sort_values(['date', 'match_id']).match_id)
    gmap = player_groups(TAB['appearances'], mids, team)
    traj = {l: {} for l in LAYERS}
    names = {}
    for n in range(1, len(mids) + 1):
        pp, _ = aggregate(TAB['appearances'], TAB['passes'], TAB['match_edges'], mids[:n], team, group_map=gmap,
                          values=TAB['player_match_values'])
        for layer in LAYERS:
            kc, tc = LAYER_SPEC[layer]
            c = pp[pp.group.isin(GROUPS) & (pp.minutes >= CFG['min_minutes']) & (pp[tc] > 0)]
            if len(c) < 2:
                continue
            names.update(dict(zip(c.player_id, c.name)))
            f = value_cores(c[kc].to_numpy(float), c[tc].to_numpy(float), c.group.to_numpy(), rates[layer],
                            alpha[layer], phi=phi[layer], n_draws=2000)
            for pid, p in zip(c.player_id, f.p_top):
                traj[layer].setdefault(int(pid), {})[n] = float(p)
    out = {'team': team, 'season': season, 'n_matches': len(mids), 'names': {int(k): v for k, v in names.items()},
           'layers': {}}
    nmax = len(mids)
    for layer in LAYERS:
        if not traj[layer]:
            continue
        final = {pid: s.get(nmax, 0.0) for pid, s in traj[layer].items()}
        top = max(final, key=final.get)
        conf = {}
        for thr in (0.5, 0.9):
            conf[str(thr)] = next((n for n in range(1, nmax + 1)
                                   if all(traj[layer][top].get(x, 0.0) > thr for x in range(n, nmax + 1))), None)
        out['layers'][layer] = {'top': top, 'top_name': names.get(top), 'final_p': final[top], 'confirm': conf,
                                'trajectories': {str(k): v for k, v in traj[layer].items()}}
    return out


def seq():
    m = load_table('matches')
    u = [('Barcelona', s) for s in CFG['seasons'].values()]
    u += [(t, CFG['league_season']) for t in sorted(set(m[m.season == CFG['league_season']].home)) if t != 'Barcelona']
    with Pool(12, initializer=_init_seq) as pool:
        res = pool.map(seq_unit, u)
    summary = {}
    for layer in LAYERS:
        c5 = [r['layers'][layer]['confirm']['0.5'] / r['n_matches'] for r in res
              if layer in r['layers'] and r['layers'][layer]['confirm']['0.5']]
        c9 = [r['layers'][layer]['confirm']['0.9'] / r['n_matches'] for r in res
              if layer in r['layers'] and r['layers'][layer]['confirm']['0.9']]
        summary[layer] = {'n_units': len(res), 'n_confirm_0.5': len(c5), 'median_share_0.5': float(np.median(c5)) if c5 else None,
                          'n_confirm_0.9': len(c9), 'median_share_0.9': float(np.median(c9)) if c9 else None}
    dump({'units': res, 'summary': summary}, 'v2_sequential.json')
    print(summary)


# ---------------------------------------------------------------- sensitivity
ROLE_TO = {'CB': 'CB', 'FB': 'FB', 'CM': 'CM', 'WM': 'WM', 'W': 'W', 'CF': 'CF', 'GK': 'GK'}


def constants_for(pp_mod, groups):
    """League rates and EB alpha per group for a modified player_period (other exposure or grouping)."""
    _, phi, _ = v2_constants()
    full = pp_mod[(pp_mod.season == CFG['league_season']) & (pp_mod.half == 'full') & pp_mod.group.isin(groups)
                  & (pp_mod.minutes >= CFG['min_minutes'])]
    rates, alpha = {}, {}
    for layer in LAYERS + ('org',):
        kc, tc = LAYER_SPEC[layer]
        f = full[full[tc] > 0]
        rates[layer] = {g: float(d[kc].sum() / d[tc].sum()) for g, d in f.groupby('group')}
        alpha[layer] = {g: eb_alpha(d[kc], d[tc], rates[layer][g], phi[layer]) for g, d in f.groupby('group')}
    return rates, phi, alpha


def six_role_tables(pp, apps, matches):
    """player_period copy with six fine roles as `group`, plus league rates and alpha for those roles."""
    rows = []
    for (team, label), d in pp.groupby(['team', 'label']):
        season = d.season.iloc[0]
        seasons = CFG['eras'].get(season, [season])
        mids = matches[matches.season.isin(seasons) & ((matches.home == team) | (matches.away == team))].match_id
        a = apps[(apps.team == team) & apps.match_id.isin(mids)]
        cols = [c for c in a.columns if c.startswith('rmin_')]
        r = a.groupby('player_id')[cols].sum()
        role = r.idxmax(axis=1).str.replace('rmin_', '', regex=False)
        dd = d.copy()
        dd['group'] = dd.player_id.map(role)
        rows.append(dd)
    p6 = pd.concat(rows, ignore_index=True)
    roles = ('CB', 'FB', 'CM', 'WM', 'W', 'CF')
    r6, _, a6 = constants_for(p6, roles)
    return p6, r6, a6


def sens():
    pp = load_table('player_period')
    ed = load_table('edges')
    barca, _, league = units()
    all_units = barca + [u for u in league if u[0] != 'Barcelona']
    variants = [('base', {}), ('alpha x0.5', {'alpha_mult': 0.5}), ('alpha x2', {'alpha_mult': 2.0}),
                ('core set 80%', {'cred': 0.8}), ('core set 95%', {'cred': 0.95}),
                ('P(VOA>0) >= 0.8', {'pos_prob': 0.8}), ('P(VOA>0) >= 0.95', {'pos_prob': 0.95}),
                ('min minutes 450', {'min_minutes': 450}), ('baseline = team-mates', {'baseline': 'team'})]
    res = {name: {f'{t}|{l}': fit_unit(pp, ed, t, l, opt) for t, l in all_units} for name, opt in variants}
    # exposure = minutes played instead of possession minutes
    pp_min = pp.copy()
    pp_min['T_own'] = pp_min['T']
    pp_min['T_opp'] = pp_min['T']
    cmin = constants_for(pp_min, GROUPS)
    res['exposure = minutes'] = {f'{t}|{l}': fit_unit(pp_min, ed, t, l, {'consts': cmin}) for t, l in all_units}
    # six roles
    p6, r6, a6 = six_role_tables(pp, load_table('appearances'), load_table('matches'))
    _, phi0, _ = v2_constants()
    consts = (r6, phi0, a6)
    res['six roles'] = {f'{t}|{l}': fit_unit(p6, ed, t, l, {'consts': consts, 'groups': ('CB', 'FB', 'CM', 'WM', 'W', 'CF')})
                        for t, l in all_units}
    base = res['base']
    summary = []
    for name, out in res.items():
        row = {'variant': name}
        for layer in LAYERS:
            keys = [k for k in base if layer in base[k]['layers'] and layer in out.get(k, {}).get('layers', {})]
            row[f'{layer}_top_same'] = float(np.mean([out[k]['layers'][layer]['top'] == base[k]['layers'][layer]['top'] for k in keys]))
            row[f'{layer}_set_same'] = float(np.mean([set(out[k]['layers'][layer]['core_set']) == set(base[k]['layers'][layer]['core_set']) for k in keys]))
            row[f'{layer}_clear_share'] = float(np.mean([out[k]['layers'][layer]['clear'] for k in keys]))
        summary.append(row)
    slim = {name: {k: {l: {'core_set': v['layers'][l]['core_set'], 'top': v['layers'][l]['top']}
                       for l in v['layers']} for k, v in out.items()} for name, out in res.items()}
    dump({'summary': summary, 'fits': slim, 'six_role_constants': {'rates': r6, 'alpha': a6}}, 'v2_sensitivity.json')
    for s in summary:
        print(s)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('half', 'all'):
        half()
    if what in ('seq', 'all'):
        seq()
    if what in ('sens', 'all'):
        sens()
