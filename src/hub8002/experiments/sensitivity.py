"""Sensitivity analysis: refit our model on E4 (Barcelona) and E5 (La Liga 2015/16) under the
alternatives of the 12 decisions, and report whether the conclusions change.

Variants: rho_max 2 / 10; prior shape 0.5 / 2; prior mean x0.5 / x2; min minutes 450; s_max 4;
no phi correction; alpha x0.5 / x2; grouping "wingers in midfield"; six finer roles
(CB / FB / central MID / wide MID / winger / CF). For the two grouping variants the league rates and
alpha are re-estimated for the new groups (same empirical-Bayes procedure as scripts/calibrate_alpha.py).
Writes results/sensitivity.json.
Usage: python -I -m hub8002.experiments.sensitivity [workers]
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .. import load_table
from ..models.bayes_hub import fit_hubs_het
from .common import CFG, LAYER_SPEC, LAYERS, constants, dump

ROLE_TO = {
    'wingers_mid': {'CB': 'DEF', 'FB': 'DEF', 'CM': 'MID', 'WM': 'MID', 'W': 'MID', 'CF': 'FWD', 'GK': 'GK'},
    'roles6': {'CB': 'CB', 'FB': 'FB', 'CM': 'CM', 'WM': 'WM', 'W': 'W', 'CF': 'CF', 'GK': 'GK'},
}
ALPHA_GRID = np.exp(np.linspace(np.log(0.3), np.log(3000), 41))
TAB = {}


def _init():
    for n in ('player_period', 'appearances', 'matches'):
        TAB[n] = load_table(n)


def units():
    m = load_table('matches')
    era_seasons = [s for v in CFG['eras'].values() for s in v]
    u = [('Barcelona', e, list(v)) for e, v in CFG['eras'].items()]
    u += [('Barcelona', s, [s]) for s in CFG['seasons'].values() if s not in era_seasons]
    u += [(t, CFG['league_season'], [CFG['league_season']]) for t in sorted(set(m[m.season == CFG['league_season']].home))
          if t != 'Barcelona']
    return u


def scheme_groups(team, seasons, scheme):
    """player_id -> group under an alternative scheme (most minutes over the period)."""
    m, a = TAB['matches'], TAB['appearances']
    mids = m[m.season.isin(seasons) & ((m.home == team) | (m.away == team))].match_id
    a = a[(a.team == team) & a.match_id.isin(mids)]
    cols = [c for c in a.columns if c.startswith('rmin_')]
    r = a.groupby('player_id')[cols].sum()
    mapping = ROLE_TO[scheme]
    g = pd.DataFrame({tgt: r[[f'rmin_{s}' for s, t in mapping.items() if t == tgt]].sum(axis=1)
                      for tgt in sorted(set(mapping.values()))})
    return g.idxmax(axis=1)


def unit_arrays(team, label, seasons, layer, scheme=None, min_minutes=None):
    pp = TAB['player_period']
    d = pp[(pp.team == team) & (pp.label == label)].copy()
    if scheme:
        d['group'] = d.player_id.map(scheme_groups(team, seasons, scheme))
    kc, tc = LAYER_SPEC[layer]
    d = d[(d.group != 'GK') & d.group.notna() & (d.minutes >= (min_minutes or CFG['min_minutes'])) & (d[tc] > 0)]
    return d[kc].to_numpy(float), d[tc].to_numpy(float), d.group.to_numpy(), list(d.name)


def league_constants(scheme):
    """League rates and EB alpha per group for an alternative grouping (2015/16)."""
    _, phi, _ = constants()
    rates, alpha = {}, {}
    m = TAB['matches']
    teams = sorted(set(m[m.season == CFG['league_season']].home))
    data = {t: {l: unit_arrays(t, CFG['league_season'], [CFG['league_season']], l, scheme) for l in LAYERS} for t in teams}
    for layer in LAYERS:
        allk = np.concatenate([data[t][layer][0] for t in teams])
        allT = np.concatenate([data[t][layer][1] for t in teams])
        allg = np.concatenate([data[t][layer][2] for t in teams])
        rates[layer] = {g: float(allk[allg == g].sum() / allT[allg == g].sum()) for g in sorted(set(allg))}
        alpha[layer] = {}
        for g in rates[layer]:
            tot = np.zeros(len(ALPHA_GRID))
            for t in teams:
                k, T, grp, _ = data[t][layer]
                sel = grp == g
                if sel.sum() < 2:
                    continue
                for j, a in enumerate(ALPHA_GRID):
                    tot[j] += fit_hubs_het(k[sel], T[sel], grp[sel], {g: rates[layer][g]}, {g: a},
                                           phi=phi[layer], s_max=3).log_ev
            alpha[layer][g] = float(ALPHA_GRID[int(np.argmax(tot))])
    return rates, alpha


def variants():
    v = [('base', {})]
    v += [('rho_max=2', {'rho_max': 2.0}), ('rho_max=10', {'rho_max': 10.0}),
          ('prior shape 0.5', {'a': 0.5}), ('prior shape 2', {'a': 2.0}),
          ('prior mean x0.5', {'rate_mult': 0.5}), ('prior mean x2', {'rate_mult': 2.0}),
          ('min minutes 450', {'min_minutes': 450}), ('s_max=4', {'s_max': 4}), ('no phi correction', {'phi': 1.0}),
          ('alpha x0.5', {'alpha_mult': 0.5}), ('alpha x2', {'alpha_mult': 2.0}),
          ('wingers in midfield', {'scheme': 'wingers_mid'}), ('six roles', {'scheme': 'roles6'})]
    return v


def run_variant(args):
    name, opt, consts = args
    rates0, phi0, alpha0 = constants()
    out = {}
    for team, label, seasons in units():
        for layer in LAYERS:
            k, T, g, names = unit_arrays(team, label, seasons, layer, opt.get('scheme'), opt.get('min_minutes'))
            if len(k) < 3:
                continue
            if opt.get('scheme'):
                rates, alpha = consts[opt['scheme']][0][layer], consts[opt['scheme']][1][layer]
            else:
                rates = {gg: r * opt.get('rate_mult', 1.0) for gg, r in rates0[layer].items()}
                alpha = {gg: a * opt.get('alpha_mult', 1.0) for gg, a in alpha0[layer].items()}
            f = fit_hubs_het(k, T, g, rates, alpha, a=opt.get('a', CFG['model']['prior_shape']),
                             rho_max=opt.get('rho_max', CFG['model']['rho_max']), s_max=opt.get('s_max', CFG['model']['s_max']),
                             phi=opt.get('phi', phi0[layer]))
            out[f'{team}|{label}|{layer}'] = {'hubs': [names[i] for i in f.hubs],
                                              'incl': {names[i]: float(p) for i, p in enumerate(f.incl_prob) if p >= 0.2},
                                              'log_bf': float(f.log_bf)}
    return name, out


def main(workers=8):
    _init()
    consts = {}
    with Pool(2, initializer=_init) as pool:
        for scheme, c in zip(ROLE_TO, pool.map(league_constants, list(ROLE_TO))):
            consts[scheme] = c
    with Pool(workers, initializer=_init) as pool:
        res = dict(pool.map(run_variant, [(n, o, consts) for n, o in variants()]))
    base = res['base']
    summary = []
    key_players = {'att': 'Lionel Messi', 'prog': 'Xavi', 'def': 'Sergio Busquets', 'org': 'Xavi', 'box': 'Lionel Messi'}
    for name, out in res.items():
        row = {'variant': name}
        for l in LAYERS:
            row[f'hubs_{l}'] = sum(len(v['hubs']) for k, v in out.items() if k.endswith('|' + l))
            row[f'{key_players[l]}|{l}'] = sum(key_players[l] in v['hubs'] for k, v in out.items()
                                               if k.startswith('Barcelona') and k.endswith('|' + l))
        row['share_units_unchanged'] = float(np.mean([set(out[k]['hubs']) == set(base[k]['hubs']) for k in base if k in out]))
        if 'box' in LAYERS:          # names used by scripts/make_figures.py (v1 tables)
            row['messi_threat_seasons'] = row['Lionel Messi|box']
        if 'org' in LAYERS:
            row['xavi_org_seasons'] = row['Xavi|org']
        summary.append(row)
    dump({'constants_alt_groupings': consts, 'summary': summary, 'fits': res}, 'sensitivity.json')
    for s in summary:
        print(s)


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:2]))
