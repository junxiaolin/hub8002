"""v2 real data: value cores (attack / organisation / defence, plus pass volume as contrast) for
Barcelona (Rijkaard era + every season; the four early seasons also singly) and the 20 teams of
La Liga 2015/16. Also records the pass-network "hubs" of v1 (most passes, top PageRank, MDL-CM per
minute) for comparison. Writes results/v2_barca.json and results/v2_league.json.
Usage: python -I -m hub8002.experiments.v2_real
"""
import json

import numpy as np

from .. import load_table
from ..models.baselines import baseline_hubs
from ..models.core_model import value_cores
from .common import CFG, RESULTS, constants, dump, unit_data

V2_LAYERS = ('att', 'prog', 'def', 'org')


def v2_constants():
    rates, phi, _ = constants()
    alpha = json.load(open(RESULTS / 'alpha_voa.json'))
    return rates, phi, alpha


def fit_unit(pp, ed, team, label, opts=None):
    opts = opts or {}
    rates, phi, alpha = opts.get('consts') or v2_constants()
    out = {'team': team, 'label': label, 'layers': {}}
    d = pp[(pp.team == team) & (pp.label == label)]
    if len(d):
        out['n_matches'] = int(d.n_matches.iloc[0])
    for layer in V2_LAYERS:
        k, T, g, names, ids, pr = unit_data(pp, team, label, layer, ed, min_minutes=opts.get('min_minutes'),
                                            groups=opts.get('groups'))
        ok = T > 0
        k, T, g = k[ok], T[ok], g[ok]
        names = [n for n, o in zip(names, ok) if o]
        ids = [i for i, o in zip(ids, ok) if o]
        if pr is not None:
            pr = [p for p, o in zip(pr, ok) if o]
        if len(k) < 2:
            continue
        base = rates[layer]
        if opts.get('baseline') == 'team':      # compare with team-mates of the same position instead of the league
            base = {gg: (k[g == gg].sum() + rates[layer][gg]) / (T[g == gg].sum() + 1.0) for gg in set(g)}
            base = {**rates[layer], **base}
        al = {gg: a * opts.get('alpha_mult', 1.0) for gg, a in alpha[layer].items()}
        vb = opts.get('voa_base', {}).get(layer)  # e.g. replacement level {group: rate}; default = base
        f = value_cores(k, T, g, base, al, phi=phi[layer], cred=opts.get('cred', 0.9),
                        pos_prob=opts.get('pos_prob', 0.9), voa_base=vb)
        L = {'players': [{'id': int(i), 'name': n, 'group': gg, 'T': float(t), 'k': float(kk),
                          'rate': float(rm), 'voa': float(v), 'voa_lo': float(lo), 'voa_hi': float(hi),
                          'p_pos': float(pp_), 'p_top': float(pt)}
                         for i, n, gg, t, kk, rm, v, lo, hi, pp_, pt in
                         zip(ids, names, g, T, k, f.rate_mean, f.voa_mean, f.voa_lo, f.voa_hi, f.p_pos, f.p_top)],
             'core_set': [names[i] for i in f.core_set], 'top': names[f.top], 'p_top_max': float(f.p_top.max()),
             'clear': f.clear, 'team_total': float(k.sum())}
        if layer == 'org' and pr is not None:
            L['v1_pass_hubs'] = {m: [names[i] for i in baseline_hubs(m, k, T, pr)]
                                 for m in ('count_top', 'pagerank_top', 'mdl_cm_min')}
        out['layers'][layer] = L
    return out


def units():
    m = load_table('matches')
    era_seasons = [s for v in CFG['eras'].values() for s in v]
    barca = [('Barcelona', e) for e in CFG['eras']] + [('Barcelona', s) for s in CFG['seasons'].values()
                                                        if s not in era_seasons]
    early = [('Barcelona', s) for s in era_seasons]
    league = [(t, CFG['league_season']) for t in sorted(set(m[m.season == CFG['league_season']].home))]
    return barca, early, league


def main():
    pp = load_table('player_period')
    ed = load_table('edges')
    barca, early, league = units()
    b = [fit_unit(pp, ed, t, l) for t, l in barca]
    e = [fit_unit(pp, ed, t, l) for t, l in early]
    lg = [fit_unit(pp, ed, t, l) for t, l in league]
    dump({'units': b, 'early_single': e}, 'v2_barca.json')
    dump({'units': lg}, 'v2_league.json')
    for u in b:
        print(u['label'], ' | '.join(f"{l}: {u['layers'][l]['core_set'] or '-'} (top {u['layers'][l]['top']} "
                                     f"{u['layers'][l]['p_top_max']:.2f})" for l in V2_LAYERS if l in u['layers']))
    print('---')
    for u in lg:
        print(u['team'][:16], ' | '.join(f"{l}: {u['layers'][l]['core_set'] or '-'}" for l in V2_LAYERS))


if __name__ == '__main__':
    main()
