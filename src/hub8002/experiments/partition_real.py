"""Cores under the data-selected position partition (2026-11-08; see partitions.py).

Selected: 'eight' (CB / FB / DM / CM / AM / WM / W / CF) as the groups players are compared with, in all
four layers; in the network null model the receiving blocks are 'four' (GK / CB / FB / MID / FWD).
'seven' (DM apart, CM + AM together) was a close second and is reported as a check; 'four' (CB / FB / MID /
FWD: labels that do not depend on the formation) as the simplest partition that beats 'three' in every layer.
Everything else as in the main runs (league 2015/16 calibration, same priors, same decision rule).
Writes results/partition_real.json.  Usage: python -I -m hub8002.experiments.partition_real
"""
import numpy as np

from .. import load_table
from ..models.net_model import build_unit, eb_alpha, league_profiles, net_cores
from .common import CFG, dump
from .net_common import calib, label_matches, league_teams, tables
from .partitions import PARTITIONS, _dst_names, player_groups
from .v2_checks import constants_for
from .v2_real import V2_LAYERS, fit_unit, units

DST = 'four'
CHOICES = ('eight', 'seven', 'four')


def node_tables(src):
    """player_period with `group` = the partition's group (per team-period), and its league constants."""
    _, pp, _, _ = tables()
    rows = []
    for (team, label), d in pp.groupby(['team', 'label']):
        g = player_groups(team, label, src)
        dd = d.copy()
        dd['group'] = dd.player_id.map(lambda p: g.get(int(p)))
        rows.append(dd[dd.group.notna()])
    import pandas as pd
    p = pd.concat(rows, ignore_index=True)
    groups = tuple(sorted(set(PARTITIONS[src].values()) - {'GK'}))
    return p, constants_for(p, groups), groups


def net_calib(src, dst=DST):
    ch, _, _, _ = tables()
    season = CFG['league_season']
    units_ = [build_unit(ch, t, label_matches(t, season), player_groups(t, season, src),
                         player_groups(t, season, dst)) for t in league_teams()]
    dest = _dst_names(dst)
    srcs = tuple(sorted({g for u in units_ for g in u.group}))
    q, gbar = league_profiles(units_, dest, srcs)
    _, _, _, phi, a = calib()
    alpha = {g: eb_alpha(units_, q, g, phi) for g in srcs}
    return q, gbar, alpha, phi, a


def net_fit(team, label, src, cal, dst=DST):
    ch, pp, _, _ = tables()
    q, gbar, alpha, phi, a = cal
    u = build_unit(ch, team, label_matches(team, label), player_groups(team, label, src),
                   player_groups(team, label, dst))
    d = pp[(pp.team == team) & (pp.label == label)].set_index('player_id')
    names = [str(d['name'].get(p, p)) for p in u.ids]
    minutes = np.array([float(d['minutes'].get(p, 0.0)) for p in u.ids])
    cand = (u.group != 'GK') & (minutes >= CFG['min_minutes'])
    f = net_cores(u, q, gbar, alpha, cand, phi=phi, shot_shape=a)
    return {'core_set': [names[i] for i in f.core_set], 'top': names[f.top], 'p_top_max': float(f.p_top.max()),
            'players': sorted([{'name': names[i], 'group': str(u.group[i]), 'voa': float(f.voa_mean[i]),
                                'p_top': float(f.p_top[i])} for i in np.flatnonzero(cand)], key=lambda r: -r['voa'])}


def main():
    ed = load_table('edges')
    barca, _, league = units()
    all_units = barca + [x for x in league if x[0] != 'Barcelona']
    import json
    from .common import RESULTS
    base_v2 = {f"{u['team']}|{u['label']}": u for f in ('v2_barca.json', 'v2_league.json')
               for u in json.load(open(RESULTS / f))['units']}
    base_net = {f"{u['team']}|{u['label']}": u for f in ('net_barca.json', 'net_league.json')
                for u in json.load(open(RESULTS / f))['units'] if u.get('mode', 'net') == 'net'}
    out = {'dst_blocks': DST, 'fits': {}, 'summary': {}, 'constants': {}}
    for src in CHOICES:
        p, consts, groups = node_tables(src)
        cal = net_calib(src)
        out['constants'][src] = {'node_rates': consts[0], 'node_alpha': consts[2], 'net_alpha': cal[2]}
        fits = {}
        for t, l in all_units:
            v = fit_unit(p, ed, t, l, {'consts': consts, 'groups': groups})
            fits[f'{t}|{l}'] = {'layers': {L: {'top': v['layers'][L]['top'], 'core_set': v['layers'][L]['core_set'],
                                               'players': [{'name': r['name'], 'group': r['group'], 'voa': r['voa'],
                                                            'p_top': r['p_top']} for r in
                                                           sorted(v['layers'][L]['players'], key=lambda r: -r['voa'])]}
                                           for L in ('att', 'prog', 'def') if L in v['layers']}}
            fits[f'{t}|{l}']['layers']['link'] = net_fit(t, l, src, cal)
        out['fits'][src] = fits
        summ = {}
        for L in ('att', 'prog', 'def', 'link'):
            keys = list(fits)
            base = (lambda k: base_net[k]) if L == 'link' else (lambda k: base_v2[k]['layers'][L])
            summ[L] = {'top_same_as_three': float(np.mean([fits[k]['layers'][L]['top'] == base(k)['top'] for k in keys])),
                       'top_same_barca': float(np.mean([fits[k]['layers'][L]['top'] == base(k)['top']
                                                        for k in keys if k.startswith('Barcelona')])),
                       'n_with_core': int(sum(bool(fits[k]['layers'][L]['core_set']) for k in keys)),
                       'n_with_core_three': int(sum(bool(base(k)['core_set']) for k in keys)), 'n_units': len(keys)}
        out['summary'][src] = summ
        print(src, {L: (round(s['top_same_as_three'], 2), s['n_with_core'], s['n_with_core_three']) for L, s in summ.items()})
    a, b = out['fits']['eight'], out['fits']['seven']
    out['summary']['eight_vs_seven_top_same'] = {L: float(np.mean([a[k]['layers'][L]['top'] == b[k]['layers'][L]['top']
                                                                   for k in a])) for L in ('att', 'prog', 'def', 'link')}
    print('eight vs seven', out['summary']['eight_vs_seven_top_same'])
    dump(out, 'partition_real.json')
    for k, v in out['fits']['eight'].items():
        if k.startswith('Barcelona'):
            print(k.split('|')[1][:8], ' | '.join(f"{L}:{v['layers'][L]['top']}{v['layers'][L]['core_set'] or '无'}"
                                                 for L in ('att', 'prog', 'def', 'link')))


if __name__ == '__main__':
    main()
