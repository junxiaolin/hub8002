"""Value over a replacement-level player (2026-11-08): the same cores, but compared with what the league's
bench players of the position produce (results/bench_rates.json) instead of the league average.
Prior, data and decision rule unchanged; only the baseline of the value moves.
  bench50: players who started < 50% of their team's matches (main replacement level)
  bench25: < 25% (sensitivity)
Three position groups, as the main runs. Writes results/rep_real.json.
Usage: python -I -m hub8002.experiments.rep_real
"""
import json

import numpy as np

from .. import load_table
from .common import RESULTS, dump
from .net_real import fit as net_fit
from .v2_real import fit_unit, units

LEVELS = ('bench50', 'bench25')


def main():
    pp, ed = load_table('player_period'), load_table('edges')
    bench = json.load(open(RESULTS / 'bench_rates.json'))
    barca, _, league = units()
    all_units = barca + [x for x in league if x[0] != 'Barcelona']
    base_v2 = {f"{u['team']}|{u['label']}": u for f in ('v2_barca.json', 'v2_league.json')
               for u in json.load(open(RESULTS / f))['units']}
    base_net = {f"{u['team']}|{u['label']}": u for f in ('net_barca.json', 'net_league.json')
                for u in json.load(open(RESULTS / f))['units'] if u.get('mode', 'net') == 'net'}
    out = {'fits': {}, 'summary': {}}
    for lev in LEVELS:
        fits = {}
        for t, l in all_units:
            v = fit_unit(pp, ed, t, l, {'voa_base': bench['node'][lev]})
            n = net_fit(t, l, opts={'q_base': bench['net'][lev]}, extras=False)
            layers = {L: {'top': v['layers'][L]['top'], 'core_set': v['layers'][L]['core_set'],
                          'players': [{'name': r['name'], 'group': r['group'], 'voa': r['voa'], 'p_top': r['p_top'],
                                       'p_pos': r['p_pos']} for r in sorted(v['layers'][L]['players'], key=lambda r: -r['voa'])]}
                      for L in ('att', 'prog', 'def') if L in v['layers']}
            layers['link'] = {'top': n['top'], 'core_set': n['core_set'],
                              'players': [{'name': r['name'], 'group': r['group'], 'voa': r['voa'], 'p_top': r['p_top'],
                                           'p_pos': r['p_pos']} for r in n['players']]}
            fits[f'{t}|{l}'] = {'layers': layers}
        out['fits'][lev] = fits
        summ = {}
        for L in ('att', 'prog', 'def', 'link'):
            base = (lambda k: base_net[k]) if L == 'link' else (lambda k: base_v2[k]['layers'][L])
            keys = list(fits)
            summ[L] = {'top_same_as_average': float(np.mean([fits[k]['layers'][L]['top'] == base(k)['top'] for k in keys])),
                       'top_same_barca': float(np.mean([fits[k]['layers'][L]['top'] == base(k)['top']
                                                        for k in keys if k.startswith('Barcelona')])),
                       'n_with_core': int(sum(bool(fits[k]['layers'][L]['core_set']) for k in keys)),
                       'n_with_core_average': int(sum(bool(base(k)['core_set']) for k in keys)), 'n_units': len(keys)}
        out['summary'][lev] = summ
        print(lev, {L: (round(s['top_same_as_average'], 2), s['n_with_core'], s['n_with_core_average'])
                    for L, s in summ.items()})
    dump(out, 'rep_real.json')
    for k, v in out['fits']['bench50'].items():
        print(k[:28], ' | '.join(f"{L}:{v['layers'][L]['top']}{v['layers'][L]['core_set'] or '无'}"
                                 for L in ('att', 'prog', 'def', 'link')))


if __name__ == '__main__':
    main()
