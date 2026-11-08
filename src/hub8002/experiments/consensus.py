"""Face validity (2026-11-08): do our cores agree with widely accepted judgements?

References (docs/consensus.yaml, with sources): Ballon d'Or top three, LFP / La Liga awards, and every
2015/16 team's top scorer (counted from the event data, penalties included). Each is checked against
our cores under three versions: the main runs (average player, three groups), value over the
replacement level (rep_real.json) and the data-selected eight-role partition (partition_real.json).
Per reference: is the player the top core / in the core set of a matching layer, and his best rank.
Also, for the top scorers, their rate relative to the positional average (why they can rank low).
Writes results/consensus.json.  Usage: python -I -m hub8002.experiments.consensus
"""
import json
from collections import Counter

import numpy as np
import yaml

from .. import ROOT, load_table
from ..io_statsbomb import load_events
from .common import CFG, RESULTS, dump

LAYERS = ('att', 'prog', 'def', 'link')


def load_versions():
    main = {}
    for f in ('v2_barca.json', 'v2_league.json'):
        for u in json.load(open(RESULTS / f))['units']:
            main.setdefault(f"{u['team']}|{u['label']}", {}).update(
                {L: {'top': u['layers'][L]['top'], 'core_set': u['layers'][L]['core_set'],
                     'ranking': [p['name'] for p in sorted(u['layers'][L]['players'], key=lambda p: -p['voa'])]}
                 for L in ('att', 'prog', 'def') if L in u['layers']})
    for f in ('net_barca.json', 'net_league.json'):
        for u in json.load(open(RESULTS / f))['units']:
            if u.get('mode', 'net') == 'net':
                main.setdefault(f"{u['team']}|{u['label']}", {})['link'] = {
                    'top': u['top'], 'core_set': u['core_set'], 'ranking': [p['name'] for p in u['players']]}
    out = {'average (main)': main}
    for name, f, key in (('replacement level', 'rep_real.json', 'bench50'),
                         ('eight roles', 'partition_real.json', 'eight'), ('four groups', 'partition_real.json', 'four')):
        fits = json.load(open(RESULTS / f))['fits'][key]
        out[name] = {k: {L: {'top': v['layers'][L]['top'], 'core_set': v['layers'][L]['core_set'],
                             'ranking': [p['name'] for p in v['layers'][L]['players']]}
                         for L in v['layers']} for k, v in fits.items()}
    return out


def check(ver, team, label, player, layers):
    u = ver.get(f'{team}|{label}')
    if not u:
        return None
    layers = LAYERS if layers == ['any'] else layers
    ranks = [u[L]['ranking'].index(player) + 1 for L in layers if L in u and player in u[L]['ranking']]
    return {'top': any(u[L]['top'] == player for L in layers if L in u),
            'core': any(player in u[L]['core_set'] for L in layers if L in u),
            'best_rank': min(ranks) if ranks else None}


def top_scorers():
    m = load_table('matches')
    goals, pens = Counter(), Counter()
    for mid in m[m.season == CFG['league_season']].match_id:
        for e in load_events(mid):
            if e['type']['name'] == 'Shot' and e['shot']['outcome']['name'] == 'Goal' and e.get('period', 1) < 5:
                goals[(e['team']['name'], e['player']['id'])] += 1
                pens[(e['team']['name'], e['player']['id'])] += e['shot']['type']['name'] == 'Penalty'
    rates = json.load(open(RESULTS / 'league_rates.json'))['att']
    v2 = {u['team']: u for u in json.load(open(RESULTS / 'v2_league.json'))['units']}
    out = []
    for team, u in sorted(v2.items()):
        pid, g = max(((p, c) for (t, p), c in goals.items() if t == team), key=lambda x: x[1])
        pl = next((p for p in u['layers']['att']['players'] if p['id'] == pid), None)
        out.append({'team': team, 'player': pl['name'] if pl else str(pid), 'goals': g, 'penalties': pens[(team, pid)],
                    'group': pl['group'] if pl else None,
                    'rate_vs_position_average': pl['rate'] / rates[pl['group']] if pl else None})
    return out


def main():
    refs = yaml.safe_load(open(ROOT / 'docs' / 'consensus.yaml'))
    versions = load_versions()
    items = [dict(kind='ballon_dor', layers=['any'], **r) for r in refs['ballon_dor']] + \
            [dict(kind='lfp', **r) for r in refs['lfp']] + \
            [dict(kind='top_scorer', label=CFG['league_season'], layers=['att'], **s) for s in top_scorers()]
    rows, summary = [], {}
    for it in items:
        row = {k: it[k] for k in ('kind', 'label', 'team', 'player', 'layers') if k in it}
        row.update({k: it[k] for k in ('year', 'rank', 'award', 'goals', 'penalties', 'rate_vs_position_average', 'check')
                    if k in it})
        row['result'] = {name: check(ver, it['team'], it['label'], it['player'], it['layers'])
                         for name, ver in versions.items()}
        rows.append(row)
    for kind in ('ballon_dor', 'lfp', 'top_scorer'):
        rs = [r for r in rows if r['kind'] == kind and all(v is not None for v in r['result'].values())]
        summary[kind] = {'n': len(rs), **{name: {'top': float(np.mean([r['result'][name]['top'] for r in rs])),
                                                 'core': float(np.mean([r['result'][name]['core'] for r in rs])),
                                                 'top3': float(np.mean([(r['result'][name]['best_rank'] or 99) <= 3
                                                                        for r in rs]))}
                                         for name in versions}}
    dump({'summary': summary, 'items': rows}, 'consensus.json')
    for kind, s in summary.items():
        print(kind, 'n', s['n'], {k: v for k, v in s.items() if k != 'n'})
    misses = [r for r in rows if r['result']['average (main)'] and not r['result']['average (main)']['core']]
    print('not in any matching core set (main):')
    for r in misses:
        print('  ', r['kind'], r['label'], r['player'], r.get('award', r.get('year', '')),
              {k: v['best_rank'] for k, v in r['result'].items() if v})


if __name__ == '__main__':
    main()
