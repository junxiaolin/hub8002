"""E7: after how many matches can a hub be confirmed? (real data, cumulative)

For each team-season, the model (bayes_nb) is refitted on the first m matches, m = 1..n (groups fixed
from the whole season). For every hub of the full season (MAP set at m = n), we record the first m
from which its inclusion probability stays above 0.9 until the end of the season.
Units: Barcelona, every season; the 20 teams of 2015/16. Both layers.
Usage: python -I -m hub8002.experiments.e7_sequential [workers]
"""
import sys
from multiprocessing import Pool

import numpy as np

from .. import load_table
from ..aggregate import aggregate, player_groups
from .common import CFG, GROUPS, LAYER_SPEC, LAYERS, dump, run_bayes

TAB = None


def _init():
    global TAB
    TAB = {n: load_table(n) for n in ('matches', 'appearances', 'passes', 'match_edges', 'player_match_values')}


def run_unit(args):
    team, season = args
    m, apps, passes, medges = TAB['matches'], TAB['appearances'], TAB['passes'], TAB['match_edges']
    mids = list(m[(m.season == season) & ((m.home == team) | (m.away == team))].sort_values(['date', 'match_id']).match_id)
    gmap = player_groups(apps, mids, team)
    traj = {l: {} for l in LAYERS}
    names = {}
    for n in range(1, len(mids) + 1):
        pp, _ = aggregate(apps, passes, medges, mids[:n], team, group_map=gmap, values=TAB['player_match_values'])
        cand = pp[pp.group.isin(GROUPS) & (pp.minutes >= CFG['min_minutes'])]
        if len(cand) < 3:
            continue
        names.update(dict(zip(cand.player_id, cand.name)))
        for layer in LAYERS:
            kc, tc = LAYER_SPEC[layer]
            c2 = cand[cand[tc] > 0]
            f = run_bayes('bayes_nb', c2[kc].to_numpy(float), c2[tc].to_numpy(float), c2.group.to_numpy(), layer)
            for pid, p in zip(c2.player_id, f.incl_prob):
                traj[layer].setdefault(int(pid), {})[n] = float(p)
            traj[layer].setdefault('_hubs', {})[n] = [int(c2.player_id.iloc[i]) for i in f.hubs]
    out = {'team': team, 'season': season, 'n_matches': len(mids), 'names': {int(k): v for k, v in names.items()},
           'layers': {}}
    nmax = len(mids)
    for layer in LAYERS:
        final = traj[layer].get('_hubs', {}).get(nmax, [])
        confirm, confirm05 = {}, {}
        for pid in final:
            series = traj[layer][pid]
            for thr, store in ((0.9, confirm), (0.5, confirm05)):
                first = None
                for n in range(1, nmax + 1):
                    if all(series.get(x, 0.0) > thr for x in range(n, nmax + 1)):
                        first = n
                        break
                store[pid] = first
        out['layers'][layer] = {'final_hubs': final, 'confirm_at': confirm, 'confirm_at_0.5': confirm05,
                                'final_incl': {pid: traj[layer][pid].get(nmax) for pid in final},
                                'trajectories': {str(k): v for k, v in traj[layer].items() if k != '_hubs'}}
    return out


def main(workers=12):
    matches = load_table('matches')
    units = [(CFG['focus_team'], s) for s in CFG['seasons'].values()]
    units += [(t, CFG['league_season']) for t in sorted(set(matches[matches.season == CFG['league_season']].home))
              if t != CFG['focus_team']]
    with Pool(workers, initializer=_init) as pool:
        res = pool.map(run_unit, units)
    conf = {l: [c for r in res for c in r['layers'][l]['confirm_at'].values() if c is not None] for l in LAYERS}
    frac = {l: [c / r['n_matches'] for r in res for c in r['layers'][l]['confirm_at'].values() if c is not None]
            for l in LAYERS}
    conf05 = {l: [c / r['n_matches'] for r in res for c in r['layers'][l]['confirm_at_0.5'].values() if c is not None]
              for l in LAYERS}
    summary = {l: {'n_final_hubs': sum(len(r['layers'][l]['final_hubs']) for r in res),
                   'n_hubs': len(conf[l]), 'median_confirm_match': float(np.median(conf[l])) if conf[l] else None,
                   'median_confirm_fraction': float(np.median(frac[l])) if frac[l] else None,
                   'n_confirm_0.5': len(conf05[l]),
                   'median_confirm_fraction_0.5': float(np.median(conf05[l])) if conf05[l] else None} for l in LAYERS}
    dump({'units': res, 'summary': summary}, 'e7_sequential.json')
    print(summary)


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:2]))
