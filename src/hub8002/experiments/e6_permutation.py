"""E6: does the model invent hubs on real data? (within-position permutation test)

In every match, the identities of players of the same position group (season group) who played are
shuffled; each match record (minutes and passes) travels with the new identity. Season totals are
then rebuilt per identity and the methods are refitted. Individual differences are wiped out while
real rotation, positions and match-to-match swings are kept, so every hub reported is a false one.
Units: 20 teams of 2015/16 + Barcelona's other seasons; 200 permutations each; both layers.
Usage: python -I -m hub8002.experiments.e6_permutation [reps] [workers]
"""
import sys
from multiprocessing import Pool

import numpy as np

from .. import load_table
from ..aggregate import player_groups
from .common import CFG, GROUPS, LAYER_SPEC, LAYERS, dump, run_methods

METHODS = ('bayes_nb', 'bayes_p', 'mdl_cm_min', 'mdl_cm')
REC = None


def unit_records(matches, apps, passes, team, season, values):
    mids = list(matches[(matches.season == season) & ((matches.home == team) | (matches.away == team))].match_id)
    gmap = player_groups(apps, mids, team)['group']
    a = apps[(apps.team == team) & apps.match_id.isin(mids)][['match_id', 'player_id', 'minutes']].copy()
    a['group'] = a.player_id.map(gmap)
    p = passes[(passes.team == team) & passes.match_id.isin(mids)]
    for layer, col in (('org', 'is_org'), ('box', 'is_box')):
        c = p[p[col]].groupby(['match_id', 'passer_id']).size()
        a[f'k_{layer}'] = [c.get((m, i), 0) for m, i in zip(a.match_id, a.player_id)]
    v = values[(values.team == team) & values.match_id.isin(mids)][['match_id', 'player_id', 'att', 'prog', 'def',
                                                                      'own_poss_min', 'opp_poss_min']]
    a = a.merge(v.rename(columns={'att': 'k_att', 'prog': 'k_prog', 'def': 'k_def'}), on=['match_id', 'player_id'], how='left')
    a = a.fillna({'k_att': 0.0, 'k_prog': 0.0, 'k_def': 0.0, 'own_poss_min': 0.0, 'opp_poss_min': 0.0})
    a['T'] = a.minutes
    a['T_own'] = a.own_poss_min
    a['T_opp'] = a.opp_poss_min
    a = a[a.group.isin(GROUPS)]
    return a.reset_index(drop=True)


def _init(records):
    global REC
    REC = records


def run_unit(args):
    key, reps, seed = args
    a = REC[key]
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(a.player_id.unique()))
    pos = {p: i for i, p in enumerate(ids)}
    gid = a.groupby('player_id').group.first().reindex(ids).to_numpy()
    blocks = [g.index.to_numpy() for _, g in a.groupby(['match_id', 'group'])]
    pid_idx = a.player_id.map(pos).to_numpy()
    mins = a.minutes.to_numpy()
    ks = {l: a[LAYER_SPEC[l][0]].to_numpy(float) for l in LAYERS}
    expo = {l: a[LAYER_SPEC[l][1]].to_numpy(float) for l in LAYERS}
    tally = {l: {m: 0 for m in METHODS} for l in LAYERS}
    sizes = {l: {m: 0 for m in METHODS} for l in LAYERS}
    for _ in range(reps):
        new = pid_idx.copy()
        for b in blocks:
            if len(b) > 1:
                new[b] = pid_idx[rng.permutation(b)]
        played = np.bincount(new, weights=mins, minlength=len(ids))
        keep = played >= CFG['min_minutes']
        for layer in LAYERS:
            k = np.bincount(new, weights=ks[layer], minlength=len(ids))
            T = np.bincount(new, weights=expo[layer], minlength=len(ids)) / 90.0
            ok = keep & (T > 0)
            res, _ = run_methods(METHODS, k[ok], T[ok], gid[ok], layer)
            for m, h in res.items():
                tally[layer][m] += len(h) > 0
                sizes[layer][m] += len(h)
    return key, {l: {m: {'fp_rate': tally[l][m] / reps, 'mean_size': sizes[l][m] / reps} for m in METHODS}
                 for l in LAYERS}


def main(reps=200, workers=12):
    matches = load_table('matches')
    apps = load_table('appearances')
    passes = load_table('passes')
    values = load_table('player_match_values')
    units = [(t, CFG['league_season']) for t in sorted(set(matches[matches.season == CFG['league_season']].home))]
    units += [(CFG['focus_team'], s) for s in CFG['seasons'].values() if s != CFG['league_season']]
    records = {f'{t}|{s}': unit_records(matches, apps, passes, t, s, values) for t, s in units}
    tasks = [(f'{t}|{s}', reps, CFG['seed'] + i) for i, (t, s) in enumerate(units)]
    with Pool(workers, initializer=_init, initargs=(records,)) as pool:
        res = dict(pool.map(run_unit, tasks))
    summary = {l: {m: float(np.mean([res[k][l][m]['fp_rate'] for k in res])) for m in METHODS} for l in LAYERS}
    dump({'reps': reps, 'units': res, 'mean_fp_rate': summary}, 'e6_permutation.json')
    print(summary)


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
