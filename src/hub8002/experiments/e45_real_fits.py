"""E4 (Barcelona, 17 seasons) and E5 (La Liga 2015/16, 20 teams): every method on every unit, both layers.

E4 units: the Rijkaard era (2004/05-2007/08 merged, decision 12) and each season from 2008/09;
the four early seasons are also fitted one by one (marked `early_single`).
E5 units: the 20 teams of 2015/16, whole season.
Writes results/e4_barca.json and results/e5_league.json.
Usage: python -I -m hub8002.experiments.e45_real_fits [workers]
"""
import sys
from multiprocessing import Pool

import numpy as np

from .. import load_table
from .common import ALL_METHODS, BAYES, CFG, LAYERS, dump, run_methods, unit_data

PP = ED = None


def _init():
    global PP, ED
    PP = load_table('player_period')
    ED = load_table('edges')


def fit_unit(args):
    team, label = args
    out = {'team': team, 'label': label, 'layers': {}}
    for layer in LAYERS:
        k, T, group, names, ids, pr = unit_data(PP, team, label, layer, ED)
        if len(k) < 3:
            continue
        res, extra = run_methods(ALL_METHODS, k, T, group, layer, pr=pr)
        d = PP[(PP.team == team) & (PP.label == label)]
        out['n_matches'] = int(d.n_matches.iloc[0])
        out['layers'][layer] = {
            'players': [{'id': int(i), 'name': n, 'group': g, 'minutes': float(t * 90), 'k': int(kk),
                         'per90': float(kk / t)} for i, n, g, t, kk in zip(ids, names, group, T, k)],
            'hubs': {m: [names[i] for i in h] for m, h in res.items()},
            'hub_ids': {m: [int(ids[i]) for i in h] for m, h in res.items()},
            'incl_prob': {m: extra[m]['incl_prob'].round(4).tolist() for m in BAYES},
            'log_bf': {m: extra[m]['log_bf'] for m in BAYES},
        }
    return out


def main(workers=12):
    barca = CFG['focus_team']
    era_seasons = [s for v in CFG['eras'].values() for s in v]
    e4_units = [(barca, era) for era in CFG['eras']]
    e4_units += [(barca, s) for s in CFG['seasons'].values() if s not in era_seasons]
    e4_early = [(barca, s) for s in era_seasons]
    matches = load_table('matches')
    e5_units = [(t, CFG['league_season']) for t in sorted(set(matches[matches.season == CFG['league_season']].home))]
    with Pool(workers, initializer=_init) as pool:
        res = pool.map(fit_unit, e4_units + e4_early + e5_units)
    n4, n4e = len(e4_units), len(e4_early)
    e4 = res[:n4]
    for r in res[n4:n4 + n4e]:
        r['early_single'] = True
    dump({'units': e4, 'early_single': res[n4:n4 + n4e]}, 'e4_barca.json')
    dump({'units': res[n4 + n4e:]}, 'e5_league.json')
    for r in e4:
        for layer in LAYERS:
            if layer in r['layers']:
                L = r['layers'][layer]
                print(r['label'], layer, 'NB:', L['hubs']['bayes_nb'], 'P:', L['hubs']['bayes_p'],
                      'MDL-CM:', L['hubs']['mdl_cm'][:4])


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:2]))
