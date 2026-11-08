"""Empirical-Bayes estimate of the individual-difference shape alpha_g (B).

For each layer and position group, alpha_g maximises the summed log marginal likelihood of the
heterogeneity-aware hub model over the 20 teams of 2015/16 (each team-group fitted on its own,
up to 3 hubs, so real hubs do not inflate the estimated spread). 'ALL' = one group of all outfield
players (for the no-position ablation). Writes results/alpha.json and results/alpha_profile.json.
"""
import json
from multiprocessing import Pool

import numpy as np

from hub8002 import load_table
from hub8002.experiments.common import CFG, GROUPS, LAYER_SPEC, RESULTS, constants, model_kwargs
from hub8002.models.bayes_hub import fit_hubs_het

GRID = np.exp(np.linspace(np.log(0.3), np.log(3000), 41))


def profile(args):
    layer, g = args
    rates, phi, _ = constants()
    pp = load_table('player_period')
    d = pp[(pp.season == CFG['league_season']) & (pp.half == 'full') & pp.group.isin(GROUPS)
           & (pp.minutes >= CFG['min_minutes'])]
    kw = model_kwargs()
    tot = np.zeros(len(GRID))
    for team, dt in d.groupby('team'):
        dg = dt if g == 'ALL' else dt[dt.group == g]
        if len(dg) < 2:
            continue
        kc, tc = LAYER_SPEC[layer]
        k, T = dg[kc].to_numpy(float), dg[tc].to_numpy(float)
        grp = np.array([g] * len(dg))
        for j, a in enumerate(GRID):
            tot[j] += fit_hubs_het(k, T, grp, {g: rates[layer][g]}, {g: a}, phi=phi[layer], **kw).log_ev
    return layer, g, tot


if __name__ == '__main__':
    jobs = [(l, g) for l in LAYER_SPEC for g in GROUPS + ('ALL',)]
    with Pool(min(len(jobs), 20)) as pool:
        res = pool.map(profile, jobs)
    alpha = {l: {} for l in LAYER_SPEC}
    prof = {l: {} for l in LAYER_SPEC}
    for layer, g, tot in res:
        j = int(np.argmax(tot))
        alpha[layer][g] = float(GRID[j])
        prof[layer][g] = {'grid': GRID.tolist(), 'log_ev': tot.tolist()}
        print(layer, g, 'alpha =', round(GRID[j], 2), ' individual CV =', round(GRID[j] ** -0.5, 3),
              ' at grid edge' if j in (0, len(GRID) - 1) else '')
    json.dump(alpha, open(RESULTS / 'alpha.json', 'w'), indent=1)
    json.dump(prof, open(RESULTS / 'alpha_profile.json', 'w'))
