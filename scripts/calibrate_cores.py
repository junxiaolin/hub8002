"""Empirical-Bayes alpha_g for the v2 value-core model: for each layer and position group, the shape of
the Gamma spread of players' rates around the league-average rate of the position (2015/16 La Liga,
outfield players with >= 90 minutes). Writes results/alpha_voa.json."""
import json

from hub8002 import load_table
from hub8002.experiments.common import CFG, GROUPS, LAYER_SPEC, RESULTS, constants
from hub8002.models.core_model import eb_alpha

LAYERS_V2 = ('att', 'prog', 'def', 'org')
rates, phi, _ = constants()
pp = load_table('player_period')
alpha = {}
for layer in LAYERS_V2:
    kc, tc = LAYER_SPEC[layer]
    d = pp[(pp.season == CFG['league_season']) & (pp.half == 'full') & pp.group.isin(GROUPS)
           & (pp.minutes >= CFG['min_minutes']) & (pp[tc] > 0)]
    alpha[layer] = {}
    for g in GROUPS:
        dg = d[d.group == g]
        alpha[layer][g] = eb_alpha(dg[kc], dg[tc], rates[layer][g], phi[layer])
    print(layer, {g: f"alpha {a:.1f} (CV {a ** -0.5:.2f})" for g, a in alpha[layer].items()})
json.dump(alpha, open(RESULTS / 'alpha_voa.json', 'w'), indent=1)
