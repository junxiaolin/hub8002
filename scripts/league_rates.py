"""League base rates (A -> B, D), dispersion phi (B) and between-player heterogeneity (B), every layer.

All from 2015/16, all 20 teams, outfield players. Layers (see LAYER_SPEC): att / prog / def are values
in expected-goal units per 90 minutes of own (att, prog) or opponent (def) possession on the pitch;
org / box are pass counts per 90 minutes played.
  results/league_rates.json : rate per group and layer (model prior mean, simulation base rate)
  results/dispersion.json   :
    phi_match  Pearson dispersion of per-match values around each player's own constant rate
               (count layers: over-dispersion; value layers: the size of one "effective count")
    het        dispersion of season totals among players of the same team and group around the group rate
"""
import json

import numpy as np
import pandas as pd

from hub8002 import ROOT, load_config, load_table
from hub8002.experiments.common import LAYER_SPEC

cfg = load_config()
season = cfg['league_season']
matches = load_table('matches')
apps = load_table('appearances')
passes = load_table('passes')
vals = load_table('player_match_values')
pp = load_table('player_period')
ids = set(matches[matches.season == season].match_id)

# ---- base rates per group (season tables, every outfield player)
full = pp[(pp.season == season) & (pp.half == 'full') & pp.group.isin(cfg['hub_groups'])]
rates = {}
for layer, (kc, tc) in LAYER_SPEC.items():
    rates[layer] = {g: float(d[kc].sum() / d[tc].sum()) for g, d in full.groupby('group')}
    rates[layer]['ALL'] = float(full[kc].sum() / full[tc].sum())
json.dump(rates, open(ROOT / 'results' / 'league_rates.json', 'w'), indent=1)
print('league rates:', json.dumps({l: {g: round(v, 4) for g, v in r.items()} for l, r in rates.items()}))

# ---- per-appearance table with every layer
a = apps[apps.match_id.isin(ids) & (apps.group != 'GK')][['match_id', 'team', 'player_id', 'minutes']]
p = passes[passes.match_id.isin(ids)]
for layer, col in (('org', 'is_org'), ('box', 'is_box')):
    c = p[p[col]].groupby(['match_id', 'passer_id']).size().rename(f'k_{layer}')
    a = a.merge(c, left_on=['match_id', 'player_id'], right_index=True, how='left')
    a[f'k_{layer}'] = a[f'k_{layer}'].fillna(0)
v = vals[vals.match_id.isin(ids)][['match_id', 'player_id', 'att', 'prog', 'def', 'own_poss_min', 'opp_poss_min']]
v = v.rename(columns={'att': 'k_att', 'prog': 'k_prog', 'def': 'k_def'})
a = a.merge(v, on=['match_id', 'player_id'], how='left').fillna(0)
a['T'] = a.minutes / 90.0
a['T_own'] = a.own_poss_min / 90.0
a['T_opp'] = a.opp_poss_min / 90.0

disp = {'phi_match': {}, 'het': {}}
for min_minutes in (1, 30):
    d = a[a.minutes >= min_minutes]
    for layer, (kc, tc) in LAYER_SPEC.items():
        g = d.groupby(['team', 'player_id'])
        r = g[kc].transform('sum') / g[tc].transform('sum')
        mu = r * d[tc]
        ok = mu > 0
        x2 = float((((d[kc] - mu) ** 2)[ok] / mu[ok]).sum())
        n_obs, n_par = int(ok.sum()), int(g.ngroups)
        disp['phi_match'][f'{layer}_min{min_minutes}'] = {'phi': x2 / (n_obs - n_par), 'n_obs': n_obs, 'n_players': n_par}

for layer, (kc, tc) in LAYER_SPEC.items():
    phi = disp['phi_match'][f'{layer}_min1']['phi']
    for thr in (450, 900):
        d = full[full.minutes >= thr]
        rows = []
        for (team, grp), g in d.groupby(['team', 'group']):
            if len(g) < 2:
                continue
            lam = g[kc].sum() / g[tc].sum()
            mu = g[tc] * lam
            k = g[kc]
            if lam <= 0:
                continue
            rows.append({'team': team, 'group': grp, 'n': len(g), 'x2': float(((k - mu) ** 2 / mu).sum()) / phi,
                         'inv_alpha_terms': list(((k - mu) ** 2 - phi * mu) / mu ** 2)})
        h = pd.DataFrame(rows)
        res = {'pearson_ratio_over_phi': float(h.x2.sum() / (h.n - 1).sum()), 'phi_match': phi}
        for grp, hg in h.groupby('group'):
            terms = np.concatenate(hg.inv_alpha_terms.values)
            inv_a = max(float(np.mean(terms)) * np.mean(hg.n / (hg.n - 1)), 1e-6)
            res[grp] = {'pearson_ratio_over_phi': float(hg.x2.sum() / (hg.n - 1).sum()),
                        'inv_alpha_mom': inv_a, 'alpha_mom': 1.0 / inv_a, 'cv_individual': inv_a ** 0.5}
        disp['het'][f'{layer}_min{thr}'] = res
json.dump(disp, open(ROOT / 'results' / 'dispersion.json', 'w'), indent=1)
for layer in LAYER_SPEC:
    h = disp['het'][f'{layer}_min450']
    print(layer, 'phi', round(disp['phi_match'][f'{layer}_min1']['phi'], 4),
          {g: (round(h[g]['pearson_ratio_over_phi'], 1), round(h[g]['cv_individual'], 2)) for g in ('DEF', 'MID', 'FWD')})
