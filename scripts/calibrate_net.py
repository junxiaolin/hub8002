"""Network extension: calibrate on La Liga 2015/16 (all 20 teams).

  q[g]     what an average player of position g does with the ball (pass to a GK / DEF / MID / FWD
           team-mate, shoot, chain ends), pooled over the league
  gbar[g]  xG per shot of position g
  phi      quasi-multinomial scale (Pearson chi-square of per-match counts around each player's season
           proportions / df): how much more players' rows vary from match to match than multinomial
  alpha[g] Dirichlet concentration of players around the average row (empirical Bayes)

  shot_shape  a = mean^2 / variance of single-shot (non-penalty) xG: each shot's xG ~ Gamma(a, a / g)

Both for the 3 position groups (+ GK) and for the six fine roles (sensitivity).
Writes results/net_calib.json.
"""
import json

import numpy as np

from hub8002 import ROOT
from hub8002.io_statsbomb import load_events
from hub8002.experiments.net_common import (DEST_GROUPS, DEST_ROLES, label_matches, league_teams, role_map,
                                            tables, unit_for)
from hub8002.models.net_model import build_unit, category_counts, eb_alpha, league_profiles, pearson_phi


def calibrate(roles):
    season = '2015/16'
    dest = DEST_ROLES if roles else DEST_GROUPS
    teams = league_teams()
    units = [unit_for(t, season, roles=roles)[0] for t in teams]
    q, gbar = league_profiles(units, dest)
    ch, pp, _, _ = tables()
    per_team = []
    for t in teams:
        groups = role_map(t, season) if roles else {
            int(p): g for p, g in pp[(pp.team == t) & (pp.label == season)].set_index('player_id')['group'].items()}
        rows = {}
        cht = ch[ch.team == t]
        for mid, d in cht.groupby('match_id'):
            u = build_unit(d, t, [mid], groups)
            for p, r in zip(u.ids, category_counts(u, dest)):
                rows.setdefault(int(p), []).append(r)
        per_team.append(rows)
    phi = pearson_phi(per_team, q)
    alpha = {g: eb_alpha(units, q, g, phi) for g in dest}
    return {'q': q, 'gbar': gbar, 'phi': phi, 'alpha': alpha}


def shot_shape():
    _, _, matches, _ = tables()
    xg = [e['shot'].get('statsbomb_xg', 0.0) for mid in matches[matches.season == '2015/16'].match_id
          for e in load_events(mid) if e['type']['name'] == 'Shot' and e['shot']['type']['name'] != 'Penalty']
    xg = np.array(xg)
    return float(xg.mean() ** 2 / xg.var()), len(xg), float(xg.mean())


def main():
    out = {'groups': calibrate(False), 'roles': calibrate(True)}
    a, n, mu = shot_shape()
    print('single-shot xG: n', n, 'mean', round(mu, 4), 'shape a', round(a, 3))
    for k in out:
        out[k]['shot_shape'] = a
    json.dump(out, open(ROOT / 'results' / 'net_calib.json', 'w'), indent=1)
    for k, c in out.items():
        print(k, 'phi', round(c['phi'], 3), 'alpha', {g: round(a, 1) for g, a in c['alpha'].items()})
        for g, row in c['q'].items():
            print('  ', g, {kk: round(v, 3) for kk, v in row.items()}, 'xG/shot', round(c['gbar'][g], 3))


if __name__ == '__main__':
    main()
