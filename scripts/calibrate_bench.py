"""Replacement level (2026-11-08): what the league's bench players of each position produce.

WAR-style value compares a player with a replacement-level player, not with an average one. We measure
the replacement level directly: La Liga 2015/16 outfield players with >= 90 minutes who started fewer
than half of their team's matches ('bench50'; 'bench25': fewer than a quarter, sensitivity).

  node   rate per 90 minutes of the relevant possession phase, per layer (att / prog / def) and position
         (pooled: sum of values / sum of exposure), plus its ratio to the league average and the quantile
         of the fitted Gamma spread of players' rates it corresponds to
  net    the bench players' profile q over {pass to GK / DEF / MID / FWD, shoot, end} (as league_profiles)

Writes results/bench_rates.json.
"""
import json

import numpy as np
from scipy.stats import gamma

from hub8002.experiments.common import CFG, GROUPS, LAYER_SPEC, RESULTS
from hub8002.experiments.net_common import DEST_GROUPS, league_teams, tables, unit_for
from hub8002.models.net_model import category_counts

THRESHOLDS = {'bench50': 0.5, 'bench25': 0.25}


def main():
    _, pp, _, _ = tables()
    rates = json.load(open(RESULTS / 'league_rates.json'))
    alpha = json.load(open(RESULTS / 'alpha_voa.json'))
    season = CFG['league_season']
    d = pp[(pp.label == season) & pp.group.isin(GROUPS) & (pp.minutes >= CFG['min_minutes'])].copy()
    d['start_share'] = d.n_starts / d.n_matches
    units = {t: unit_for(t, season)[0] for t in league_teams()}
    out = {'definition': 'outfield players of La Liga 2015/16 with >= 90 minutes who started fewer than '
                         '50% (bench50) / 25% (bench25) of their team matches', 'node': {}, 'net': {}}
    for name, thr in THRESHOLDS.items():
        b = d[d.start_share < thr]
        node, info = {}, {}
        for layer in ('att', 'prog', 'def'):
            kc, tc = LAYER_SPEC[layer]
            node[layer], info[layer] = {}, {}
            for g in GROUPS:
                x = b[(b.group == g) & (b[tc] > 0)]
                r = float(x[kc].sum() / x[tc].sum())
                a = alpha[layer][g]
                node[layer][g] = r
                info[layer][g] = {'ratio_to_average': r / rates[layer][g],
                                  'gamma_quantile': float(gamma.cdf(r / rates[layer][g], a, scale=1 / a)),
                                  'n_players': int(len(x))}
        bench_ids = {(t, int(p)) for t, p in zip(b.team, b.player_id)}
        q = {}
        for g in DEST_GROUPS:
            tot = np.zeros(len(DEST_GROUPS) + 2)
            for t, u in units.items():
                mask = np.array([(u.group[i] == g) and ((t, int(u.ids[i])) in bench_ids) for i in range(len(u.ids))])
                if g == 'GK':           # keepers are never candidates: keep the league-average keeper
                    mask = u.group == 'GK'
                tot += category_counts(u, DEST_GROUPS)[mask].sum(0)
            q[g] = dict(zip(DEST_GROUPS + ('SHOT', 'END'), (tot / tot.sum()).tolist()))
        out['node'][name] = node
        out['node'][name + '_info'] = info
        out['net'][name] = q
        print(name, {l: {g: f"{v['ratio_to_average']:.2f}x q{v['gamma_quantile']:.2f}" for g, v in i.items()}
                     for l, i in info.items()})
        print('   END rate', {g: round(q[g]['END'], 3) for g in GROUPS})
    avg = json.load(open(RESULTS / 'net_calib.json'))['groups']['q']
    print('average END rate', {g: round(avg[g]['END'], 3) for g in GROUPS})
    json.dump(out, open(RESULTS / 'bench_rates.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
