"""Acceptance check for the simulator (B checks D): simulated 2015/16 seasons without planted hubs
should match the real data on position rates, exposure and between-player dispersion.
Writes results/generator_check.json."""
import json

import numpy as np

from hub8002 import ROOT, load_table
from hub8002.experiments.common import CFG, GROUPS, LAYERS, constants
from hub8002.sim.generator import rotation_minutes, simulate_team
from hub8002.sim.squads import squad_structures

rates, phi, alpha = constants()
sq = squad_structures()
pp = load_table('player_period')
real = pp[(pp.season == CFG['league_season']) & (pp.half == 'full') & pp.group.isin(GROUPS)
          & (pp.minutes >= CFG['min_minutes'])]
rng = np.random.default_rng(CFG['seed'])


def summarise(rows):
    """rows: list of (team, group, k, T) -> rate per group and Pearson ratio of season totals within team-group."""
    out = {}
    for g in GROUPS:
        ks = np.concatenate([k for t, gg, k, T in rows if gg == g])
        Ts = np.concatenate([T for t, gg, k, T in rows if gg == g])
        x2 = df = 0.0
        for t, gg, k, T in rows:
            if gg != g or len(k) < 2:
                continue
            lam = k.sum() / T.sum()
            mu = T * lam
            x2 += float(((k - mu) ** 2 / mu).sum())
            df += len(k) - 1
        out[g] = {'rate': float(ks.sum() / Ts.sum()), 'pearson_ratio': x2 / df}
    return out


res = {}
for layer in LAYERS:
    rr = [(t, g, d[f'k_{layer}'].to_numpy(float), d['T'].to_numpy(float))
          for (t, g), d in real.groupby(['team', 'group'])]
    real_s = summarise(rr)
    sims = []
    for rep in range(50):
        rows = []
        for team, s in sq.items():
            T = rotation_minutes(s['minutes'], 'real', rng)
            k, _ = simulate_team(T, s['group'], rates[layer], [], 1.0, phi[layer], rng, alpha=alpha[layer])
            for g in GROUPS:
                sel = s['group'] == g
                rows.append((team, g, k[sel].astype(float), T[sel]))
        sims.append(summarise(rows))
    sim_s = {g: {key: float(np.mean([s[g][key] for s in sims])) for key in ('rate', 'pearson_ratio')} for g in GROUPS}
    res[layer] = {g: {'real': real_s[g], 'sim': sim_s[g],
                      'rate_rel_err': abs(sim_s[g]['rate'] / real_s[g]['rate'] - 1)} for g in GROUPS}
T_real = real['T'].to_numpy()
T_sim = np.concatenate([rotation_minutes(s['minutes'], 'real', rng) for s in sq.values()])
res['exposure'] = {'real_mean_T': float(T_real.mean()), 'sim_mean_T': float(T_sim.mean()),
                   'real_n_players': int(len(T_real)), 'sim_n_players': int(len(T_sim))}
json.dump(res, open(ROOT / 'results' / 'generator_check.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
