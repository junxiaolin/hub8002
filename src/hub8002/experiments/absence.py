"""Absence check (2026-11-08): does a player's estimated value predict what happens when he is missing?

A check that does not rely on our own definition of the truth (the simulations do). La Liga 2015/16, the
only season with every match, so opponent strength can be adjusted for.

Outcome per team-match: non-penalty xG for (attack, progression, linking) or against (defence), from the
possession chains. Adjusted for team, opponent and home: OLS y = mu + attack_team + defence_opp + home,
residual e. For player i of team t: present = played >= 45 minutes, absent = did not play;
observed effect d_i = mean(e | absent) - mean(e | present) (for: a drop is d < 0; against: a rise is d > 0).
Players with >= 3 absences and >= 10 presences (cross-fitted: per half, >= 2 and >= 5).

Predictors (per 90 minutes played): our VOA (average baseline, three groups), VOA over the replacement
level, VOA under the data-selected eight-role partition and under four groups (CB / FB / MID / FWD), raw value per 90 (no position, no shrinkage),
passes per 90 (pass hub), and share of the team's minutes (the coach's choice) as a reference.
Cross-fitting (main): the predictors are estimated on one half of the season and the absence effect is
measured on the other half (both directions pooled), so that a lucky spell cannot raise both a player's
value and his team's xG in the same matches. The in-sample version (whole season for both) is reported
alongside to show how much that shared noise inflates the association.
Statistic: weighted correlation between predicted value and the observed effect (sign so that positive =
in the predicted direction), weights 1 / (1/n_absent + 1/n_present). Null: within each team, permute
the residuals over matches (keeps every player's absence pattern); 2000 permutations, one-sided p.
For linking (team xG units) also the weighted slope of the observed drop on the predicted value.
Writes results/absence.json.  Usage: python -I -m hub8002.experiments.absence
"""
import json

import numpy as np

from .. import load_table
from .common import CFG, RESULTS, dump

N_PERM = 2000
MIN_ABS, MIN_PRES = 3, 10


def team_match_xg():
    ch = load_table('chains')
    m = load_table('matches')
    m = m[m.season == CFG['league_season']]
    xg = ch[ch.j == -1].groupby(['match_id', 'team']).xg.sum()
    rows = []
    for r in m.itertuples():
        for team, opp, home in ((r.home, r.away, 1), (r.away, r.home, 0)):
            rows.append({'match_id': r.match_id, 'team': team, 'opp': opp, 'home': home,
                         'xg_for': float(xg.get((r.match_id, team), 0.0)),
                         'xg_against': float(xg.get((r.match_id, opp), 0.0))})
    return rows


def residuals(rows):
    """OLS of xG for on team attack + opponent defence + home; residuals for and against per team-match."""
    teams = sorted({r['team'] for r in rows})
    ix = {t: k for k, t in enumerate(teams)}
    nt = len(teams)
    X = np.zeros((len(rows), 1 + 2 * nt + 1))
    y = np.array([r['xg_for'] for r in rows])
    for a, r in enumerate(rows):
        X[a, 0] = 1
        X[a, 1 + ix[r['team']]] = 1
        X[a, 1 + nt + ix[r['opp']]] = 1
        X[a, -1] = r['home']
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    res_for = {(r['match_id'], r['team']): e[a] for a, r in enumerate(rows)}
    res_against = {(r['match_id'], r['team']): res_for[(r['match_id'], r['opp'])] for r in rows}
    return res_for, res_against


def predictors_fit(label):
    """{(team, name): {predictor: {layer: value per 90}}}, all models fitted on the matches of `label`."""
    from .net_real import fit as net_fit
    from .partition_real import DST, net_calib, net_fit as net_fit_p, node_tables
    from .v2_real import fit_unit
    pp, ed = load_table('player_period'), load_table('edges')
    bench = json.load(open(RESULTS / 'bench_rates.json'))
    p8, c8, g8 = _part('eight')
    p4, c4, g4 = _part('four')
    d = pp[pp.label == label]
    mins = {(t, n): m for t, n, m in zip(d.team, d.name, d.minutes)}
    out = {}

    def put(pred, layer, team, name, voa):
        m = mins.get((team, name))
        if m and m > 0:
            out.setdefault((team, name), {}).setdefault(pred, {})[layer] = voa / (m / 90.0)

    teams = sorted(set(load_table('matches').query('season == @CFG["league_season"]').home))
    for t in teams:
        for pred, opts, table in (('voa_average', {}, pp), ('voa_replacement', {'voa_base': bench['node']['bench50']}, pp),
                                  ('voa_eight_roles', {'consts': c8, 'groups': g8}, p8),
                                  ('voa_four_groups', {'consts': c4, 'groups': g4}, p4)):
            v = fit_unit(table, ed, t, label, opts)
            for L in ('att', 'prog', 'def'):
                for p in v['layers'].get(L, {}).get('players', []):
                    put(pred, L, t, p['name'], p['voa'])
                    if pred == 'voa_average':
                        put('raw_per90', L, t, p['name'], p['k'])
        for pred, opts in (('voa_average', {}), ('voa_replacement', {'q_base': bench['net']['bench50']})):
            n = net_fit(t, label, opts={**opts, 'n_draws': 1000}, extras=False)
            for p in n['players']:
                put(pred, 'link', t, p['name'], p['voa'])
        for src, pred in (('eight', 'voa_eight_roles'), ('four', 'voa_four_groups')):
            for p in net_fit_p(t, label, src, _part_net(src))['players']:
                put(pred, 'link', t, p['name'], p['voa'])
    nm = d.groupby('team').n_matches.max()
    for t, n, k, m in zip(d.team, d.name, d.k_org, d.minutes):
        if (t, n) in out:
            for L in ('att', 'prog', 'def', 'link'):
                put('passes_per90', L, t, n, k)
                out[(t, n)].setdefault('minutes_share', {})[L] = m / (90.0 * nm[t])
    return out


_CACHE = {}


def _part(src):
    if ('node', src) not in _CACHE:
        from .partition_real import node_tables
        _CACHE[('node', src)] = node_tables(src)
    return _CACHE[('node', src)]


def _part_net(src):
    if ('net', src) not in _CACHE:
        from .partition_real import net_calib
        _CACHE[('net', src)] = net_calib(src)
    return _CACHE[('net', src)]


def wcorr(x, y, w):
    x, y, w = map(np.asarray, (x, y, w))
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


PRED_NAMES = ('voa_average', 'voa_replacement', 'voa_eight_roles', 'voa_four_groups', 'raw_per90', 'passes_per90',
              'minutes_share')


def evaluate(blocks, res_for, res_against, rng, n_perm=N_PERM):
    """blocks: list of (preds, {team: [match ids]}, min_abs, min_pres). Players are pooled over blocks; the
    permutation shuffles residuals within each team and block."""
    apps = load_table('appearances')
    played = {(r.team, r.name, r.match_id): r.minutes for r in apps.itertuples()}
    players, units = [], []
    for b, (preds, tm, min_abs, min_pres) in enumerate(blocks):
        for (team, name), pr in preds.items():
            ms = tm.get(team)
            if not ms:
                continue
            mins = np.array([played.get((team, name, mid), 0.0) for mid in ms])
            absent, present = mins == 0, mins >= 45
            if absent.sum() >= min_abs and present.sum() >= min_pres:
                players.append({'unit': (b, team), 'team': team, 'name': name, 'absent': absent, 'present': present,
                                'pred': pr, 'w': 1.0 / (1.0 / absent.sum() + 1.0 / present.sum())})
        units += [((b, t), ms) for t, ms in tm.items()]
    units = dict(units)
    E_for = {u: np.array([res_for[(mid, u[1])] for mid in ms]) for u, ms in units.items()}
    E_ag = {u: np.array([res_against[(mid, u[1])] for mid in ms]) for u, ms in units.items()}

    def effects(E):
        return np.array([E[p['unit']][p['absent']].mean() - E[p['unit']][p['present']].mean() for p in players])

    perms = [{u: rng.permutation(len(ms)) for u, ms in units.items()} for _ in range(n_perm)]
    out = {'n_player_units': len(players), 'layers': {}}
    for L in ('att', 'prog', 'link', 'def'):
        E = E_ag if L == 'def' else E_for
        sign = 1.0 if L == 'def' else -1.0        # positive = in the predicted direction
        obs = sign * effects(E)
        null_eff = [sign * effects({u: E[u][pm[u]] for u in units}) for pm in perms]
        res = {}
        for pn in PRED_NAMES:
            keep = np.array([pn in p['pred'] and L in p['pred'][pn] for p in players])
            if keep.sum() < 20:
                continue
            x = np.array([p['pred'][pn][L] for p, k in zip(players, keep) if k])
            w = np.array([p['w'] for p, k in zip(players, keep) if k])
            r = wcorr(x, obs[keep], w)
            null = np.array([wcorr(x, ne[keep], w) for ne in null_eff])
            res[pn] = {'n': int(keep.sum()), 'corr': float(r), 'p_value': float((1 + (null >= r).sum()) / (1 + n_perm))}
            if pn.startswith('voa') and L in ('att', 'link'):
                xm = np.average(x, weights=w)
                res[pn]['slope'] = float(np.sum(w * (x - xm) * obs[keep]) / np.sum(w * (x - xm) ** 2))
        res['mean_effect_xg'] = float(np.average(obs, weights=[p['w'] for p in players]))
        out['layers'][L] = res
        print('  ', L, {k: (round(v['corr'], 3), round(v['p_value'], 3)) for k, v in res.items() if isinstance(v, dict)})
    return out


def main():
    from ..aggregate import season_halves
    rows = team_match_xg()
    res_for, res_against = residuals(rows)
    m = load_table('matches')
    season = CFG['league_season']
    teams = sorted(set(m[m.season == season].home))
    full = {t: sorted(set(m[(m.season == season) & ((m.home == t) | (m.away == t))].match_id)) for t in teams}
    halves = {t: season_halves(m, t, season) for t in teams}
    h1 = {t: halves[t][0] for t in teams}
    h2 = {t: halves[t][1] for t in teams}
    rng = np.random.default_rng(CFG['seed'])
    print('cross-fitted (value from one half, absence effect in the other)')
    p1, p2 = predictors_fit(f'{season} H1'), predictors_fit(f'{season} H2')
    cross = evaluate([(p1, h2, 2, 5), (p2, h1, 2, 5)], res_for, res_against, rng)
    print('in-sample (whole season for both)')
    pf = predictors_fit(season)
    insample = evaluate([(pf, full, MIN_ABS, MIN_PRES)], res_for, res_against, rng)
    dump({'n_perm': N_PERM, 'cross_fitted': cross, 'in_sample': insample}, 'absence.json')


if __name__ == '__main__':
    main()
