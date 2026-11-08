"""Real 2015/16 squad structures for the simulator: who played, minutes per match, groups, co-presence."""
import numpy as np

from .. import load_config, load_table

CFG = load_config()


def squad_structures(season=None, min_minutes=None):
    """{team: dict(player_ids, names, group, minutes (matches x players, date order), copresence (players x players))}.

    Players: outfield, with at least `min_minutes` over the season (decision 7). Groups: season group.
    """
    season = season or CFG['league_season']
    min_minutes = CFG['min_minutes'] if min_minutes is None else min_minutes
    matches = load_table('matches')
    apps = load_table('appearances')
    pp = load_table('player_period')
    medges = load_table('match_edges')
    vals = load_table('player_match_values')
    ms = matches[matches.season == season].sort_values(['date', 'match_id'])
    out = {}
    for team in sorted(set(ms.home)):
        mids = list(ms[(ms.home == team) | (ms.away == team)].match_id)
        d = pp[(pp.team == team) & (pp.label == season)]
        cand = d[d.group.isin(CFG['hub_groups']) & (d.minutes >= min_minutes)]
        ids = list(cand.player_id)
        pos = {p: i for i, p in enumerate(ids)}
        a = apps[(apps.team == team) & apps.match_id.isin(mids) & apps.player_id.isin(ids)]
        M = np.zeros((len(mids), len(ids)))
        OWN = np.zeros_like(M)
        OPP = np.zeros_like(M)
        mrow = {m: i for i, m in enumerate(mids)}
        for mid, pid, mn in zip(a.match_id, a.player_id, a.minutes):
            M[mrow[mid], pos[pid]] = mn
        v = vals[(vals.team == team) & vals.match_id.isin(mids) & vals.player_id.isin(ids)]
        for mid, pid, o1, o2 in zip(v.match_id, v.player_id, v.own_poss_min, v.opp_poss_min):
            OWN[mrow[mid], pos[pid]] = o1
            OPP[mrow[mid], pos[pid]] = o2
        e = medges[(medges.team == team) & medges.match_id.isin(mids) & medges.i.isin(ids) & medges.j.isin(ids)]
        C = np.zeros((len(ids), len(ids)))
        for i, j, c in zip(e.i, e.j, e.copresence_min):
            C[pos[i], pos[j]] += c
        out[team] = {'player_ids': ids, 'names': list(cand.name), 'group': cand.group.to_numpy(),
                     'minutes': M, 'own': OWN, 'opp': OPP, 'copresence': C}
    return out
