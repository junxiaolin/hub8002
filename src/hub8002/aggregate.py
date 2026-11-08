"""Any set of matches -> per-player count table and directed edge table for one team.

`match_edges` (built once) holds, per match and team, every ordered pair of players who shared the
pitch: pass counts in both layers and co-presence minutes. `aggregate` then only sums rows, so
E3 (half seasons) and E7 (first m matches) can call it many times cheaply.
"""
from itertools import permutations

import numpy as np
import pandas as pd

from .minutes import overlap, parse_intervals

GROUPS = ('GK', 'DEF', 'MID', 'FWD')
LAYER_COL = {'org': 'is_org', 'box': 'is_box', 'box_pilot': 'is_box_pilot'}
VALUE_COLS = ['att', 'prog', 'def', 'npxg', 'xa']


def build_match_edges(apps, passes):
    """Ordered pairs (i, j) per match/team with w_org, w_box, w_box_pilot and copresence_min."""
    ok = passes[passes.recipient_id.notna()].copy()
    ok['recipient_id'] = ok.recipient_id.astype('int64')
    w = (ok.groupby(['match_id', 'team', 'passer_id', 'recipient_id'])[['is_org', 'is_box', 'is_box_pilot']]
         .sum().rename(columns={'is_org': 'w_org', 'is_box': 'w_box', 'is_box_pilot': 'w_box_pilot'})
         .reset_index().rename(columns={'passer_id': 'i', 'recipient_id': 'j'}))
    rows = []
    for (mid, team), g in apps.groupby(['match_id', 'team']):
        ivs = {pid: parse_intervals(s) for pid, s in zip(g.player_id, g.intervals)}
        for a, b in permutations(ivs, 2):
            c = overlap(ivs[a], ivs[b])
            if c > 0:
                rows.append((mid, team, a, b, c))
    co = pd.DataFrame(rows, columns=['match_id', 'team', 'i', 'j', 'copresence_min'])
    e = co.merge(w, on=['match_id', 'team', 'i', 'j'], how='outer')
    for c in ('w_org', 'w_box', 'w_box_pilot', 'copresence_min'):
        e[c] = e[c].fillna(0)
    for c in ('w_org', 'w_box', 'w_box_pilot'):
        e[c] = e[c].astype('int64')
    return e


def player_groups(apps, match_ids, team):
    """Each player's group = the group with most minutes over `match_ids`; also the share of minutes."""
    a = apps[(apps.team == team) & apps.match_id.isin(match_ids)]
    m = a.groupby('player_id')[[f'min_{g}' for g in GROUPS]].sum()
    tot = m.sum(axis=1).replace(0, np.nan)
    grp = m.idxmax(axis=1).str.replace('min_', '', regex=False)
    share = m.max(axis=1) / tot
    return pd.DataFrame({'group': grp, 'group_share': share})


def aggregate(apps, passes, match_edges, match_ids, team, label='', group_match_ids=None, group_map=None,
              values=None):
    """Returns (player_period, edges) for `team` over `match_ids`.

    player_period: player_id, name, group, group_share, minutes, T (= minutes / 90), n_apps, n_starts,
                   k_org, k_box, k_box_pilot (made) and k_org_in, k_box_in (received); with `values`
                   (player_match_values) also k_att, k_prog, k_def, k_npxg, k_xa and T_own / T_opp
                   (own / opponent possession minutes on the pitch, in units of 90).
    edges: i, j, w_org, w_box, w_box_pilot, copresence_min.
    Groups come from `group_map` if given, else from minutes over `group_match_ids` (default: match_ids).
    """
    match_ids = set(match_ids)
    a = apps[(apps.team == team) & apps.match_id.isin(match_ids)]
    pp = a.groupby('player_id').agg(name=('name', 'last'), minutes=('minutes', 'sum'),
                                    n_apps=('match_id', 'nunique'), n_starts=('started', 'sum'))
    if group_map is None:
        group_map = player_groups(apps, group_match_ids or match_ids, team)
    pp = pp.join(group_map[['group', 'group_share']], how='left')
    p = passes[(passes.team == team) & passes.match_id.isin(match_ids)]
    for layer, col in LAYER_COL.items():
        pp[f'k_{layer}'] = p[p[col]].groupby('passer_id').size().reindex(pp.index, fill_value=0)
        if layer != 'box_pilot':
            pp[f'k_{layer}_in'] = (p[p[col]].groupby('recipient_id').size()
                                   .reindex(pp.index, fill_value=0))
    pp['T'] = pp['minutes'] / 90.0
    if values is not None:
        v = values[(values.team == team) & values.match_id.isin(match_ids)]
        sums = v.groupby('player_id')[VALUE_COLS + ['own_poss_min', 'opp_poss_min']].sum().reindex(pp.index, fill_value=0.0)
        for c in VALUE_COLS:
            pp[f'k_{c}'] = sums[c]
        pp['T_own'] = sums['own_poss_min'] / 90.0
        pp['T_opp'] = sums['opp_poss_min'] / 90.0
    pp.insert(0, 'team', team)
    pp.insert(1, 'label', label)
    e = match_edges[(match_edges.team == team) & match_edges.match_id.isin(match_ids)]
    edges = e.groupby(['i', 'j'])[['w_org', 'w_box', 'w_box_pilot', 'copresence_min']].sum().reset_index()
    edges.insert(0, 'team', team)
    edges.insert(1, 'label', label)
    return pp.reset_index(), edges


def season_halves(matches, team, season):
    """Match ids of a team-season split by date into first ceil(n/2) and the rest."""
    m = matches[(matches.season == season) & ((matches.home == team) | (matches.away == team))]
    ids = list(m.sort_values(['date', 'match_id']).match_id)
    h = (len(ids) + 1) // 2
    return ids[:h], ids[h:]
