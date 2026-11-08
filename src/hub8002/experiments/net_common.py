"""Shared helpers for the network extension (linking core): units per team-period, calibration."""
import json
from functools import lru_cache

import numpy as np

from .. import load_table
from ..aggregate import season_halves
from ..models.net_model import build_unit
from .common import CFG, GROUPS, RESULTS

ROLES = ('CB', 'FB', 'CM', 'WM', 'W', 'CF')
DEST_GROUPS = ('GK',) + tuple(GROUPS)
DEST_ROLES = ('GK',) + ROLES


@lru_cache(None)
def tables():
    return load_table('chains'), load_table('player_period'), load_table('matches'), load_table('appearances')


def team_match_ids(matches, team, seasons):
    m = matches[matches.season.isin(seasons) & ((matches.home == team) | (matches.away == team))]
    return list(m.sort_values(['date', 'match_id']).match_id)


def label_matches(team, label):
    _, _, matches, _ = tables()
    if label in CFG['eras']:
        return team_match_ids(matches, team, CFG['eras'][label])
    season, _, half = label.partition(' ')
    if half:
        h1, h2 = season_halves(matches, team, season)
        return h1 if half == 'H1' else h2
    return team_match_ids(matches, team, [season])


@lru_cache(None)
def role_map(team, label):
    """player -> fine role (most minutes) over the season (or era) of `label`."""
    _, _, matches, apps = tables()
    season = label.split(' ')[0] if label not in CFG['eras'] else label
    seasons = CFG['eras'].get(season, [season])
    a = apps[(apps.team == team) & apps.match_id.isin(team_match_ids(matches, team, seasons))]
    cols = [c for c in a.columns if c.startswith('rmin_')]
    r = a.groupby('player_id')[cols].sum()
    return {int(p): x for p, x in r.idxmax(axis=1).str.replace('rmin_', '', regex=False).items()}


def unit_for(team, label, *, roles=False, match_ids=None, min_minutes=None):
    """-> (NetUnit, candidate mask, names, minutes) for one team-period."""
    ch, pp, _, _ = tables()
    d = pp[(pp.team == team) & (pp.label == label)].set_index('player_id')
    groups = role_map(team, label) if roles else {int(p): g for p, g in d['group'].items()}
    ids = label_matches(team, label) if match_ids is None else match_ids
    u = build_unit(ch, team, ids, groups)
    names = [str(d['name'].get(p, p)) for p in u.ids]
    minutes = np.array([float(d['minutes'].get(p, 0.0)) for p in u.ids])
    mm = CFG['min_minutes'] if min_minutes is None else min_minutes
    cand = np.isin(u.group, ROLES if roles else GROUPS) & (minutes >= mm)
    return u, cand, names, minutes


def league_teams():
    _, _, matches, _ = tables()
    return sorted(set(matches[matches.season == CFG['league_season']].home))


def calib(roles=False):
    c = json.load(open(RESULTS / 'net_calib.json'))['roles' if roles else 'groups']
    return c['q'], c['gbar'], c['alpha'], c['phi'], c['shot_shape']
