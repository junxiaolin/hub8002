"""Step A (cont.): match-level edges, then player_period and edges for every team-period.

Periods (column `label`):
  "<season>"          a whole team-season (all 20 teams in 2015/16; Barcelona in every season)
  "<season> H1/H2"    first / second half of the season by date (E3)
  "<era name>"        merged Barcelona seasons (decision 12: Rijkaard 2004-08)
Groups are fixed per team-season (most minutes over the whole season), also for the halves.
"""
import pandas as pd

from hub8002 import load_config, load_table, save_table
from hub8002.aggregate import aggregate, build_match_edges, player_groups, season_halves

cfg = load_config()
matches = load_table('matches')
apps = load_table('appearances')
passes = load_table('passes')
try:
    values = load_table('player_match_values')
except FileNotFoundError:
    values = None

medges = build_match_edges(apps, passes)
save_table(medges, 'match_edges')
print('match_edges', len(medges))


def team_matches(team, seasons):
    m = matches[matches.season.isin(seasons) & ((matches.home == team) | (matches.away == team))]
    return list(m.sort_values(['date', 'match_id']).match_id)


pp_all, ed_all = [], []
units = [(t, cfg['league_season']) for t in sorted(set(matches[matches.season == cfg['league_season']].home))]
units += [(cfg['focus_team'], s) for s in cfg['seasons'].values() if s != cfg['league_season']]
for team, season in units:
    ids = team_matches(team, [season])
    gmap = player_groups(apps, ids, team)
    h1, h2 = season_halves(matches, team, season)
    for label, sub in ((season, ids), (f'{season} H1', h1), (f'{season} H2', h2)):
        pp, ed = aggregate(apps, passes, medges, sub, team, label=label, group_map=gmap, values=values)
        half = label[len(season):].strip() or 'full'
        pp['season'], pp['half'], pp['n_matches'] = season, half, len(sub)
        ed['season'], ed['half'] = season, half
        pp_all.append(pp)
        ed_all.append(ed)
for era, seasons in cfg['eras'].items():
    team = cfg['focus_team']
    ids = team_matches(team, seasons)
    pp, ed = aggregate(apps, passes, medges, ids, team, label=era, values=values)
    pp['season'], pp['half'], pp['n_matches'] = era, 'era', len(ids)
    ed['season'], ed['half'] = era, 'era'
    pp_all.append(pp)
    ed_all.append(ed)

player_period = pd.concat(pp_all, ignore_index=True)
edges = pd.concat(ed_all, ignore_index=True)
save_table(player_period, 'player_period')
save_table(edges, 'edges')
print('player_period', len(player_period), 'edges', len(edges))
