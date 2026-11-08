"""Reading StatsBomb open-data files (matches, events, lineups) from data/raw."""
import json
from pathlib import Path

from . import ROOT

RAW = ROOT / 'data' / 'raw'


def load_match_list(season_id, raw=RAW):
    return json.load(open(Path(raw) / 'matches' / f'11_{season_id}.json'))


def load_events(match_id, raw=RAW):
    return json.load(open(Path(raw) / 'events' / f'{match_id}.json'))


def load_lineups(match_id, raw=RAW):
    return json.load(open(Path(raw) / 'lineups' / f'{match_id}.json'))


def match_rows(cfg, raw=RAW):
    """One dict per match in the 17 La Liga seasons that we downloaded (Barcelona + all of 2015/16)."""
    rows = []
    for sid, season in cfg['seasons'].items():
        for m in load_match_list(sid, raw):
            home = m['home_team']['home_team_name']
            away = m['away_team']['away_team_name']
            if season != cfg['league_season'] and cfg['focus_team'] not in (home, away):
                continue

            def mgr(side):
                ms = m[side].get('managers') or []
                return '; '.join(x.get('name', '') for x in ms) or None

            rows.append({'match_id': m['match_id'], 'season_id': sid, 'season': season,
                         'date': m['match_date'], 'kick_off': m.get('kick_off'),
                         'home': home, 'away': away,
                         'home_score': m['home_score'], 'away_score': m['away_score'],
                         'home_manager': mgr('home_team'), 'away_manager': mgr('away_team')})
    return rows
