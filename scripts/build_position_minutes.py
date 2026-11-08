"""Minutes per player, match and raw StatsBomb position (2026-11-08), so that any position partition
(three groups, six roles, eight roles with holding / central / attacking midfield apart, ...) can be
built and compared. Same minute accounting as build_tables.py (real half lengths, red cards).
Writes data/interim/position_minutes.parquet: match_id, team, player_id, pmin_<code> per raw position.
Usage: python -I scripts/build_position_minutes.py [workers]
"""
import json
import sys
from multiprocessing import Pool

import pandas as pd

from hub8002 import load_config, save_table
from hub8002.io_statsbomb import load_events, load_lineups, match_rows
from hub8002.minutes import Clock, appearances, period_lengths

CFG = load_config()
# raw StatsBomb position -> short code
CODE = {'Goalkeeper': 'GK', 'Right Back': 'RB', 'Right Center Back': 'RCB', 'Center Back': 'CB',
        'Left Center Back': 'LCB', 'Left Back': 'LB', 'Right Wing Back': 'RWB', 'Left Wing Back': 'LWB',
        'Right Defensive Midfield': 'RDM', 'Center Defensive Midfield': 'CDM', 'Left Defensive Midfield': 'LDM',
        'Right Midfield': 'RM', 'Right Center Midfield': 'RCM', 'Center Midfield': 'CM',
        'Left Center Midfield': 'LCM', 'Left Midfield': 'LM', 'Right Attacking Midfield': 'RAM',
        'Center Attacking Midfield': 'CAM', 'Left Attacking Midfield': 'LAM', 'Right Wing': 'RW',
        'Left Wing': 'LW', 'Right Center Forward': 'RCF', 'Center Forward': 'CF', 'Left Center Forward': 'LCF',
        'Secondary Striker': 'SS'}


def one(m):
    mid = m['match_id']
    try:
        events = load_events(mid)
        lineups = load_lineups(mid)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return mid, None, repr(exc)
    clock = Clock(period_lengths(events))
    apps, _ = appearances(lineups, clock, mid, CFG['position_groups'], CODE)
    keep = ['match_id', 'team', 'player_id', 'minutes'] + [f'rmin_{c}' for c in sorted(set(CODE.values()))]
    rows = [{k.replace('rmin_', 'pmin_'): a[k] for k in keep} for a in apps]
    return mid, rows, None


def main(workers=8):
    out, errors = [], []
    with Pool(workers) as pool:
        for mid, rows, err in pool.imap_unordered(one, match_rows(CFG), chunksize=4):
            if err:
                errors.append((mid, err))
            else:
                out.extend(rows)
    df = pd.DataFrame(out)
    save_table(df, 'position_minutes')
    pcols = [c for c in df.columns if c.startswith('pmin_')]
    print(f'rows {len(df)}  errors {len(errors)}  max |sum(pmin) - minutes| = '
          f'{(df[pcols].sum(axis=1) - df.minutes).abs().max():.4f}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
