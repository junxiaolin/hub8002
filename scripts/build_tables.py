"""Step A: build data/interim/{matches,appearances,passes}.parquet from data/raw (one match per task)."""
import json
import sys
from multiprocessing import Pool

import pandas as pd

from hub8002 import ROOT, load_config, save_table
from hub8002.io_statsbomb import load_events, load_lineups, match_rows
from hub8002.minutes import Clock, appearances, period_lengths
from hub8002.passes import pass_rows

CFG = load_config()
OUT = ROOT / 'data' / 'interim'


def one(m):
    mid = m['match_id']
    try:
        events = load_events(mid)
        lineups = load_lineups(mid)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return mid, None, None, None, repr(exc)
    clock = Clock(period_lengths(events))
    apps, _ = appearances(lineups, clock, mid, CFG['position_groups'], CFG['position_roles'])
    passes = pass_rows(events, clock, mid, CFG)
    info = {'duration_min': clock.duration, **{f'len_p{p}': l for p, l in clock.lengths.items()}}
    return mid, info, apps, passes, None


def main(workers=8):
    rows = match_rows(CFG)
    infos, apps, passes, errors = {}, [], [], []
    with Pool(workers) as pool:
        for mid, info, a, p, err in pool.imap_unordered(one, rows, chunksize=4):
            if err:
                errors.append((mid, err))
                continue
            infos[mid] = info
            apps.extend(a)
            passes.extend(p)
    matches = pd.DataFrame(rows)
    matches = matches[matches.match_id.isin(infos)].copy()
    for k in ('duration_min', 'len_p1', 'len_p2'):
        matches[k] = matches.match_id.map(lambda i: infos[i].get(k))
    save_table(matches.sort_values(['season', 'date', 'match_id']).reset_index(drop=True), 'matches')
    save_table(pd.DataFrame(apps), 'appearances')
    save_table(pd.DataFrame(passes), 'passes')
    print(f'matches {len(matches)}  appearances {len(apps)}  passes {len(passes)}  errors {len(errors)}')
    for e in errors[:10]:
        print('  error', e)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
