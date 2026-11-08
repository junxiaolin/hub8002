"""Step A (v2): fit xT on all matches, then per appearance: values of the three kinds of contribution
and possession-adjusted exposure. Writes results/xt_grid.json and data/interim/player_match_values.parquet."""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from hub8002 import ROOT, load_table, save_table
from hub8002.io_statsbomb import load_events
from hub8002.minutes import Clock, parse_intervals, period_lengths
from hub8002.value import action_values, fit_xt, possession_exposure, possession_segments, xt_counts

XT = None


def counts(mid):
    return xt_counts(load_events(mid))


def _init(xt):
    global XT
    XT = xt


def values(args):
    mid, apps = args
    ev = load_events(mid)
    clock = Clock(period_lengths(ev))
    seg = possession_segments(ev, clock)
    segarr = (np.array([s[0] for s in seg]), np.array([s[1] for s in seg]), np.array([s[2] for s in seg]))
    vals = action_values(ev, XT)
    rows = []
    for team, pid, iv in apps:
        own, opp = possession_exposure(parse_intervals(iv), segarr, team)
        v = vals.get((team, pid), {})
        rows.append({'match_id': mid, 'team': team, 'player_id': pid, 'own_poss_min': own, 'opp_poss_min': opp,
                     **{k: v.get(k, 0.0) for k in ('att', 'prog', 'def', 'npxg', 'xa')}})
    poss = {}
    for s0, s1, tm in seg:
        poss[tm] = poss.get(tm, 0.0) + (s1 - s0)
    return rows, mid, poss


def main(workers=8):
    matches = load_table('matches')
    apps = load_table('appearances')
    mids = list(matches.match_id)
    with Pool(workers) as pool:
        c = pool.map(counts, mids, chunksize=8)
    xt = fit_xt(c)
    json.dump({'nx': xt.shape[0], 'ny': xt.shape[1], 'xt': xt.round(6).tolist()}, open(ROOT / 'results' / 'xt_grid.json', 'w'))
    print('xT fitted: max', round(float(xt.max()), 3), 'at cell', np.unravel_index(xt.argmax(), xt.shape))
    by_match = {mid: list(zip(g.team, g.player_id, g.intervals)) for mid, g in apps.groupby('match_id')}
    rows, poss = [], {}
    with Pool(workers, initializer=_init, initargs=(xt,)) as pool:
        for r, mid, p in pool.imap_unordered(values, [(m, by_match[m]) for m in mids], chunksize=4):
            rows.extend(r)
            poss[mid] = p
    df = pd.DataFrame(rows)
    save_table(df, 'player_match_values')
    json.dump({str(k): v for k, v in poss.items()}, open(ROOT / 'results' / 'possession_minutes.json', 'w'))
    print('player_match_values', len(df), df[['att', 'prog', 'def', 'own_poss_min', 'opp_poss_min']].describe().round(3).to_string())


if __name__ == '__main__':
    main()
