"""Network extension: possession chains per match and team -> data/interim/chains.parquet
(columns match_id, team, i, j, n, xg; j = player id, or -1 SHOT, -2 END, -3 START).
Usage: python -I scripts/build_chains.py [workers]
"""
import sys
from multiprocessing import Pool

import pandas as pd

from hub8002 import load_table, save_table
from hub8002.chains import chain_counts
from hub8002.io_statsbomb import load_events


def one(mid):
    return [(mid, t, i, j, n, x) for (t, i, j), (n, x) in chain_counts(load_events(mid)).items()]


def main(workers=8):
    mids = list(load_table('matches').match_id)
    rows = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(one, mids, chunksize=8):
            rows.extend(r)
    df = pd.DataFrame(rows, columns=['match_id', 'team', 'i', 'j', 'n', 'xg'])
    save_table(df, 'chains')
    k = df.groupby('j').n.sum()
    per = df.groupby(['match_id', 'team']).n.sum()
    print('rows', len(df), '| chains', int(k.get(-3, 0)), 'shots', int(k.get(-1, 0)), 'ends', int(k.get(-2, 0)),
          'transitions', int(df[df.j >= 0].n.sum()), '| events per team-match', round(float(per.mean()), 1))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
