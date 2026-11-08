"""Numbers used in the talk's opening hook, so that they come from a script (not copied by hand).

Messi's rank in passes made (all completed passes, the `org` layer) among Barcelona's regular starters
(started >= 50% of the team's matches in the season), by season total and per 90 minutes.
Writes results/hook_numbers.json.
"""
import json

from hub8002 import ROOT, load_table


def main():
    pp = load_table('player_period')
    b = pp[(pp.team == 'Barcelona') & ~pp.label.str.contains(' H') & ~pp.label.str.contains('Rijkaard')]
    rows = []
    for label, d in b.groupby('label'):
        reg = d[d.n_starts >= 0.5 * d.n_matches.max()].copy()
        reg['per90'] = reg.k_org / reg.minutes * 90
        m = reg[reg.name.str.contains('Messi', na=False)]
        if m.empty:
            continue
        i = m.index[0]
        rows.append({'season': label, 'n_regulars': len(reg),
                     'rank_total': int(reg.k_org.rank(ascending=False, method='min')[i]),
                     'rank_per90': int(reg.per90.rank(ascending=False, method='min')[i]),
                     'top_per90': str(reg.loc[reg.per90.idxmax(), 'name'])})
    late = [r for r in rows if r['season'] >= '2008/09']
    out = {'definition': 'regular starter = started >= 50% of the team matches; passes = org layer (completed passes made)',
           'seasons': rows,
           'from_2008_09': {'rank_per90_range': [min(r['rank_per90'] for r in late), max(r['rank_per90'] for r in late)],
                            'rank_total_range': [min(r['rank_total'] for r in late), max(r['rank_total'] for r in late)],
                            'n_regulars_range': [min(r['n_regulars'] for r in late), max(r['n_regulars'] for r in late)]}}
    json.dump(out, open(ROOT / 'results' / 'hook_numbers.json', 'w'), indent=1, ensure_ascii=False)
    print(out['from_2008_09'])
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
