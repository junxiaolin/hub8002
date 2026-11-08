"""Step A checks -> results/data_check.json (+ a readable results/data_check.md).

1. files: every match in the lists has an events and a lineup file that parsed
2. minutes: per team-match, sum of minutes = 11 x duration - minutes lost to sendings-off (+-1 min)
3. goalkeepers: exactly one starting goalkeeper per team-match
4. off reasons: counts of how players left the pitch
5. position groups: players whose main group holds < 70% of their season minutes
6. Messi's share of Barcelona's passes into the box, new vs pilot definition
7. five random matches with substitution times, for a manual check against match reports
"""
import json
import random

import numpy as np
import pandas as pd

from hub8002 import ROOT, load_config, load_table
from hub8002.minutes import parse_intervals

cfg = load_config()
matches = load_table('matches')
apps = load_table('appearances')
passes = load_table('passes')
pp = load_table('player_period')
out = {}

ids = [int(x) for x in open(ROOT / 'data' / 'match_ids.txt').read().split()]
out['files'] = {'expected': len(ids), 'parsed': int(matches.match_id.nunique()),
                'missing': sorted(set(ids) - set(matches.match_id))}

# 2. minutes
dur = matches.set_index('match_id').duration_min
rows = []
for (mid, team), g in apps.groupby(['match_id', 'team']):
    d = dur[mid]
    lost = float(np.sum(d - g.red_card_min.dropna().clip(upper=d)))
    # also minutes lost when a player leaves without replacement (permanent player-off / injury, subs used up)
    grid = np.linspace(0, d, 2001)[:-1] + d / 4000
    on = np.zeros_like(grid)
    for s in g.intervals:
        for a, b in parse_intervals(s):
            on += (grid >= a) & (grid < b)
    integral = float(on.mean() * d)
    temp_off = bool((g.intervals.str.count(';') > 0).any()
                    or g.off_reason.isin(['Player Off', 'Injury', 'Replaced after Player Off']).any())
    rows.append({'match_id': mid, 'team': team, 'duration': d, 'sum_minutes': float(g.minutes.sum()),
                 'temp_off': temp_off,
                 'red_lost': lost, 'dev': 11 * d - float(g.minutes.sum()) - lost,
                 'max_on_pitch': int(on.max()), 'min_on_pitch': int(on.min()),
                 'short_handed_min': 11 * d - integral})
mc = pd.DataFrame(rows)
mc['ok'] = mc.dev.abs() <= 1.0
mc['ok_explained'] = mc.ok | ((mc.max_on_pitch <= 11) & (mc.dev > 0) & mc.temp_off)
bad = mc[~mc.ok_explained]
out['minutes'] = {
    'team_matches': len(mc), 'within_1_min': int(mc.ok.sum()), 'share_ok': float(mc.ok.mean()),
    'ok_or_explained_by_treatment': int(mc.ok_explained.sum()),
    'unexplained': int((~mc.ok_explained).sum()),
    'max_on_pitch_gt_11': int((mc.max_on_pitch > 11).sum()),
    'with_red_card': int((mc.red_lost > 0).sum()),
    'deviations': bad.sort_values('dev', key=abs, ascending=False).head(25)
                     .round(2).to_dict('records')}

# 3. goalkeepers among starters
starters = apps[apps.started]
gk = starters.assign(gk=starters.min_GK > 0).groupby(['match_id', 'team']).gk.sum()
out['goalkeepers'] = {'team_matches': int(len(gk)), 'exactly_one': int((gk == 1).sum()),
                      'exceptions': gk[gk != 1].reset_index().to_dict('records')}

# 4. off reasons
out['off_reasons'] = apps.off_reason.value_counts().to_dict()

# 5. group shares (whole team-seasons only)
full = pp[pp.half == 'full']
low = full[(full.group_share < 0.7) & (full.minutes >= cfg['min_minutes'])]
out['group_share_below_70'] = {
    'n_player_seasons': int(len(full[full.minutes >= cfg['min_minutes']])), 'n_below': int(len(low)),
    'list': low.sort_values('minutes', ascending=False)[['team', 'season', 'name', 'group', 'group_share', 'minutes']]
               .round(3).to_dict('records')}

# 6. Messi's share of passes into the box
MESSI = 5503
mm = matches[(matches.home == 'Barcelona') | (matches.away == 'Barcelona')][['match_id', 'season']]
bp = passes[(passes.team == 'Barcelona')].merge(mm, on='match_id')
share = {}
for season, g in bp.groupby('season'):
    row = {}
    for col in ('is_box', 'is_box_pilot'):
        gg = g[g[col]]
        by = gg.groupby('passer_id').size().sort_values(ascending=False)
        row[col] = {'messi_share': float(by.get(MESSI, 0) / max(len(gg), 1)),
                    'messi_rank': int(list(by.index).index(MESSI) + 1) if MESSI in by.index else None,
                    'team_total': int(len(gg))}
    share[season] = row
out['messi_box_share'] = share

# 7. five random matches for manual checks
random.seed(cfg['seed'])
sample = random.sample(sorted(matches.match_id), 5)
man = []
for mid in sample:
    m = matches[matches.match_id == mid].iloc[0]
    a = apps[(apps.match_id == mid) & (~apps.started | (apps.off_reason != 'Final Whistle'))]
    man.append({'match_id': int(mid), 'date': m.date, 'fixture': f'{m.home} {m.home_score}-{m.away_score} {m.away}',
                'changes': a.sort_values(['team', 'on_min'])[['team', 'name', 'started', 'on_min', 'off_min', 'off_reason']]
                           .round(1).to_dict('records')})
out['manual_check_sample'] = man

json.dump(out, open(ROOT / 'results' / 'data_check.json', 'w'), ensure_ascii=False, indent=1, default=str)

# readable summary
L = ['# 数据检查报告', '',
     f"- 比赛文件：应有 {out['files']['expected']}，成功解析 {out['files']['parsed']}，缺失 {len(out['files']['missing'])}",
     f"- 上场分钟：{out['minutes']['within_1_min']}/{out['minutes']['team_matches']} 个队-场误差 ≤ 1 分钟"
     f"（{out['minutes']['share_ok']:.1%}）；其余中可由伤停离场（Player Off / 伤退无人替补）解释的 "
     f"{out['minutes']['ok_or_explained_by_treatment'] - out['minutes']['within_1_min']}，无法解释 {out['minutes']['unexplained']}；"
     f"含红牌的队-场 {out['minutes']['with_red_card']}；"
     f"任一时刻场上超过 11 人的队-场 {out['minutes']['max_on_pitch_gt_11']}",
     f"- 首发门将：{out['goalkeepers']['exactly_one']}/{out['goalkeepers']['team_matches']} 个队-场恰好 1 名",
     f"- 下场原因：{out['off_reasons']}",
     f"- 主位置组占比 < 70% 的球员-赛季：{out['group_share_below_70']['n_below']}/"
     f"{out['group_share_below_70']['n_player_seasons']}（名单见 json，交 E 过目）",
     '', '## 梅西传进禁区的份额', '', '| 赛季 | 新定义份额 | 排名 | 全队 | 试跑定义份额 | 排名 | 全队 |', '|---|---|---|---|---|---|---|']
for s, r in share.items():
    a, b = r['is_box'], r['is_box_pilot']
    L.append(f"| {s} | {a['messi_share']:.1%} | {a['messi_rank']} | {a['team_total']} | "
             f"{b['messi_share']:.1%} | {b['messi_rank']} | {b['team_total']} |")
L += ['', '## 无法解释的分钟误差（前 25 个）', '',
      '| match_id | team | duration | sum_minutes | red_lost | dev | short_handed_min |', '|---|---|---|---|---|---|---|']
for r in out['minutes']['deviations']:
    L.append(f"| {r['match_id']} | {r['team']} | {r['duration']} | {r['sum_minutes']} | {r['red_lost']} | {r['dev']} | {r['short_handed_min']} |")
open(ROOT / 'results' / 'data_check.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[:8]))
