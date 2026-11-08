"""Data review (2026-11-08): the manual checks of the acceptance criteria, as far as they can be scripted.

1. Minutes: the team-matches whose minute total is off and not explained by red cards / temporary
   absences — what the event data say about each (substitution, player off / on events vs lineup spells).
2. External check of five sample matches: our substitution minutes vs published match pages
   (docs/minutes_external_check.yaml, with sources).
3. Positions: the player-seasons whose main group has < 70% of their minutes — which raw StatsBomb
   positions they split between, and whether they appear among the reported cores.
Writes results/data_review.json and results/data_review.md.
"""
import json
from collections import Counter

import yaml

from hub8002 import ROOT, load_table
from hub8002.experiments.net_common import label_matches
from hub8002.io_statsbomb import load_events

R = ROOT / 'results'
FAM = {'GK': 'GK', 'RB': 'FB', 'LB': 'FB', 'RWB': 'FB', 'LWB': 'FB', 'RCB': 'CB', 'CB': 'CB', 'LCB': 'CB',
       'RDM': 'DM', 'CDM': 'DM', 'LDM': 'DM', 'RCM': 'CM', 'CM': 'CM', 'LCM': 'CM', 'RAM': 'AM', 'CAM': 'AM',
       'LAM': 'AM', 'RM': 'WM', 'LM': 'WM', 'RW': 'W', 'LW': 'W', 'RCF': 'CF', 'CF': 'CF', 'LCF': 'CF', 'SS': 'CF'}
FAM_ZH = {'FB': '边后卫', 'CB': '中卫', 'DM': '后腰', 'CM': '中前卫', 'AM': '前腰', 'WM': '边前卫', 'W': '边锋', 'CF': '中锋'}
# what the events show for the three unexplained team-matches (checked by hand against the event data)
EXPLAIN = {
    70220: 'StatsBomb 阵容数据错误：事件里阿比达尔在下半场开始（45:00）被阿德里亚诺换下，但阵容中他的出场时段一直到终场，'
           '多算约 46 分钟（占他本季约 2%）。',
    3825873: '真实的 10 人时段：切里舍夫 81:06 受伤离场（三个换人名额已用完），92:57 才回场；'
             '我们只少算了他回场后的约 2 分钟。',
    69222: '阵容与事件的换人时间相差约 30 秒，造成 13 人重叠半分钟，可忽略。',
}


def minutes_section():
    dc = json.load(open(R / 'data_check.json'))['minutes']
    apps = load_table('appearances')
    cases = []
    for x in dc['deviations']:
        if x['ok_explained']:
            continue
        mid, team = x['match_id'], x['team']
        evs = [f"{e['period']}-{e['minute']}:{e['second']:02d} {e['type']['name']} {e['player']['name']}"
               for e in load_events(mid) if e['team']['name'] == team
               and e['type']['name'] in ('Substitution', 'Player Off', 'Player On')]
        cases.append({'match_id': mid, 'team': team, 'deviation_min': x['dev'], 'max_on_pitch': x['max_on_pitch'],
                      'min_on_pitch': x['min_on_pitch'], 'events': evs, 'explanation': EXPLAIN.get(mid, '')})
    return {'team_matches': dc['team_matches'], 'within_1_min': dc['within_1_min'],
            'ok_or_explained': dc['ok_or_explained_by_treatment'], 'unexplained_cases': cases}


def external_section():
    ext = yaml.safe_load(open(ROOT / 'docs' / 'minutes_external_check.yaml'))
    pairs = [{'match_id': m, 'off': a, 'on': b, 'ours': o, 'source': s, 'agree': abs(o - s) <= 2}
             for m, a, b, o, s in ext['pairs']]
    return {'n_pairs': len(pairs), 'n_agree': sum(p['agree'] for p in pairs), 'pairs': pairs,
            'sources': ext['sources'], 'notes': ext['notes']}


def positions_section():
    dc = json.load(open(R / 'data_check.json'))['group_share_below_70']
    pm, pp = load_table('position_minutes'), load_table('player_period')
    core = {}
    for f in ('v2_barca.json', 'v2_league.json'):
        for u in json.load(open(R / f))['units']:
            for L in ('att', 'prog', 'def'):
                x = u['layers'][L]
                for n in set(x['core_set']) | {x['top']}:
                    core.setdefault((u['team'], u['label'], n), []).append(L + ('（头号）' if n == x['top'] else ''))
    for f in ('net_barca.json', 'net_league.json'):
        for u in json.load(open(R / f))['units']:
            if u.get('mode', 'net') == 'net':
                for n in set(u['core_set']) | {u['top']}:
                    core.setdefault((u['team'], u['label'], n), []).append('link' + ('（头号）' if n == u['top'] else ''))
    rows, pairs = [], Counter()
    for x in dc['list']:
        pid = pp[(pp.team == x['team']) & (pp.label == x['season']) & (pp.name == x['name'])].player_id
        if pid.empty:
            continue
        q = pm[(pm.team == x['team']) & (pm.player_id == int(pid.iloc[0])) & pm.match_id.isin(label_matches(x['team'], x['season']))]
        fam = Counter()
        for c, v in q[[c for c in q.columns if c.startswith('pmin_')]].sum().items():
            fam[FAM[c[5:]]] += v
        tot = sum(fam.values())
        top = [f for f, v in fam.most_common(2) if v > 0]
        pair = ' / '.join(FAM_ZH.get(f, f) for f in sorted(top))
        pairs[pair] += 1
        rows.append({'team': x['team'], 'season': x['season'], 'name': x['name'], 'group': x['group'],
                     'group_share': x['group_share'], 'minutes': x['minutes'], 'split': pair,
                     'split_shares': {FAM_ZH.get(f, f): fam[f] / tot for f in top},
                     'in_cores': core.get((x['team'], x['season'], x['name']), [])})
    return {'n_player_seasons': dc['n_player_seasons'], 'n_below': len(rows), 'splits': pairs.most_common(),
            'n_in_cores': sum(bool(r['in_cores']) for r in rows), 'rows': rows}


def main():
    out = {'minutes': minutes_section(), 'external': external_section(), 'positions': positions_section()}
    json.dump(out, open(R / 'data_review.json', 'w'), ensure_ascii=False, indent=1)
    m, e, p = out['minutes'], out['external'], out['positions']
    L = ['# 数据核对（2026-11-08，由 scripts/data_review.py 生成）', '',
         '## 1. 上场分钟', '',
         f"{m['team_matches']} 个队-场中，{m['within_1_min']} 个严格在 1 分钟内（{m['within_1_min'] / m['team_matches']:.1%}），"
         f"{m['ok_or_explained']} 个达标或可由红牌、临时离场解释。其余 {len(m['unexplained_cases'])} 个逐一查了事件数据：", '']
    for c in m['unexplained_cases']:
        L.append(f"- 比赛 {c['match_id']}（{c['team']}），偏差 {c['deviation_min']:+.1f} 分钟：{c['explanation']}")
    L += ['', '## 2. 五场比赛对照外部比赛报告', '',
          f"32 次换人中 {e['n_agree']} 次与外部来源一致（同一对球员、分钟相差不超过 2）。", '',
          '| 比赛 | 换下 | 换上 | 我们 | 外部来源 | 一致 |', '|---|---|---|---|---|---|']
    for q in e['pairs']:
        L.append(f"| {q['match_id']} | {q['off']} | {q['on']} | {q['ours']}′ | {q['source']}′ | {'是' if q['agree'] else '否'} |")
    L += [''] + [f"- 说明（{k}）：{v}" for k, v in e['notes'].items()] + [''] + \
         [f"- 来源（{k}）：{v}" for k, v in e['sources'].items()]
    L += ['', '## 3. 主要位置不明确的球员-赛季', '',
          f"{p['n_player_seasons']} 个球员-赛季中有 {p['n_below']} 个主要位置组的分钟占比低于 70%，"
          f"其中 {p['n_in_cores']} 个出现在已报出的核心（头号或核心集合）里。按原始位置看，他们在哪两类之间切换：", '',
          '| 切换 | 人数 |', '|---|---|'] + [f'| {k} | {v} |' for k, v in p['splits']]
    L += ['', '出现在核心里的：', '', '| 球队 | 赛季 | 球员 | 归入 | 占比 | 切换 | 出现在 |', '|---|---|---|---|---|---|---|']
    for r in p['rows']:
        if r['in_cores']:
            L.append(f"| {r['team']} | {r['season']} | {r['name']} | {r['group']} | {r['group_share']:.0%} | "
                     f"{', '.join(f'{k} {v:.0%}' for k, v in r['split_shares'].items())} | {', '.join(r['in_cores'])} |")
    (R / 'data_review.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L[:12]))
    print('external', e['n_agree'], '/', e['n_pairs'], '| positions', p['n_below'], 'in cores', p['n_in_cores'])


if __name__ == '__main__':
    main()
