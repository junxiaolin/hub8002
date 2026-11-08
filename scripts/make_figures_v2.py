"""v2 figures (figures/v2_*.pdf|png) and tables (results/tables_v2.md)."""
import json

import numpy as np

from hub8002 import ROOT
from hub8002.viz.style import NEUTRAL, SEQ, SERIES, plt, save, setup

R = ROOT / 'results'
F = ROOT / 'figures'
setup()
DIM = {'att': 'attacking core', 'prog': 'organising core', 'def': 'defensive core', 'org': 'pass volume (contrast)'}
DIM_ZH = {'att': '进攻', 'prog': '组织', 'def': '防守', 'org': '传球量（对照）'}
T = []


def load(n):
    p = R / n
    return json.load(open(p)) if p.exists() else None


def short(name):
    return name.replace('Rijkaard 2004-08', 'Rijk. 04-08')


def fig_heatmap(b):
    units = b['units']
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.4), gridspec_kw={'hspace': 0.5})
    for ax, layer in zip(axes, ('att', 'prog', 'def')):
        probs = {}
        for ui, u in enumerate(units):
            for p in u['layers'].get(layer, {}).get('players', []):
                probs.setdefault(p['name'], {})[ui] = p['p_top']
        names = sorted(probs, key=lambda n: -max(probs[n].values()))[:9]
        names.sort(key=lambda n: min(probs[n]))
        M = np.full((len(names), len(units)), np.nan)
        for i, n in enumerate(names):
            for ui, pr in probs[n].items():
                M[i, ui] = pr
        im = ax.imshow(np.nan_to_num(M, nan=0.0), aspect='auto', cmap=SEQ, vmin=0, vmax=1, interpolation='nearest')
        for i in range(len(names)):
            for j in range(len(units)):
                if np.isnan(M[i, j]):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='#f0efec', lw=0))
                elif names[i] in units[j]['layers'][layer]['core_set']:
                    ax.text(j, i, '●', ha='center', va='center', fontsize=5,
                            color='white' if M[i, j] > 0.55 else '#0b0b0b')
        ax.set_yticks(range(len(names)), names)
        ax.set_xticks(range(len(units)), [short(u['label']) for u in units], rotation=60, ha='right')
        ax.grid(False)
        ax.set_title(f'({"abc"[("att", "prog", "def").index(layer)]}) {DIM[layer]}: probability of being the most valuable player (● = in the core set)',
                     loc='left', fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label('P(#1)')
    save(fig, F / 'v2_fig1_barca_cores.pdf')


def fig_voa_season(b, season='2011/12'):
    u = next(u for u in b['units'] if u['label'] == season)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.2), gridspec_kw={'wspace': 0.75})
    gcol = {'DEF': SERIES[2], 'MID': SERIES[0], 'FWD': SERIES[1]}
    for ax, layer in zip(axes, ('att', 'prog', 'def')):
        ps = sorted(u['layers'][layer]['players'], key=lambda p: p['voa'])[-10:]
        y = np.arange(len(ps))
        ax.barh(y, [p['voa'] for p in ps], height=0.65, color=[gcol[p['group']] for p in ps])
        ax.errorbar([p['voa'] for p in ps], y, xerr=[[p['voa'] - p['voa_lo'] for p in ps], [p['voa_hi'] - p['voa'] for p in ps]],
                    fmt='none', ecolor='#52514e', elinewidth=0.8, capsize=1.5)
        ax.axvline(0, color=NEUTRAL, lw=0.8)
        ax.set_yticks(y, [p['name'].split()[-1] for p in ps])
        ax.set_title(DIM[layer], loc='left', fontsize=8)
        ax.set_xlabel('value over an average player\nof his position (xG)')
        ax.grid(axis='y', visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in gcol.values()]
    axes[2].legend(handles, list(gcol), loc='lower right', fontsize=6.5)
    fig.suptitle(f'Barcelona {season}: posterior mean and 90% interval', x=0.01, ha='left', fontsize=9)
    save(fig, F / 'v2_fig2_barca_voa_2011_12.pdf')


def fig_sim(sim):
    rows = [r for r in sim['rows'] if r['world'] == 'gamma']
    ms = sorted({r['m'] for r in rows})
    names = {'ours': 'Our rule', 'raw_voa': 'Value over average, no shrinkage', 'total': 'Most total value',
             'per90': 'Most value per 90 (no position)', 'ratio': 'Largest ratio to position average'}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for ax, layer in zip(axes, ('att', 'prog', 'def')):
        for i, mth in enumerate(names):
            y = [next(r['accuracy'][mth] for r in rows if r['layer'] == layer and r['m'] == m) for m in ms]
            ax.plot(ms, y, marker='o', color=SERIES[i], label=names[mth], markersize=4)
        ax.set_xscale('log')
        ax.set_xticks(ms, [str(m) for m in ms])
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.set_xlabel('matches observed')
        ax.set_title(DIM[layer], loc='left', fontsize=8)
    axes[0].set_ylabel('picks the truly most\nvaluable player')
    axes[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=6.5)
    save(fig, F / 'v2_fig3_sim_accuracy.pdf')


def fig_seq(seq):
    picks = [('att', 'Lionel Messi', ('2010/11', '2011/12', '2015/16', '2018/19')),
             ('prog', 'Xavi', ('2008/09', '2009/10', '2010/11', '2011/12')),
             ('def', 'Gerard Piqué', ('2010/11', '2013/14', '2015/16', '2019/20'))]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ax, (layer, who, seasons) in zip(axes, picks):
        i = 0
        for u in seq['units']:
            if u['team'] != 'Barcelona' or u['season'] not in seasons or layer not in u['layers']:
                continue
            pid = next((k for k, v in u['names'].items() if v == who), None)
            tr = u['layers'][layer]['trajectories'].get(str(pid))
            if not tr:
                continue
            xs = sorted(int(x) for x in tr)
            ax.plot(xs, [tr[str(x)] for x in xs], color=SERIES[i], label=u['season'], lw=1.4)
            i += 1
        ax.axhline(0.5, color=NEUTRAL, ls='--', lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlabel('matches into the season')
        ax.set_title(f'{who}, {DIM[layer]}', loc='left', fontsize=8)
        ax.legend(loc='lower right', fontsize=6)
    axes[0].set_ylabel('P(most valuable in team)')
    save(fig, F / 'v2_fig4_sequential.pdf')


def tables(b, lg, sim, half, seq, sens):
    L = ['## 表 A：巴萨各时期的核心（核心集合；括号内为头号核心成为全队第一的概率）', '',
         '| 时期 | 进攻核心 | 组织核心 | 防守核心 | 传球量最多（对照） | v1 传球网络：PageRank 第一 |', '|---|---|---|---|---|---|']
    for u in b['units']:
        c = []
        for layer in ('att', 'prog', 'def'):
            Lr = u['layers'][layer]
            c.append((', '.join(Lr['core_set']) or '无明确核心') + f" ({Lr['p_top_max']:.2f})")
        o = u['layers']['org']
        c.append(', '.join(o.get('v1_pass_hubs', {}).get('count_top', [])))
        c.append(', '.join(o.get('v1_pass_hubs', {}).get('pagerank_top', [])))
        L.append(f"| {u['label']} | " + ' | '.join(c) + ' |')
    T.append('\n'.join(L))
    L = ['## 表 B：西甲 2015/16 各队的核心', '', '| 球队 | 进攻核心 | 组织核心 | 防守核心 | 传球量最多（对照） |', '|---|---|---|---|---|']
    for u in lg['units']:
        c = [', '.join(u['layers'][l]['core_set']) or '无明确核心' for l in ('att', 'prog', 'def')]
        c.append(', '.join(u['layers']['org'].get('v1_pass_hubs', {}).get('count_top', [])))
        L.append(f"| {u['team']} | " + ' | '.join(c) + ' |')
    T.append('\n'.join(L))
    L = ['## 表 C：模拟检验（真实名单与上场时间；"gamma" 世界与模型假设一致，"lognormal" 世界故意不一致）', '',
         '| 维度 | 场数 | 世界 | 本规则选中真正头号 | 不做收缩 | 总价值最多 | 每90分钟最多（不分位置） | 与位置均值之比最大 | 本规则平均给出的把握 | 90%核心集合覆盖率 | 平均集合人数 | "明确核心"的正确率 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sim['rows']:
        a = r['accuracy']
        cp = '—' if r['clear_precision'] is None else f"{r['clear_precision']:.2f}"
        L.append(f"| {DIM_ZH[r['layer']]} | {r['m']} | {r['world']} | {a['ours']:.2f} | {a['raw_voa']:.2f} | {a['total']:.2f} | "
                 f"{a['per90']:.2f} | {a['ratio']:.2f} | {r['mean_p_top']:.2f} | {r['coverage90']:.2f} | {r['mean_set_size']:.1f} | {cp} |")
    T.append('\n'.join(L))
    L = ['## 表 D：前半季预测后半季（34 个队-赛季）', '',
         '| 维度 | 本模型的预测改进（相对位置均值） | 原始每90分钟数据的预测改进 | 本模型更准的队-赛季比例 | 前后半季头号核心相同 | 核心集合重合度 |', '|---|---|---|---|---|---|']
    for layer, s in half['summary'].items():
        sk = s['skill_vs_position_average']
        L.append(f"| {DIM_ZH[layer]} | {sk['ours']:.0%} | {sk['raw_per90']:.0%} | {s['share_units_ours_beats_raw']:.0%} | "
                 f"{s['top_core_same_both_halves']:.0%} | {s['core_set_jaccard']:.2f} |")
    T.append('\n'.join(L))
    L = ['## 表 E：逐场确认（36 个队-赛季）：赛季末的头号核心从第几场起一直保持', '',
         '| 维度 | 一直 > 0.5 的队-赛季数 | 中位数（占赛季比例） | 一直 > 0.9 的队-赛季数 | 中位数（占赛季比例） |', '|---|---|---|---|---|']
    for layer, s in seq['summary'].items():
        L.append(f"| {DIM_ZH[layer]} | {s['n_confirm_0.5']} | {s['median_share_0.5']:.2f} | {s['n_confirm_0.9']} | "
                 f"{s['median_share_0.9'] if s['median_share_0.9'] is None else round(s['median_share_0.9'], 2)} |")
    T.append('\n'.join(L))
    L = ['## 表 F：敏感性（与基准相比，头号核心 / 核心集合不变的比例）', '',
         '| 设定 | 进攻 头号 | 进攻 集合 | 组织 头号 | 组织 集合 | 防守 头号 | 防守 集合 |', '|---|---|---|---|---|---|---|']
    for s in sens['summary']:
        L.append(f"| {s['variant']} | " + ' | '.join(f"{s[f'{l}_top_same']:.0%} | {s[f'{l}_set_same']:.0%}" for l in ('att', 'prog', 'def')) + ' |')
    T.append('\n'.join(L))


def main():
    b, lg, sim, half, seq, sens = (load(n) for n in ('v2_barca.json', 'v2_league.json', 'v2_sim.json', 'v2_half.json',
                                                     'v2_sequential.json', 'v2_sensitivity.json'))
    fig_heatmap(b)
    fig_voa_season(b)
    fig_sim(sim)
    fig_seq(seq)
    tables(b, lg, sim, half, seq, sens)
    open(R / 'tables_v2.md', 'w').write('\n\n'.join(T) + '\n')
    print(sorted(p.name for p in F.glob('v2_*.pdf')))


if __name__ == '__main__':
    main()
