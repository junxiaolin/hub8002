"""Network extension: figures (figures/net_*.pdf|png) and tables (results/tables_net.md)."""
import json

import numpy as np
from scipy.stats import spearmanr

from hub8002 import ROOT
from hub8002.viz.style import INK2, NEUTRAL, SEQ, SERIES, plt, save, setup

R = ROOT / 'results'
F = ROOT / 'figures'
setup()
T = []


def load(n):
    return json.load(open(R / n))


def short(label):
    return label.replace('Rijkaard 2004-08', 'Rijk. 04-08')


def pct(x):
    return f'{100 * x:.0f}%'


def core_txt(core, top, p):
    return f"{', '.join(core) if core else '无明确核心'}（头号 {top} {p:.2f}）" if not core else f"{', '.join(core)} ({p:.2f})"


# ------------------------------------------------------------------ figures
def heat(ax, units, get_players, get_core, title):
    probs = {}
    for ui, u in enumerate(units):
        for p in get_players(u):
            probs.setdefault(p['name'], {})[ui] = p['p_top']
    names = sorted(probs, key=lambda n: -max(probs[n].values()))[:8]
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
            elif names[i] in get_core(units[j]):
                ax.text(j, i, '●', ha='center', va='center', fontsize=5, color='white' if M[i, j] > 0.55 else '#0b0b0b')
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks(range(len(units)), [short(u['label']) for u in units], rotation=60, ha='right')
    ax.grid(False)
    ax.set_title(title, loc='left', fontsize=8)
    return im


def fig_heat(nb, vb):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.8), gridspec_kw={'hspace': 0.6})
    heat(axes[0], vb['units'], lambda u: u['layers']['prog']['players'], lambda u: u['layers']['prog']['core_set'],
         '(a) organising core by xT added (v2): P(most valuable) (● = core set)')
    im = heat(axes[1], nb['units'], lambda u: u['players'], lambda u: u['core_set'],
              '(b) linking core on the possession network: P(most valuable) (● = core set)')
    for ax in axes:
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label('P(#1)')
    save(fig, F / 'net_fig1_barca_linking.pdf')


def fig_decomp(nb):
    hi = {'Xavi': SERIES[0], 'Sergio Busquets': SERIES[1], 'Lionel Messi': SERIES[2], 'Dani Alves': SERIES[3],
          'Jordi Alba': SERIES[4]}
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    rest = [(p['retain'], p['choice'], p['touches']) for u in nb['units'] if u['label'] != 'Rijkaard 2004-08'
            for p in u['players'] if p['touches'] >= 800 and p['name'] not in hi]
    ax.scatter([1000 * r[0] for r in rest], [1000 * r[1] for r in rest], s=[r[2] / 120 for r in rest], color=NEUTRAL,
               alpha=0.55, lw=0, label='other Barça players (≥ 800 touches)')
    for name, col in hi.items():
        pts = [(p['retain'], p['choice'], p['touches']) for u in nb['units'] if u['label'] != 'Rijkaard 2004-08'
               for p in u['players'] if p['name'] == name and p['touches'] >= 800]
        ax.scatter([1000 * r[0] for r in pts], [1000 * r[1] for r in pts], s=[r[2] / 120 for r in pts], color=col,
                   lw=0.8, edgecolor='#fcfcfb', label=name)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel('keeping the ball: expected goals added per 1,000 touches')
    ax.set_ylabel('choosing the receiver: expected goals\nadded per 1,000 touches')
    ax.set_title('Where the linking value comes from (one dot = player-season, size = touches)', loc='left', fontsize=8)
    lg = ax.legend(loc='upper right')
    for h in lg.legend_handles:
        h.set_sizes([22])
    save(fig, F / 'net_fig2_decomposition.pdf')


def fig_sim(sim):
    s = sim['summary']
    ms = [5, 10, 19, 38]
    lab = {'ours': 'ours (network VOA, posterior)', 'plugin': 'network VOA without shrinkage', 'retention': 'best ball retention',
           'touches': 'most touches', 'pagerank': 'PageRank (pass network)'}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={'wspace': 0.3}, sharey=True)
    for ax, world in zip(axes, ('base', 'overdispersed')):
        for k, col in zip(lab, SERIES):
            ax.plot(ms, [s[f'{world}|{m}']['hit'][k] for m in ms], marker='o', color=col, label=lab[k])
        ax.plot(ms, [s[f'{world}|{m}']['mean_p_top_pick'] for m in ms], ls='--', color=SERIES[0], lw=1.0,
                label='ours: stated P(#1)')
        ax.set_xticks(ms)
        ax.set_xlabel('matches observed')
        ax.set_ylim(0, 0.8)
        ax.set_title({'base': '(a) data as over-dispersed as the real data',
                      'overdispersed': '(b) twice as over-dispersed as the model assumes'}[world], loc='left', fontsize=8)
    axes[0].set_ylabel('share picking the true #1 linker')
    axes[1].legend(loc='lower right', fontsize=6)
    save(fig, F / 'net_fig3_sim.pdf')


# ------------------------------------------------------------------ tables
def rank_of(players, name, key='voa'):
    order = [p['name'] for p in sorted(players, key=lambda p: -p[key])]
    return order.index(name) + 1 if name in order else None


def table_barca(nb, vb):
    L = ['## 表 N1：巴萨各时期的串联核心与对照', '',
         '核心集合后括号内为头号核心成为全队第一的概率。"组织核心（xT）"为 v2 的结果；"触球最多""传球最多"是简单计数。最后一列是布斯克茨在两种标准下的队内排名。', '',
         '| 时期 | 串联核心（网络） | 组织核心（xT，v2） | 触球最多 | 传球最多 | 布斯克茨排名：串联 / xT |', '|---|---|---|---|---|---|']
    vmap = {u['label']: u for u in vb['units']}
    for u in nb['units']:
        v = vmap[u['label']]['layers']
        bq = rank_of(u['players'], 'Sergio Busquets')
        bq2 = rank_of(v['prog']['players'], 'Sergio Busquets')
        L.append(f"| {u['label']} | {', '.join(u['core_set']) or '无明确核心'} ({u['p_top_max']:.2f}; 头号 {u['top']}) | "
                 f"{', '.join(v['prog']['core_set']) or '无明确核心'} | {u['touches_top']} | "
                 f"{', '.join(v['org'].get('v1_pass_hubs', {}).get('count_top', []))} | "
                 f"{bq if bq else '—'} / {bq2 if bq2 else '—'} |")
    T.append('\n'.join(L))


def table_league(nl, vl):
    L = ['## 表 N2：西甲 2015/16 各队的串联核心', '', '| 球队 | 串联核心 | 头号（P） | 组织核心（xT，v2） | 传球最多 |', '|---|---|---|---|---|']
    vmap = {u['team']: u for u in vl['units']}
    for u in nl['units']:
        v = vmap[u['team']]['layers']
        L.append(f"| {u['team']} | {', '.join(u['core_set']) or '无明确核心'} | {u['top']} ({u['p_top_max']:.2f}) | "
                 f"{', '.join(v['prog']['core_set']) or '无明确核心'} | {', '.join(v['org'].get('v1_pass_hubs', {}).get('count_top', []))} |")
    T.append('\n'.join(L))


def table_decomp(nb):
    L = ['## 表 N3：串联价值的来源（巴萨部分球员-赛季）', '',
         '每千次触球比同位置普通球员多创造的预期进球，拆成"不丢球"和"传给谁"两部分；VOA 为整个赛季的网络价值增量（预期进球）。', '',
         '| 球员 | 赛季 | 触球 | 网络 VOA | 不丢球 / 千次触球 | 传给谁 / 千次触球 | 丢球率 |', '|---|---|---|---|---|---|---|']
    pick = {'Xavi': ('2008/09', '2010/11', '2013/14'), 'Sergio Busquets': ('2010/11', '2015/16', '2020/21'),
            'Lionel Messi': ('2010/11', '2014/15'), 'Dani Alves': ('2010/11', '2015/16'), 'Jordi Alba': ('2018/19',),
            'Gerard Piqué': ('2015/16', '2019/20'), 'Ivan Rakitić': ('2017/18',), 'Andrés Iniesta': ('2008/09',)}
    for u in nb['units']:
        for p in u['players']:
            if u['label'] in pick.get(p['name'], ()):
                L.append(f"| {p['name']} | {u['label']} | {p['touches']:.0f} | {p['voa']:.2f} | {1000 * p['retain']:.2f} | "
                         f"{1000 * p['choice']:.2f} | {p['end_rate']:.3f} |")
    T.append('\n'.join(L))


def table_relations(nb, vb):
    """How the linking value relates to touches and to xT added (Barça seasons, players >= 900 min)."""
    vmap = {u['label']: u for u in vb['units']}
    r_t, r_x = [], []
    for u in nb['units']:
        ps = [p for p in u['players'] if p['minutes'] >= 900]
        prog = {p['name']: p['voa'] for p in vmap[u['label']]['layers']['prog']['players']}
        net = [p['voa'] for p in ps]
        r_t.append(spearmanr(net, [p['touches'] for p in ps])[0])
        both = [(p['voa'], prog[p['name']]) for p in ps if p['name'] in prog]
        r_x.append(spearmanr([b[0] for b in both], [b[1] for b in both])[0])
    same_touch = np.mean([u['top'] == u['touches_top'] for u in nb['units']])
    T.append('\n'.join(['## 表 N4：串联价值与触球数、xT 价值的关系（巴萨各赛季，上场 ≥ 900 分钟的球员）', '',
                        '| 指标 | 数值 |', '|---|---|',
                        f'| 队内秩相关：网络 VOA vs 触球数（各赛季平均） | {np.mean(r_t):.2f} |',
                        f'| 队内秩相关：网络 VOA vs v2 组织（xT）VOA（各赛季平均） | {np.mean(r_x):.2f} |',
                        f'| 头号串联核心 = 触球最多的人 | {pct(same_touch)}（{len(nb["units"])} 个时期） |']))


def table_sim(sim):
    s = sim['summary']
    L = ['## 表 N5：模拟检验（2015/16 西甲 20 队的真实名单与传球结构，每队每种设定 100 次）', '',
         '"基准世界"的逐场过度离散与真实数据相同（φ ≈ 2.1）；"过度离散世界"的 φ ≈ 4，是模型假设的两倍。', '',
         '| 世界 | 场数 | 本模型选中真正头号 | 不做收缩 | 不丢球最好 | 触球最多 | PageRank | 本模型平均给出的 P(#1) | 90% 集合覆盖率 | "明确核心"占比 | "明确核心"正确率 | 90% 区间覆盖率 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for world in ('base', 'overdispersed'):
        for m in (5, 10, 19, 38):
            x = s[f'{world}|{m}']
            h = x['hit']
            cc = f"{x['clear_correct']:.2f}" if x['clear_correct'] is not None else '—'
            L.append(f"| {'基准' if world == 'base' else '过度离散'} | {m} | {h['ours']:.2f} | {h['plugin']:.2f} | {h['retention']:.2f} | "
                     f"{h['touches']:.2f} | {h['pagerank']:.2f} | {x['mean_p_top_pick']:.2f} | {x['cred_cover']:.2f} | "
                     f"{pct(x['clear_share'])} | {cc} | {x['interval_cover_90']:.2f} |")
    T.append('\n'.join(L))


def table_half(h):
    s = h['summary']
    g = s['log_score_gain_per_touch_vs_position_average']
    T.append('\n'.join(['## 表 N6：前半季预测后半季', '', '| 指标 | 本模型 | 前半季原始比例 | 触球数 |', '|---|---|---|---|',
                        f"| 预测后半季持球去向：比位置平均多出的对数得分（每次触球） | {g['ours']:.4f} | {g['raw']:.4f} | — |",
                        f"| 本模型优于原始比例的队-赛季占比 | {pct(s['share_units_ours_beats_raw'])} | — | — |",
                        f"| 前半季 VOA 与后半季 VOA 的秩相关 | {s['mean_spearman']['ours']:.2f} | {s['mean_spearman']['plugin']:.2f} | {s['mean_spearman']['touches']:.2f} |",
                        f"| 前后半季头号串联核心相同 | {pct(s['top_core_same_both_halves'])}（巴萨 {pct(s['top_core_same_barca'])}） | — | — |",
                        '', f"共 {s['n_units']} 个队-赛季。"]))


def table_sens(sv):
    L = ['## 表 N7：敏感性（巴萨各时期 + 西甲 2015/16 其余 19 队，共 33 个单位）', '',
         '| 设定 | 头号核心不变 | 其中巴萨 | 核心集合完全不变 | 有明确核心的单位数 |', '|---|---|---|---|---|']
    for k, v in sv['summary'].items():
        L.append(f"| {k} | {pct(v['top_same'])} | {pct(v['top_same_barca'])} | {pct(v['core_set_same'])} | {v['n_with_core']} |")
    T.append('\n'.join(L))


def main():
    nb, nl = load('net_barca.json'), load('net_league.json')
    vb, vl = load('v2_barca.json'), load('v2_league.json')
    sim, half, sv = load('net_sim.json'), load('net_half.json'), load('net_sensitivity.json')
    fig_heat(nb, vb)
    fig_decomp(nb)
    fig_sim(sim)
    table_barca(nb, vb)
    table_league(nl, vl)
    table_decomp(nb)
    table_relations(nb, vb)
    table_sim(sim)
    table_half(half)
    table_sens(sv)
    (R / 'tables_net.md').write_text('# 网络扩展（串联核心）结果表\n\n由 `make net` 产出。\n\n' + '\n\n'.join(T) + '\n')
    print(sorted(p.name for p in F.glob('net_*.pdf')))


if __name__ == '__main__':
    main()
