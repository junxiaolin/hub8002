"""Figures for the final report (2026-11-08), every number read from results/*.json:

  v3_fig1_four_cores   Barcelona: the top core of each kind per period (attack, progression, linking, defence)
  v3_fig2_partitions   description length of seven position partitions, per layer (relative to three groups)
  v3_fig3_absence      absence check: does the predicted value say what happens when a player is missing?
  v3_fig4_consensus    agreement with widely accepted judgements (Ballon d'Or, La Liga awards, top scorers)
"""
import json

import numpy as np
from matplotlib.patches import Rectangle

from hub8002 import ROOT
from hub8002.viz.style import GRID, INK, INK2, NEUTRAL, SERIES, SURFACE, plt, save, setup

R, F = ROOT / 'results', ROOT / 'figures'
setup()
ZH = {'Lionel Messi': '梅西', 'Xavi': '哈维', 'Sergio Busquets': '布斯克茨', 'Gerard Piqué': '皮克',
      'Carles Puyol': '普约尔', "Samuel Eto'o": '埃托奥', 'Luis Suárez': '苏亚雷斯', 'Jordi Alba': '阿尔巴',
      'Ivan Rakitić': '拉基蒂奇', 'Andrés Iniesta': '伊涅斯塔', 'Cesc Fàbregas': '法布雷加斯',
      'Javier Mascherano': '马斯切拉诺', 'Sergi Roberto': '塞尔吉·罗伯托', 'Dani Alves': '阿尔维斯',
      'Rafael Márquez': '马克斯', 'Yaya Touré': '亚亚·图雷'}
# colour follows the player (fixed order, never cycled); everyone else neutral
PCOL = {'Lionel Messi': SERIES[0], 'Xavi': SERIES[1], 'Sergio Busquets': SERIES[2], 'Gerard Piqué': SERIES[3],
        'Carles Puyol': SERIES[4], "Samuel Eto'o": SERIES[5], 'Luis Suárez': SERIES[6]}
DARK_FILL = {SERIES[0], SERIES[5], SERIES[6], SERIES[1]}


def load(n):
    return json.load(open(R / n))


def short(label):
    return '里杰卡尔德\n2004–08' if label.startswith('Rijkaard') else label[2:4] + '/' + label[5:7]


def fig1():
    v2 = {u['label']: u for u in load('v2_barca.json')['units']}
    net = {u['label']: u for u in load('net_barca.json')['units'] if u.get('mode', 'net') == 'net'}
    labels = list(v2)
    widths = [4 if l.startswith('Rijkaard') else 1 for l in labels]
    x0 = np.concatenate([[0], np.cumsum(widths)[:-1]])
    rows = [('进攻', lambda l: v2[l]['layers']['att']), ('推进', lambda l: v2[l]['layers']['prog']),
            ('串联', lambda l: net[l]), ('防守', lambda l: v2[l]['layers']['def'])]
    fig, ax = plt.subplots(figsize=(9.2, 3.1))
    for r, (name, get) in enumerate(rows):
        y = len(rows) - 1 - r
        for l, x, w in zip(labels, x0, widths):
            u = get(l)
            top, cs = u['top'], u['core_set']
            col = PCOL.get(top, NEUTRAL)
            clear = bool(cs)
            ax.add_patch(Rectangle((x + 0.04, y + 0.08), w - 0.08, 0.84, facecolor=col if clear else SURFACE,
                                   edgecolor=col, linewidth=1.4 if not clear else 0, hatch=None if clear else '////'))
            txt = ZH.get(top, top.split()[-1])
            if clear and len(cs) > 1:
                txt += f'\n+{len(cs) - 1}'
            tc = 'white' if (clear and col in DARK_FILL) else INK
            ax.text(x + w / 2, y + 0.5, txt, ha='center', va='center', fontsize=6.4 if w == 1 else 7.5, color=tc,
                    bbox=None if clear else dict(facecolor=SURFACE, edgecolor='none', pad=0.6))
        ax.text(-0.25, y + 0.5, name, ha='right', va='center', fontsize=8.5, color=INK)
    ax.set_xlim(-0.1, x0[-1] + widths[-1] + 0.1)
    ax.set_ylim(0, len(rows))
    ax.set_xticks(x0 + np.array(widths) / 2)
    ax.set_xticklabels([short(l) for l in labels], fontsize=6.8)
    ax.set_yticks([])
    ax.grid(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title('巴萨四类核心的交接（每格为该时期的头号；实心 = 报出明确核心，"+n" = 并列核心人数；斜线 = 无明确核心）',
                 loc='left', fontsize=8.5, color=INK)
    save(fig, F / 'v3_fig1_four_cores.pdf')


def fig2():
    ps = load('partition_selection.json')
    parts = ['one', 'three', 'three_w', 'four', 'six', 'seven', 'eight']
    names = ['不分', '三类', '三类*', '四类', '六类', '七类', '八类']
    panels = [('进攻', 'att'), ('推进', 'prog'), ('防守', 'def'), ('串联（网络）', 'net_src4')]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.7), sharex=True)
    x = np.arange(len(parts))
    for ax, (title, key) in zip(axes, panels):
        d = [ps['summary'][key]['delta_dl_vs_three'][p] for p in parts]
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.plot(x, d, color=SERIES[0], marker='o', markersize=4.5, linewidth=1.6, label='只换"和谁比"')
        if key == 'net_src4':
            d2 = [ps['summary']['net']['delta_dl_vs_three'][p] for p in parts]
            ax.plot(x, d2, color=SERIES[1], marker='s', markersize=4, linewidth=1.4, linestyle='--',
                    label='目的地分块也随之变细')
            ax.legend(loc='upper left', fontsize=6.3)
        ax.set_title(title, loc='left', fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=6.5)
    axes[0].set_ylabel('描述长度变化（nats，相对三类；越低越好）', fontsize=7)
    fig.suptitle('位置划分的模型选择：数据偏好更细的"比较对象"划分，但网络的接球分块不宜过细（三类* = 边锋归中场）',
                 x=0.01, ha='left', fontsize=8.5, y=1.03)
    fig.tight_layout()
    save(fig, F / 'v3_fig2_partitions.pdf')


def fig3():
    ab = load('absence.json')['cross_fitted']
    preds = [('voa_average', '本方案（平均基准）'), ('voa_replacement', '替补基准'), ('voa_four_groups', '四类位置'),
             ('voa_eight_roles', '八类位置'), ('raw_per90', '每 90 分钟原始价值'), ('passes_per90', '每 90 分钟传球数'),
             ('minutes_share', '出场时间占比')]
    panels = [('进攻', 'att'), ('推进', 'prog'), ('串联', 'link'), ('防守', 'def')]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6), sharey=True, sharex=True)
    y = np.arange(len(preds))[::-1]
    for ax, (title, L) in zip(axes, panels):
        ax.axvline(0, color=INK2, linewidth=0.8)
        for yy, (k, nm) in zip(y, preds):
            v = ab['layers'][L].get(k)
            if not v:
                continue
            sig = v['p_value'] < 0.05
            col = SERIES[0] if k.startswith('voa') else NEUTRAL
            ax.plot([v['corr']], [yy], marker='o', markersize=6, markerfacecolor=col if sig else SURFACE,
                    markeredgecolor=col, markeredgewidth=1.4, linestyle='none')
            if sig:
                ax.text(v['corr'] + 0.012, yy, f"{v['corr']:.2f}（p={v['p_value']:.3f}）", va='center', fontsize=6, color=INK)
        ax.set_title(title, loc='left', fontsize=8.5)
        ax.set_xlim(-0.08, 0.2)
        ax.set_xlabel('加权相关', fontsize=7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([nm for _, nm in preds], fontsize=7)
    fig.suptitle(f"缺阵检验（2015/16，交叉拟合，{ab['n_player_units']} 个球员-半季）：只有进攻价值能预测缺阵时球队 xG 的下降"
                 '（实心 = 置换检验 p < 0.05）', x=0.01, ha='left', fontsize=8.5, y=1.04)
    fig.tight_layout()
    save(fig, F / 'v3_fig3_absence.pdf')


def fig4():
    cs = load('consensus.json')['summary']
    vers = [('average (main)', '主线（平均基准、三类）'), ('replacement level', '替补基准'),
            ('four groups', '四类位置'), ('eight roles', '八类位置')]
    refs = [('ballon_dor', '金球奖前三'), ('lfp', '西甲官方奖项'), ('top_scorer', '2015/16 各队射手王')]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    w = 0.19
    for i, (k, nm) in enumerate(vers):
        vals = [cs[r][k]['core'] * 100 for r, _ in refs]
        xs = np.arange(len(refs)) + (i - 1.5) * (w + 0.02)
        ax.bar(xs, vals, width=w, color=SERIES[i] if i < 3 else SERIES[6], label=nm, edgecolor=SURFACE, linewidth=1)
        for xx, vv in zip(xs, vals):
            ax.text(xx, vv + 1.5, f'{vv:.0f}', ha='center', fontsize=6.3, color=INK)
    ax.set_xticks(np.arange(len(refs)))
    ax.set_xticklabels([f"{nm}（{cs[r]['n']} 条）" for r, nm in refs], fontsize=7.5)
    ax.set_ylabel('获奖者 / 射手王在对应核心集合中（%）', fontsize=7)
    ax.set_ylim(0, 100)
    ax.legend(ncol=2, fontsize=6.5, loc='upper right')
    ax.set_title('与公认评价的一致程度', loc='left', fontsize=8.5)
    save(fig, F / 'v3_fig4_consensus.pdf')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    print('figures written')
