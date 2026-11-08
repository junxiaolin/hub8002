"""Tables for the 2026-11-08 improvements (results/tables_v3.md): replacement level, data-selected position
partition, absence check, consensus check. Every number is read from results/*.json."""
import json

from hub8002 import ROOT

R = ROOT / 'results'
L4 = ('att', 'prog', 'def', 'link')
NAME = {'att': '进攻', 'prog': '推进', 'def': '防守', 'link': '串联', 'net': '串联（网络，分块随划分）',
        'net_src': '串联（只换比较对象，目的地三类）', 'net_src4': '串联（只换比较对象，目的地四类）'}
T = []


def load(n):
    return json.load(open(R / n))


def table(title, header, rows):
    T.append(f'\n### {title}\n')
    T.append('| ' + ' | '.join(header) + ' |')
    T.append('|' + '---|' * len(header))
    for r in rows:
        T.append('| ' + ' | '.join(str(x) for x in r) + ' |')


def pct(x):
    return f'{100 * x:.0f}%'


def main():
    b = load('bench_rates.json')
    table('V1 替补水平（2015/16 首发不到一半的球员）相对同位置平均', ['层', '后卫', '中场', '前锋'],
          [[NAME[l]] + [f"{b['node']['bench50_info'][l][g]['ratio_to_average']:.2f}x（第 {100 * b['node']['bench50_info'][l][g]['gamma_quantile']:.0f} 百分位）"
                        for g in ('DEF', 'MID', 'FWD')] for l in ('att', 'prog', 'def')]
          + [['串联：每次触球丢球率'] + [f"{b['net']['bench50'][g]['END']:.3f}" for g in ('DEF', 'MID', 'FWD')]])

    s0, s1 = load('v2_sim.json'), load('v2_sim_bench50.json')
    rows = []
    for m in (5, 38):
        for l in ('att', 'prog', 'def'):
            a = next(r for r in s0['rows'] if r['layer'] == l and r['m'] == m and r['world'] == 'gamma')
            c = next(r for r in s1['rows'] if r['layer'] == l and r['m'] == m and r['world'] == 'gamma')
            rows.append([f'{NAME[l]} {m} 场', pct(a['accuracy']['ours']), pct(c['accuracy']['ours']), pct(c['accuracy']['raw_voa']),
                         pct(c['accuracy']['total']), pct(c['accuracy']['per90']), f"{c['mean_p_top']:.2f}", pct(c['coverage90'])])
    n0, n1 = load('net_sim.json')['summary'], load('net_sim_bench50.json')['summary']
    for m in (5, 38):
        a, c = n0[f'base|{m}'], n1[f'base|{m}']
        rows.append([f'串联 {m} 场', pct(a['hit']['ours']), pct(c['hit']['ours']), pct(c['hit']['plugin']),
                     pct(c['hit']['touches']), pct(c['hit']['pagerank']), f"{c['mean_p_top_pick']:.2f}", pct(c['cred_cover'])])
    table('V2 模拟：标准答案改为"超出替补水平最多"后的命中率',
          ['设定', '本方案（平均基准）', '本方案（替补基准）', '不收缩', '总价值 / 触球最多', '每 90 分钟 / PageRank',
           'P(#1) 均值', '90% 集合覆盖'], rows)

    rr = load('rep_real.json')['summary']
    table('V3 真实数据：替补基准与平均基准的头号核心是否相同', ['层', '相同（全部）', '相同（巴萨）', '有明确核心（替补 / 平均）'],
          [[NAME[l], pct(rr['bench50'][l]['top_same_as_average']), pct(rr['bench50'][l]['top_same_barca']),
            f"{rr['bench50'][l]['n_with_core']} / {rr['bench50'][l]['n_with_core_average']}（共 {rr['bench50'][l]['n_units']}）"] for l in L4])

    ps = load('partition_selection.json')
    parts = list(ps['partitions'])
    rows = []
    for l in ('att', 'prog', 'def', 'net', 'net_src', 'net_src4'):
        s = ps['summary'][l]
        rows.append([NAME[l]] + [f"{s['delta_dl_vs_three'][p]:+.0f}" for p in parts] + [s['best_by_dl'], s['best_by_loto_cv']])
    o = ps['summary']['overall']
    rows.append(['合计（进攻+推进+防守+串联四类目的地）'] + [f"{o['total_dl_vs_three'][p]:+.0f}" for p in parts]
                + [o['best_by_dl'], o['best_by_loto_cv']])
    table('V4 位置划分的描述长度（相对三类位置，单位 nats，越低越好）', ['层'] + parts + ['描述长度最优', '留一队交叉验证最优'], rows)
    T.append('\n划分：' + '；'.join(f"{p} = {' / '.join(g for g in gs if g != 'GK')}" for p, gs in ps['partitions'].items()))

    pr = load('partition_real.json')
    table('V5 真实数据：按其他划分重算后，头号核心与三类位置相同的比例', ['划分', '进攻', '推进', '防守', '串联'],
          [[p] + [pct(pr['summary'][p][l]['top_same_as_three']) for l in L4] for p in ('eight', 'seven', 'four')])
    rows = []
    for k, v in pr['fits']['eight'].items():
        if k.startswith('Barcelona'):
            f4 = pr['fits']['four'][k]['layers']
            rows.append([k.split('|')[1]] + [v['layers'][l]['top'] for l in L4] + [f4['def']['top'], f4['link']['top']])
    table('V6 巴萨头号核心：八类位置（及四类位置下的防守、串联）', ['时期', '进攻', '推进', '防守', '串联', '防守（四类）', '串联（四类）'], rows)

    ab = load('absence.json')
    names = {'voa_average': '本方案（平均基准，三类）', 'voa_replacement': '替补基准', 'voa_eight_roles': '八类位置',
             'voa_four_groups': '四类位置', 'raw_per90': '每 90 分钟原始价值', 'passes_per90': '每 90 分钟传球数',
             'minutes_share': '出场时间占比'}
    for key, title in (('cross_fitted', '交叉拟合（主结果）'), ('in_sample', '同一批比赛（会虚高）')):
        rows = []
        for pn, nm in names.items():
            row = [nm]
            for l in L4:
                v = ab[key]['layers'][l].get(pn)
                row.append(f"{v['corr']:.2f}（p={v['p_value']:.3f}）" if v else '—')
            rows.append(row)
        table(f"V7 缺阵检验，{title}：预测价值与缺阵时球队 xG 变化的加权相关（{ab[key]['n_player_units']} 个球员-半季）"
              if key == 'cross_fitted' else f"V7b 缺阵检验，{title}（{ab[key]['n_player_units']} 名球员）",
              ['预测指标'] + [NAME[l] for l in L4], rows)

    cs = load('consensus.json')
    vers = [v for v in cs['summary']['ballon_dor'] if v != 'n']
    rows = []
    for kind, nm in (('ballon_dor', '金球奖前三'), ('lfp', 'LFP / 西甲奖项'), ('top_scorer', '2015/16 各队射手王')):
        s = cs['summary'][kind]
        rows.append([f"{nm}（{s['n']} 条）"] + [f"{pct(s[v]['top'])} / {pct(s[v]['core'])} / {pct(s[v]['top3'])}" for v in vers])
    table('V8 公认评价对照：是头号 / 在核心集合 / 进前三（对应类别）', ['参照'] + vers, rows)

    (R / 'tables_v3.md').write_text('# 改进实验表格（2026-11-08，由 scripts/make_tables_v3.py 生成）\n' + '\n'.join(T) + '\n')
    print('\n'.join(T))


if __name__ == '__main__':
    main()
