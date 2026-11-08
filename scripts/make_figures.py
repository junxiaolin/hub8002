"""Figures 1, 3-8 and the result tables (figures/*.pdf + .png, results/tables.md) from results/*.json."""
import json

import numpy as np

from hub8002 import ROOT
from hub8002.viz.style import INK2, NEUTRAL, SEQ, SERIES, plt, save, setup

R = ROOT / 'results'
F = ROOT / 'figures'
F.mkdir(exist_ok=True)
setup()


def load(name):
    p = R / name
    return json.load(open(p)) if p.exists() else None


NAME = {'bayes_nb': 'Our model', 'bayes_p': 'Plan Poisson model', 'nb_no_position': 'No position',
        'nb_no_minutes': 'No minutes', 'nb_no_shared_rho': 'No shared rho', 'mdl_cm': 'MDL-CM',
        'mdl_er': 'MDL-ER', 'mdl_cm_min': 'MDL-CM (per-minute)', 'mdl_er_min': 'MDL-ER (per-minute)',
        'count_top': 'Most passes', 'per90_top': 'Most per 90', 'pagerank_top': 'Top PageRank',
        'mean2sd': 'Mean + 2SD', 'average': 'Average', 'loubar': 'Loubar'}
# fixed colour per method (colour follows the entity)
COL = {'bayes_nb': SERIES[0], 'bayes_p': SERIES[1], 'mdl_cm': SERIES[2], 'mdl_cm_min': SERIES[3],
       'mean2sd': SERIES[4], 'count_top': SERIES[5], 'per90_top': SERIES[6], 'pagerank_top': SERIES[7]}
MARK = {'bayes_nb': 'o', 'bayes_p': 's', 'mdl_cm': '^', 'mdl_cm_min': 'v', 'mean2sd': 'D', 'count_top': 'P',
        'per90_top': 'X', 'pagerank_top': '*'}
TABLES = []


def fig3_e1_false_positives(e1):
    rows = e1['rows']
    methods = ['bayes_nb', 'bayes_p', 'mdl_cm', 'mdl_cm_min', 'mean2sd', 'count_top']
    rots = ['equal', 'real', 'rest']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, het, title in zip(axes, ('none', 'measured'), ('(a) no individual differences', '(b) measured individual differences')):
        for m in methods:
            y = [next(r['any_rate'] for r in rows if r['method'] == m and r['het'] == het and r['phi'] == 'measured'
                      and r['rotation'] == rot and r['hub'] == 'none') for rot in rots]
            ax.plot(range(3), y, marker=MARK[m], color=COL[m], label=NAME[m], markersize=6)
        ax.axhline(0.05, color=NEUTRAL, lw=1, ls='--')
        ax.set_xticks(range(3), ['equal minutes', 'real rotation', 'real + 30% rest'])
        ax.set_title(title, loc='left')
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel('false-positive rate\n(>= 1 hub reported, none planted)')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    save(fig, F / 'fig3_e1_false_positives.pdf')
    # table
    L = ['## E1 false-positive rate (organising layer, phi measured)', '',
         '| method | het | equal | real | rest |', '|---|---|---|---|---|']
    for het in ('none', 'measured'):
        for m in e1_methods(rows):
            v = [next(r['any_rate'] for r in rows if r['method'] == m and r['het'] == het and r['phi'] == 'measured'
                      and r['rotation'] == rot and r['hub'] == 'none') for rot in rots]
            L.append(f"| {NAME[m]} | {het} | " + ' | '.join(f'{x:.2f}' for x in v) + ' |')
    TABLES.append('\n'.join(L))


def e1_methods(rows):
    seen = []
    for r in rows:
        if r['method'] not in seen:
            seen.append(r['method'])
    return seen


def fig4_e1_detection(e1):
    rows = e1['rows']
    methods = ['bayes_nb', 'bayes_p', 'mdl_cm', 'mdl_cm_min', 'per90_top', 'count_top']
    rhos = [1.25, 1.5, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, g, title in zip(axes, ('MID', 'FWD'), ('(a) hub planted in midfield', '(b) hub planted in attack')):
        for m in methods:
            y = [next(r['exact_rate'] for r in rows if r['method'] == m and r['het'] == 'measured' and r['phi'] == 'measured'
                      and r['rotation'] == 'real' and r['hub'] == g and r['rho'] == rho) for rho in rhos]
            fp = next(r['any_rate'] for r in rows if r['method'] == m and r['het'] == 'measured' and r['phi'] == 'measured'
                      and r['rotation'] == 'real' and r['hub'] == 'none')
            ax.plot(rhos, y, marker=MARK[m], color=COL[m], label=f'{NAME[m]} (false pos. {fp:.0%})', markersize=6)
        ax.set_xticks(rhos)
        ax.set_xlabel('hub multiplier rho')
        ax.set_title(title, loc='left')
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel('exact recovery\n(reported set = planted hub)')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    save(fig, F / 'fig4_e1_detection.pdf')
    L = ['## E1 detection / exact recovery (real rotation, measured phi and individual differences)', '',
         '| method | group | rho | detection | exact |', '|---|---|---|---|---|']
    for m in e1_methods(rows):
        for g in ('MID', 'FWD'):
            for rho in rhos:
                r = next(r for r in rows if r['method'] == m and r['het'] == 'measured' and r['phi'] == 'measured'
                         and r['rotation'] == 'real' and r['hub'] == g and r['rho'] == rho)
                L.append(f"| {NAME[m]} | {g} | {rho} | {r['detect_rate']:.2f} | {r['exact_rate']:.2f} |")
    TABLES.append('\n'.join(L))


def fig5_e2(e2):
    rows = e2['rows']
    ms = e2['m_values']
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharey=True, sharex=True)
    for ri, (het, htitle) in enumerate((('measured', 'individual differences'), ('none', 'identical players'))):
        for ci, (layer, ltitle) in enumerate((('org', 'organising'), ('box', 'threat'))):
            ax = axes[ri, ci]
            i = 0
            for rho in (2.0, 3.0):
                for g in ('MID', 'FWD'):
                    y = [next(r['detect_rate'] for r in rows if r['het'] == het and r['method'] == 'bayes_nb'
                              and r['layer'] == layer and r['m'] == m and r['hub'] == g and r['rho'] == rho) for m in ms]
                    ax.plot(ms, y, marker='o', color=SERIES[i], label=f'{g}, rho = {rho:g}', markersize=4)
                    i += 1
            fp = [next(r['any_rate'] for r in rows if r['het'] == het and r['method'] == 'bayes_nb' and r['layer'] == layer
                       and r['m'] == m and r['hub'] == 'none') for m in ms]
            ax.plot(ms, fp, color=NEUTRAL, ls='--', marker='x', label='false positives (no hub)')
            ax.axhline(0.8, color=GRIDLINE, lw=0.8)
            ax.set_xscale('log')
            ax.set_xticks(ms, [str(m) for m in ms])
            ax.minorticks_off()
            ax.set_ylim(-0.03, 1.05)
            ax.set_title(f'({"abcd"[2 * ri + ci]}) {ltitle}, {htitle}', loc='left', fontsize=8)
            if ri == 1:
                ax.set_xlabel('matches observed')
        axes[ri, 0].set_ylabel('detection rate (our model)')
    axes[0, 1].legend(loc='center left', bbox_to_anchor=(1.0, 0.0))
    save(fig, F / 'fig5_e2_matches.pdf')
    L = ['## Table 1: matches needed (detection >= 80%, false positives <= 5%; "-" = not reached within 38)', '',
         '| world | method | rho | org MID | org FWD | box MID | box FWD |', '|---|---|---|---|---|---|---|']
    t = e2['matches_needed']
    for het in ('measured', 'none'):
        for m in ('bayes_nb', 'bayes_p', 'mdl_cm_min'):
            for rho in (1.25, 1.5, 2.0, 3.0):
                cells = [t.get(f'{het}|{m}|{l}|{g}|{rho}') for l in ('org', 'box') for g in ('MID', 'FWD')]
                L.append(f'| {"individual differences" if het == "measured" else "identical players"} | {NAME[m]} | {rho} | '
                         + ' | '.join('-' if c is None else str(c) for c in cells) + ' |')
    TABLES.append('\n'.join(L))


GRIDLINE = '#c9c8c3'


def fig6_e4_heatmap(e4):
    units = e4['units']
    labels = [u['label'] for u in units]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.4), gridspec_kw={'hspace': 0.45})
    for ax, layer, title in zip(axes, ('org', 'box'), ('(a) organising hubs (open-play passes)',
                                                       '(b) threat hubs (passes into the box)')):
        probs = {}
        for ui, u in enumerate(units):
            L = u['layers'].get(layer)
            if not L:
                continue
            for p, pr in zip(L['players'], L['incl_prob']['bayes_nb']):
                probs.setdefault(p['name'], {})[ui] = pr
        names = sorted(probs, key=lambda n: -max(probs[n].values()))[:12]
        names.sort(key=lambda n: (min(probs[n]), -max(probs[n].values())))
        M = np.full((len(names), len(units)), np.nan)
        for i, n in enumerate(names):
            for ui, pr in probs[n].items():
                M[i, ui] = pr
        im = ax.imshow(np.nan_to_num(M, nan=-0.05), aspect='auto', cmap=SEQ, vmin=0, vmax=1, interpolation='nearest')
        for i in range(len(names)):
            for j in range(len(units)):
                if np.isnan(M[i, j]):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='#f0efec', lw=0))
        ax.set_yticks(range(len(names)), names)
        ax.set_xticks(range(len(units)), [l.replace('Rijkaard 2004-08', 'Rijk. 04-08') for l in labels], rotation=60,
                      ha='right')
        ax.grid(False)
        ax.set_title(title, loc='left')
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label('posterior inclusion probability')
    save(fig, F / 'fig6_e4_barca_heatmap.pdf')
    L = ['## E4 Barcelona: hubs per season (our model / plan Poisson model / MDL-CM)', '',
         '| period | matches | organising: ours | Poisson | MDL-CM | threat: ours | Poisson | MDL-CM |',
         '|---|---|---|---|---|---|---|---|']
    for u in units:
        c = []
        for layer in ('org', 'box'):
            Lr = u['layers'].get(layer, {'hubs': {}})
            for m in ('bayes_nb', 'bayes_p', 'mdl_cm'):
                c.append(', '.join(Lr['hubs'].get(m, [])) or '—')
        L.append(f"| {u['label']} | {u.get('n_matches', '')} | " + ' | '.join(c) + ' |')
    TABLES.append('\n'.join(L))


def fig7_e5(e5):
    units = e5['units']
    L = ['## E5 La Liga 2015/16: hubs per team', '',
         '| team | organising: ours | Poisson | MDL-CM | MDL-CM (per-min) | threat: ours | Poisson |',
         '|---|---|---|---|---|---|---|']
    counts = {m: [] for m in ('bayes_nb', 'bayes_p', 'mdl_cm', 'mdl_cm_min')}
    for u in units:
        o, b = u['layers']['org'], u['layers'].get('box', {'hubs': {}})
        grp = {p['name']: p['group'] for p in o['players']}

        def fmt(names):
            return ', '.join(f'{n} ({grp.get(n, "?")})' for n in names) or '—'
        L.append(f"| {u['team']} | {fmt(o['hubs']['bayes_nb'])} | {fmt(o['hubs']['bayes_p'])} | "
                 f"{fmt(o['hubs']['mdl_cm'])} | {fmt(o['hubs']['mdl_cm_min'])} | "
                 f"{', '.join(b['hubs'].get('bayes_nb', [])) or '—'} | {', '.join(b['hubs'].get('bayes_p', [])) or '—'} |")
        for m in counts:
            counts[m].append(len(o['hubs'][m]))
    TABLES.append('\n'.join(L))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    xs = np.arange(4)
    for i, m in enumerate(counts):
        c = np.array(counts[m])
        vals = [np.mean(c == 0), np.mean(c == 1), np.mean(c == 2), np.mean(c >= 3)]
        ax.bar(xs + (i - 1.5) * 0.2, vals, width=0.18, color=COL[m], label=NAME[m])
    ax.set_xticks(xs, ['0', '1', '2', '>= 3'])
    ax.set_xlabel('hubs reported per team (organising layer)')
    ax.set_ylabel('share of the 20 teams')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    save(fig, F / 'fig7_e5_league.pdf')


def fig8_e7(e7):
    picks = [('Lionel Messi', 'box', ('2011/12', '2014/15', '2015/16', '2020/21')),
             ('Xavi', 'org', ('2008/09', '2009/10', '2010/11', '2011/12'))]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (who, layer, seasons) in zip(axes, picks):
        i = 0
        for u in e7['units']:
            if u['team'] != 'Barcelona':
                continue
            ids = [int(k) for k, v in u['names'].items() if v == who]
            if not ids or str(ids[0]) not in u['layers'][layer]['trajectories']:
                continue
            tr = u['layers'][layer]['trajectories'][str(ids[0])]
            if u['season'] not in seasons:
                continue
            xs = sorted(int(x) for x in tr)
            ax.plot(xs, [tr[str(x)] if str(x) in tr else tr[x] for x in xs], color=SERIES[i], label=u['season'])
            i += 1
        ax.axhline(0.9, color=NEUTRAL, ls='--', lw=1)
        ax.set_xlabel('matches into the season')
        ax.set_title(f"{who}, {'threat' if layer == 'box' else 'organising'} layer", loc='left')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left', ncol=2, fontsize=6.5)
    axes[0].set_ylabel('posterior inclusion probability')
    save(fig, F / 'fig8_e7_sequential.pdf')
    s = e7['summary']
    L = ['## E7 confirmation time', '',
         '| layer | hubs at season end | stay > 0.9 | median share of season (0.9) | stay > 0.5 | median share of season (0.5) |',
         '|---|---|---|---|---|---|']
    def r2(x):
        return '—' if x is None else f'{x:.2f}'
    for layer in ('org', 'box'):
        v = s[layer]
        L.append(f"| {layer} | {v['n_final_hubs']} | {v['n_hubs']} | {r2(v['median_confirm_fraction'])} | "
                 f"{v['n_confirm_0.5']} | {r2(v['median_confirm_fraction_0.5'])} |")
    L += ['', '| unit | layer | hub | final probability | first match staying > 0.5 | > 0.9 |', '|---|---|---|---|---|---|']
    for u in e7['units']:
        for layer in ('org', 'box'):
            Lr = u['layers'][layer]
            for pid in Lr['final_hubs']:
                nm = u['names'].get(str(pid), pid)
                L.append(f"| {u['team']} {u['season']} | {layer} | {nm} | {Lr['final_incl'].get(str(pid), Lr['final_incl'].get(pid, 0)):.2f} | "
                         f"{Lr['confirm_at_0.5'].get(str(pid), Lr['confirm_at_0.5'].get(pid))} / {u['n_matches']} | "
                         f"{Lr['confirm_at'].get(str(pid), Lr['confirm_at'].get(pid))} |")
    TABLES.append('\n'.join(L))


SEASONS_E7 = ('2009/10', '2010/11', '2011/12', '2014/15', '2015/16', '2008/09', '2018/19')


def table_e3(e3):
    s = e3['summary']
    L = ['## Table 2a: E3 first half -> second half (gain in log predictive density over "no hub", mean per unit, 95% CI)', '']
    for layer in ('org', 'box'):
        L += [f"**{layer}** ({s[layer]['n_units']} units; saturated reference gain {s[layer]['saturated_gain_mean']:.1f})", '',
              '| method | gain | 95% CI | Jaccard H1-H2 | units with a hub |', '|---|---|---|---|---|']
        for m, v in s[layer].items():
            if not isinstance(v, dict):
                continue
            j = '—' if v['jaccard_mean'] is None else f"{v['jaccard_mean']:.2f}"
            L.append(f"| {NAME[m]} | {v['gain_mean']:.2f} | [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] | {j} | "
                     f"{v['share_units_with_hub']:.2f} |")
        L.append('')
    TABLES.append('\n'.join(L))


def table_e6(e6):
    s = e6['mean_fp_rate']
    L = ['## Table 2b: E6 within-position permutation (share of permutations with >= 1 hub, mean over units)', '',
         '| method | organising | threat |', '|---|---|---|']
    for m in s['org']:
        L.append(f"| {NAME[m]} | {s['org'][m]:.3f} | {s['box'][m]:.3f} |")
    TABLES.append('\n'.join(L))


def fig1_season(e4, season='2010/11'):
    u = next((u for u in e4['units'] if u['label'] == season), None)
    if u is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    L = u['layers']['org']
    order = np.argsort([p['k'] for p in L['players']])
    names = [L['players'][i]['name'] for i in order]
    gcol = {'DEF': SERIES[2], 'MID': SERIES[0], 'FWD': SERIES[1]}
    axes[0].barh(range(len(order)), [L['players'][i]['k'] for i in order], height=0.7,
                 color=[gcol[L['players'][i]['group']] for i in order])
    axes[0].set_yticks(range(len(order)), names)
    axes[0].set_xlabel('open-play passes made (season total)')
    axes[0].set_title(f'(a) Barcelona {season}: pass counts', loc='left')
    for layer, col, off, lab in (('org', SERIES[0], -0.18, 'organising'), ('box', SERIES[1], 0.18, 'threat')):
        Lr = u['layers'][layer]
        pmap = {p['name']: pr for p, pr in zip(Lr['players'], Lr['incl_prob']['bayes_nb'])}
        axes[1].barh(np.arange(len(order)) + off, [pmap.get(n, 0) for n in names], height=0.34, color=col, label=lab)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('posterior probability of being a hub (our model)')
    axes[1].set_title('(b) hub probability', loc='left')
    axes[1].legend(loc='lower right')
    for ax in axes:
        ax.grid(axis='y', visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in gcol.values()]
    axes[0].legend(handles, list(gcol), loc='lower right', title='group')
    save(fig, F / 'fig1_barca_season.pdf')


def table_calibration():
    rates, disp, alpha = load('league_rates.json'), load('dispersion.json'), load('alpha.json')
    if not (rates and disp and alpha):
        return
    L = ['## Calibration (La Liga 2015/16)', '',
         '| layer | group | rate per 90 | individual spread alpha (CV) | between-player dispersion of season totals |',
         '|---|---|---|---|---|']
    for layer in ('org', 'box'):
        het = disp['het'][f'{layer}_min450']
        for g in ('DEF', 'MID', 'FWD'):
            ratio = het[g].get('pearson_ratio', het[g].get('pearson_ratio_over_phi', 0) * het['phi_match'])
            L.append(f"| {layer} | {g} | {rates[layer][g]:.2f} | {alpha[layer][g]:.1f} ({alpha[layer][g] ** -0.5:.2f}) | "
                     f"{ratio:.1f} x Poisson |")
        L.append(f"| {layer} | per-match phi | {disp['phi_match'][f'{layer}_min1']['phi']:.2f} | | |")
    TABLES.append('\n'.join(L))


def table_sensitivity(sens):
    L = ['## Sensitivity (our model on E4 + E5 units)', '',
         '| variant | organising hubs | threat hubs | units unchanged | Messi threat-hub seasons | Xavi organising-hub seasons |',
         '|---|---|---|---|---|---|']
    for r in sens['summary']:
        L.append(f"| {r['variant']} | {r['hubs_org']} | {r['hubs_box']} | {r['share_units_unchanged']:.0%} | "
                 f"{r['messi_threat_seasons']} | {r['xavi_org_seasons']} |")
    TABLES.append('\n'.join(L))


def main():
    e1, e2, e3, e4, e5, e6, e7 = (load(n) for n in ('e1_null_power.json', 'e2_matches_needed.json', 'e3_half_season.json',
                                                    'e4_barca.json', 'e5_league.json', 'e6_permutation.json',
                                                    'e7_sequential.json'))
    if e4:
        fig1_season(e4)
        fig6_e4_heatmap(e4)
    if e1:
        fig3_e1_false_positives(e1)
        fig4_e1_detection(e1)
    if e2:
        fig5_e2(e2)
    if e3:
        table_e3(e3)
    if e5:
        fig7_e5(e5)
    if e6:
        table_e6(e6)
    if e7:
        fig8_e7(e7)
    table_calibration()
    sens = load('sensitivity.json')
    if sens:
        table_sensitivity(sens)
    open(R / 'tables.md', 'w').write('\n\n'.join(TABLES) + '\n')
    print('figures:', sorted(p.name for p in F.glob('*.pdf')))


if __name__ == '__main__':
    main()
