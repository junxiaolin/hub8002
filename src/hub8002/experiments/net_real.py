"""Network extension on real data: linking cores (network VOA, mode 'net') for Barcelona and the 20 teams
of La Liga 2015/16; for reference also the 'full' version (incl. shooting) and the plug-in estimate
(raw rows, no shrinkage). Writes results/net_barca.json and results/net_league.json.
Usage: python -I -m hub8002.experiments.net_real
"""
import numpy as np

from ..models.net_model import net_cores
from .common import dump
from .net_common import calib, unit_for
from .v2_real import units


def fit(team, label, *, roles=False, mode='net', opts=None, extras=True):
    opts = opts or {}
    u, cand, names, minutes = unit_for(team, label, roles=roles, min_minutes=opts.get('min_minutes'))
    q, gbar, alpha, phi, a = calib(roles)
    if opts.get('phi_off'):
        phi = 1.0
    kw = dict(phi=phi, alpha_mult=opts.get('alpha_mult', 1.0), cred=opts.get('cred', 0.9),
              pos_prob=opts.get('pos_prob', 0.9), n_draws=opts.get('n_draws', 2000), seed=opts.get('seed', 0),
              kappa=opts.get('kappa', 5.0), repl=opts.get('repl', 'receptions'), shot_shape=a,
              q_base=opts.get('q_base'))
    f = net_cores(u, q, gbar, alpha, cand, mode=mode, **kw)
    touches = u.touches
    out = {'team': team, 'label': label, 'mode': mode, 'n_holders': len(u.ids),
           'team_xg_model': f.team_xg, 'team_xg_obs': float(u.X.sum()), 'chains': float(u.N0.sum()),
           'players': [{'id': int(u.ids[i]), 'name': names[i], 'group': str(u.group[i]), 'minutes': float(minutes[i]),
                        'touches': float(touches[i]), 'visits': float(f.visits[i]), 'edge': float(f.edge[i]),
                        'end_rate': float(u.E[i] / touches[i]) if touches[i] else None,
                        'retain': float(f.retain[i]), 'choice': float(f.choice[i]),
                        'voa': float(f.voa_mean[i]), 'voa_lo': float(f.voa_lo[i]), 'voa_hi': float(f.voa_hi[i]),
                        'p_pos': float(f.p_pos[i]), 'p_top': float(f.p_top[i])}
                       for i in np.flatnonzero(cand)],
           'core_set': [names[i] for i in f.core_set], 'top': names[f.top], 'p_top_max': float(f.p_top.max()),
           'clear': f.clear}
    out['players'].sort(key=lambda r: -r['voa'])
    if extras and mode == 'net':
        g = net_cores(u, q, gbar, alpha, cand, mode='full', **kw)
        out['full'] = {'core_set': [names[i] for i in g.core_set], 'top': names[g.top],
                       'p_top_max': float(g.p_top.max()),
                       'voa': {names[i]: float(g.voa_mean[i]) for i in np.flatnonzero(cand)}}
        p = net_cores(u, q, gbar, alpha, cand, mode='net', plugin=True, phi=phi, repl=kw['repl'], shot_shape=a,
                      q_base=kw['q_base'])
        out['plugin_top'] = names[p.top]
        out['touches_top'] = names[int(np.flatnonzero(cand)[np.argmax(touches[cand])])]
    return out


def main():
    barca, early, league = units()
    b = [fit(t, l) for t, l in barca]
    e = [fit(t, l) for t, l in early]
    lg = [fit(t, l) for t, l in league]
    dump({'units': b, 'early_single': e}, 'net_barca.json')
    dump({'units': lg}, 'net_league.json')
    for u in b + lg:
        top3 = ', '.join(f"{r['name'].split()[0]} {r['voa']:.2f}" for r in u['players'][:3])
        print(f"{u['team'][:12]:12s} {u['label']:17s} core {u['core_set'] or '-'} (top {u['top']} {u['p_top_max']:.2f})"
              f" | {top3} | full {u['full']['top']} | xG model/obs {u['team_xg_model']:.1f}/{u['team_xg_obs']:.1f}")


if __name__ == '__main__':
    main()
