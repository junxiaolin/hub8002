"""Which position partition do the data support? (2026-11-08)

The "average player of the position" is the baseline of every core, so the partition matters (six roles
keep only 36-73% of top cores). Here the data choose. Candidates, built from minutes per raw StatsBomb
position (scripts/build_position_minutes.py); each player takes the group he played most minutes in:

  one     all outfield players together (no position)
  three   DEF / MID / FWD                                (current)
  three_w DEF / MID incl. wingers / CF
  four    CB / FB / MID / FWD
  six     CB / FB / CM (holding, central, attacking) / WM / W / CF   (current sensitivity)
  seven   CB / FB / DM / CM (central + attacking) / WM / W / CF
  eight   CB / FB / DM / CM / AM / WM / W / CF

Criterion, on La Liga 2015/16 (the calibration season), separately for each layer:
  node layers (att / prog / def): Gamma-Poisson (negative binomial) marginal likelihood of each player's
      value given his exposure, per group (lambda_g, alpha_g) at their maximum;
  network (linking): Dirichlet-multinomial marginal likelihood of each holder's full row (to which
      team-mate, shot, end), prior centre = the partition's average-player row, alpha_g per group.
  Description length DL = -log ML + (#hyper-parameters / 2) log n (two-part code, BIC form; n = players),
  in nats; lower is better. The data are the same under every partition, so DLs are comparable.
  'net' lets the partition define both whom a holder is compared with and the receiving blocks of the
  null model; 'net_src' keeps the receiving blocks at GK / DEF / MID / FWD and varies only the former;
  'net_src4' the same with receiving blocks GK / CB / FB / MID / FWD (the best receiving blocks under 'net').
  The partition chosen for all layers is the one with the lowest total DL over att, prog, def and net_src4
  (and the highest total leave-one-team-out score), see summary['overall'].
  Robustness: leave-one-team-out predictive log-likelihood (hyper-parameters fitted on the other 19 teams,
  evaluated on the held-out team); higher is better.

Writes results/partition_selection.json.  Usage: python -I -m hub8002.experiments.partitions [workers]
"""
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

from .. import load_table
from ..models.core_model import eb_alpha as eb_alpha_node
from ..models.core_model import nb_loglik
from ..models.net_model import _counts, build_unit, dm_loglik, eb_alpha, league_profiles, replacement_rows
from .common import CFG, LAYER_SPEC, constants, dump
from .net_common import calib, label_matches, league_teams, tables

DEF_ = ['RB', 'RCB', 'CB', 'LCB', 'LB', 'RWB', 'LWB']
CB_, FB_ = ['RCB', 'CB', 'LCB'], ['RB', 'LB', 'RWB', 'LWB']
DM_, CM_, AM_ = ['RDM', 'CDM', 'LDM'], ['RCM', 'CM', 'LCM'], ['RAM', 'CAM', 'LAM']
WM_, W_, CF_ = ['RM', 'LM'], ['RW', 'LW'], ['RCF', 'CF', 'LCF', 'SS']


def _p(**groups):
    out = {'GK': 'GK'}
    for g, codes in groups.items():
        out.update({c: g for c in codes})
    return out


PARTITIONS = {
    'one': _p(OUT=DEF_ + DM_ + CM_ + AM_ + WM_ + W_ + CF_),
    'three': _p(DEF=DEF_, MID=DM_ + CM_ + AM_ + WM_, FWD=W_ + CF_),
    'three_w': _p(DEF=DEF_, MID=DM_ + CM_ + AM_ + WM_ + W_, FWD=CF_),
    'four': _p(CB=CB_, FB=FB_, MID=DM_ + CM_ + AM_ + WM_, FWD=W_ + CF_),
    'six': _p(CB=CB_, FB=FB_, CM=DM_ + CM_ + AM_, WM=WM_, W=W_, CF=CF_),
    'seven': _p(CB=CB_, FB=FB_, DM=DM_, CM=CM_ + AM_, WM=WM_, W=W_, CF=CF_),
    'eight': _p(CB=CB_, FB=FB_, DM=DM_, CM=CM_, AM=AM_, WM=WM_, W=W_, CF=CF_),
}
NODE_LAYERS = ('att', 'prog', 'def')


@lru_cache(None)
def _pmin():
    return load_table('position_minutes')


def player_groups(team, label, partition):
    """{player_id: group} for one team-period: the group with most minutes over the period's matches."""
    pm = _pmin()
    d = pm[(pm.team == team) & pm.match_id.isin(label_matches(team, label))]
    cols = [c for c in d.columns if c.startswith('pmin_')]
    s = d.groupby('player_id')[cols].sum()
    mp = PARTITIONS[partition]
    g = s.T.groupby(lambda c: mp[c[5:]]).sum().T                      # players x groups
    return {int(p): str(x) for p, x in g.idxmax(axis=1).items()}


# ---------------------------------------------------------------- node layers
def node_table(partition):
    """2015/16 league players (>= min minutes, outfield): team, group, and (k, T) per layer."""
    _, pp, _, _ = tables()
    season = CFG['league_season']
    d = pp[(pp.label == season) & (pp.minutes >= CFG['min_minutes'])].copy()
    grp = {}
    for t in league_teams():
        grp.update({(t, p): g for p, g in player_groups(t, season, partition).items()})
    d['pgroup'] = [grp.get((t, int(p))) for t, p in zip(d.team, d.player_id)]
    return d[d.pgroup.notna() & (d.pgroup != 'GK')]


def _fit_node(k, T, phi):
    """Joint maximum of the NB marginal likelihood over (lambda, alpha) for one group -> (lam, alpha, ll)."""
    lam0 = k.sum() / T.sum()
    a0 = eb_alpha_node(k, T, lam0, phi)
    ks, Ts = k / phi, T / phi
    f = lambda x: -nb_loglik(ks, Ts, np.exp(x[0]), np.exp(x[1]))
    res = minimize(f, [np.log(lam0), np.log(a0)], method='L-BFGS-B',
                   bounds=[(np.log(lam0) - 3, np.log(lam0) + 3), (np.log(0.1), np.log(1e4))])
    return float(np.exp(res.x[0])), float(np.exp(res.x[1])), float(-res.fun)


def node_scores(partition):
    _, phi, _ = constants()
    d = node_table(partition)
    out = {}
    for layer in NODE_LAYERS:
        kc, tc = LAYER_SPEC[layer]
        x = d[d[tc] > 0]
        ll, groups = 0.0, {}
        for g, dg in x.groupby('pgroup'):
            lam, al, l = _fit_node(dg[kc].to_numpy(float), dg[tc].to_numpy(float), phi[layer])
            ll += l
            groups[g] = {'n': int(len(dg)), 'lambda': lam, 'alpha': al}
        n_par = 2 * len(groups)
        cv = 0.0                                   # leave one team out
        for t in x.team.unique():
            tr, te = x[x.team != t], x[x.team == t]
            for g, dg in te.groupby('pgroup'):
                trg = tr[tr.pgroup == g]
                if len(trg) < 3:
                    trg = tr                       # unseen tiny group: fall back to everyone
                lam, al, _ = _fit_node(trg[kc].to_numpy(float), trg[tc].to_numpy(float), phi[layer])
                cv += nb_loglik(dg[kc].to_numpy(float) / phi[layer], dg[tc].to_numpy(float) / phi[layer], lam, al)
        out[layer] = {'log_ml': ll, 'n_params': n_par, 'n_players': int(len(x)),
                      'dl': -ll + 0.5 * n_par * np.log(len(x)), 'loto_cv': float(cv), 'groups': groups}
    return out


# ---------------------------------------------------------------- network layer
def net_units(partition):
    ch, _, _, _ = tables()
    season = CFG['league_season']
    return {t: build_unit(ch, t, label_matches(t, season), player_groups(t, season, partition))
            for t in league_teams()}


def _net_fit(units, dest, phi):
    q, _ = league_profiles(units, dest)
    alpha = {g: eb_alpha(units, q, g, phi) for g in dest}
    return q, alpha


def _net_ll(units, q, alpha, phi):
    ll = 0.0
    for u in units:
        c, pbar = _counts(u) / phi, replacement_rows(u, q)
        a = np.array([alpha[g] for g in u.group])[:, None] * pbar
        ll += dm_loglik(c, a)
    return ll


def net_scores(partition):
    _, _, _, phi, _ = calib()
    units = net_units(partition)
    dest = tuple(sorted({g for u in units.values() for g in u.group}, key=lambda g: (g != 'GK', g)))
    q, alpha = _net_fit(list(units.values()), dest, phi)
    ll = _net_ll(list(units.values()), q, alpha, phi)
    n_holders = int(sum(len(u.ids) for u in units.values()))
    n_par = len(dest) * (len(dest) + 1) + len(dest)    # rows over dest + shot + end (minus 1), and alphas
    cv = 0.0
    for t, u in units.items():
        rest = [v for s, v in units.items() if s != t]
        qt, at = _net_fit(rest, dest, phi)
        cv += _net_ll([u], qt, at, phi)
    return {'log_ml': ll, 'n_params': n_par, 'n_holders': n_holders,
            'dl': -ll + 0.5 * n_par * np.log(n_holders), 'loto_cv': float(cv),
            'alpha': alpha, 'q': q}


# network, decoupled: the partition only says whom a holder is compared with (source rows); the blocks
# that receive passes in the null model stay the three groups GK / DEF / MID / FWD for every partition.
def _dst_names(dst_partition):
    return ('GK',) + tuple(sorted({g for g in PARTITIONS[dst_partition].values()} - {'GK'}))


def _units_src_dst(partition, dst_partition='three'):
    ch, _, _, _ = tables()
    season = CFG['league_season']
    out = {}
    for t in league_teams():
        ids = label_matches(t, season)
        src = build_unit(ch, t, ids, player_groups(t, season, partition))
        dst = build_unit(ch, t, ids, player_groups(t, season, dst_partition)).group
        out[t] = (src, dst)
    return out


def _profiles_src_dst(pairs, DST):
    srcs = sorted({g for u, _ in pairs for g in u.group})
    q = {}
    for g in srcs:
        tot = np.zeros(len(DST) + 2)
        for u, dst in pairs:
            m = u.group == g
            for h, name in enumerate(DST):
                tot[h] += u.C[m][:, dst == name].sum()
            tot[-2] += u.S[m].sum()
            tot[-1] += u.E[m].sum()
        q[g] = tot / tot.sum()
    return q


def _pbar_src_dst(u, dst, q, DST):
    """Average-player rows: source row q[group of i] over DST blocks, shared within a block by receptions."""
    n = len(u.ids)
    w = u.C.sum(0)
    hidx = np.array([DST.index(x) for x in dst])
    out = np.zeros((n, n + 2))
    for i in range(n):
        qi = q[u.group[i]]
        for h in range(len(DST)):
            js = np.flatnonzero((hidx == h) & (np.arange(n) != i))
            if len(js) and w[js].sum() > 0:
                out[i, js] = qi[h] * w[js] / w[js].sum()
        out[i, n], out[i, n + 1] = qi[-2], qi[-1]
    return out / out.sum(1, keepdims=True)


def _fit_src_dst(pairs, phi, DST):
    q = _profiles_src_dst(pairs, DST)
    rows = {}
    for u, dst in pairs:
        pb, c = _pbar_src_dst(u, dst, q, DST), _counts(u) / phi
        for g in set(u.group):
            m = u.group == g
            rows.setdefault(g, []).append((c[m], pb[m]))
    alpha = {}
    for g, rr in rows.items():
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda la: -sum(dm_loglik(c, np.exp(la) * p) for c, p in rr),
                              bounds=(np.log(1.0), np.log(1e5)), method='bounded')
        alpha[g] = float(np.exp(res.x))
    return q, alpha


def _ll_src_dst(pairs, q, alpha, phi, DST):
    ll = 0.0
    for u, dst in pairs:
        a = np.array([alpha.get(g, 100.0) for g in u.group])[:, None] * _pbar_src_dst(u, dst, q, DST)
        ll += dm_loglik(_counts(u) / phi, a)
    return ll


def net_scores_decoupled(partition, dst_partition='three'):
    _, _, _, phi, _ = calib()
    DST = _dst_names(dst_partition)
    units = _units_src_dst(partition, dst_partition)
    pairs = list(units.values())
    q, alpha = _fit_src_dst(pairs, phi, DST)
    ll = _ll_src_dst(pairs, q, alpha, phi, DST)
    n_holders = int(sum(len(u.ids) for u, _ in pairs))
    n_par = len(q) * (len(DST) + 1) + len(q)
    cv = 0.0
    for t, pr in units.items():
        qt, at = _fit_src_dst([v for s, v in units.items() if s != t], phi, DST)
        cv += _ll_src_dst([pr], qt, at, phi, DST)
    return {'log_ml': ll, 'n_params': n_par, 'n_holders': n_holders,
            'dl': -ll + 0.5 * n_par * np.log(n_holders), 'loto_cv': float(cv), 'alpha': alpha}


def score(partition):
    return partition, {'node': node_scores(partition), 'net': net_scores(partition),
                       'net_src': net_scores_decoupled(partition), 'net_src4': net_scores_decoupled(partition, 'four')}


def main(workers=7):
    with Pool(min(workers, len(PARTITIONS))) as pool:
        res = dict(pool.map(score, list(PARTITIONS)))
    layers = list(NODE_LAYERS) + ['net', 'net_src', 'net_src4']
    get = lambda p, l: res[p][l] if l.startswith('net') else res[p]['node'][l]
    summary = {}
    for l in layers:
        best_dl = min(PARTITIONS, key=lambda p: get(p, l)['dl'])
        best_cv = max(PARTITIONS, key=lambda p: get(p, l)['loto_cv'])
        summary[l] = {'best_by_dl': best_dl, 'best_by_loto_cv': best_cv,
                      'delta_dl_vs_three': {p: get(p, l)['dl'] - get('three', l)['dl'] for p in PARTITIONS},
                      'delta_cv_vs_three': {p: get(p, l)['loto_cv'] - get('three', l)['loto_cv'] for p in PARTITIONS}}
        print(l, 'best DL', best_dl, 'best CV', best_cv)
        for p in PARTITIONS:
            print(f"   {p:8s} dDL {summary[l]['delta_dl_vs_three'][p]:9.1f}   dCV {summary[l]['delta_cv_vs_three'][p]:9.1f}")
    tot = {p: sum(get(p, l)['dl'] for l in ('att', 'prog', 'def', 'net_src4')) for p in PARTITIONS}
    totcv = {p: sum(get(p, l)['loto_cv'] for l in ('att', 'prog', 'def', 'net_src4')) for p in PARTITIONS}
    summary['overall'] = {'total_dl_vs_three': {p: tot[p] - tot['three'] for p in PARTITIONS},
                          'total_cv_vs_three': {p: totcv[p] - totcv['three'] for p in PARTITIONS},
                          'best_by_dl': min(tot, key=tot.get), 'best_by_loto_cv': max(totcv, key=totcv.get)}
    print('overall', summary['overall'])
    dump({'partitions': {p: sorted({g for g in m.values()}) for p, m in PARTITIONS.items()},
          'summary': summary, 'scores': res}, 'partition_selection.json')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
