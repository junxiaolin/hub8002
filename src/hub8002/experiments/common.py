"""Shared setup for experiments: tables, calibrated constants and the list of methods."""
import json
from functools import lru_cache

import numpy as np

from .. import ROOT, load_config, load_table
from ..models.baselines import baseline_hubs, pagerank_scores
from ..models.bayes_hub import fit_hubs, fit_hubs_het

CFG = load_config()
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
# layer -> (count column, exposure column) in player_period
LAYER_SPEC = {'att': ('k_att', 'T_own'), 'prog': ('k_prog', 'T_own'), 'def': ('k_def', 'T_opp'),
              'org': ('k_org', 'T'), 'box': ('k_box', 'T')}
VALUE_LAYERS = ('att', 'prog', 'def')
import os as _os
# v1 experiments (pass-network hubs) run with HUB8002_LAYERS=org,box; default = config 'layers'
LAYERS = tuple(_os.environ['HUB8002_LAYERS'].split(',')) if _os.environ.get('HUB8002_LAYERS') \
    else tuple(CFG.get('layers', ['att', 'prog', 'def', 'org']))
LAYER_NAME = {'att': '进攻核心', 'prog': '组织核心', 'def': '防守核心', 'org': '传球量（对照）', 'box': '传进禁区（旧威胁层）'}
GROUPS = tuple(CFG['hub_groups'])

BAYES = ('bayes_nb', 'bayes_p', 'nb_no_position', 'nb_no_minutes', 'nb_no_shared_rho')
BASELINES = ('mdl_cm', 'mdl_er', 'mdl_cm_min', 'mdl_er_min', 'count_top', 'per90_top', 'pagerank_top',
             'mean2sd', 'average', 'loubar')
ALL_METHODS = BAYES + BASELINES
LABEL = {
    'bayes_nb': '本模型（含个体差异）', 'bayes_p': '原方案泊松模型', 'nb_no_position': '消融：去掉位置',
    'nb_no_minutes': '消融：去掉分钟', 'nb_no_shared_rho': '消融：不共享 ρ', 'mdl_cm': 'MDL-CM',
    'mdl_er': 'MDL-ER', 'mdl_cm_min': 'MDL-CM（按分钟折算）', 'mdl_er_min': 'MDL-ER（按分钟折算）',
    'count_top': '传球数第一', 'per90_top': '每 90 分钟第一', 'pagerank_top': 'PageRank 第一',
    'mean2sd': '均值 + 2SD', 'average': 'Average', 'loubar': 'Loubar'}


@lru_cache(None)
def constants():
    rates = json.load(open(RESULTS / 'league_rates.json'))
    disp = json.load(open(RESULTS / 'dispersion.json'))
    alpha = json.load(open(RESULTS / 'alpha.json')) if (RESULTS / 'alpha.json').exists() else None
    # count layers: phi >= 1 (Poisson floor); value layers: phi is the scale of one effective count
    phi = {layer: (disp['phi_match'][f'{layer}_min1']['phi'] if layer in VALUE_LAYERS
                   else max(1.0, disp['phi_match'][f'{layer}_min1']['phi']))
           for layer in LAYER_SPEC if f'{layer}_min1' in disp['phi_match']}
    return rates, phi, alpha


def model_kwargs():
    m = CFG['model']
    return dict(a=m['prior_shape'], rho_max=m['rho_max'], s_max=m['s_max'])


def run_bayes(method, k, T, group, layer, *, alpha=None, phi=None, rates=None, n_grid=None):
    """One of the Bayesian variants -> HubFit-like (hubs, incl_prob, log_bf)."""
    r0, p0, a0 = constants()
    rates = rates or r0[layer]
    phi = p0[layer] if phi is None else phi
    alpha = alpha or (a0[layer] if a0 else None)
    kw = model_kwargs()
    k, T, group = np.asarray(k, float), np.asarray(T, float), np.asarray(group)
    if method == 'bayes_p':
        return fit_hubs(k, T, group, {g: rates[g] for g in GROUPS}, phi=phi,
                        n_grid=n_grid or CFG['model']['n_rho_grid'], **kw)
    ng = n_grid or CFG['model']['n_rho_grid_het']
    nl = CFG['model']['n_lambda_grid']
    if method == 'bayes_nb':
        return fit_hubs_het(k, T, group, rates, alpha, phi=phi, n_grid=ng, n_lam=nl, **kw)
    if method == 'nb_no_position':
        return fit_hubs_het(k, T, np.array(['ALL'] * len(k)), {'ALL': rates['ALL']}, {'ALL': alpha['ALL']},
                            phi=phi, n_grid=ng, n_lam=nl, **kw)
    if method == 'nb_no_minutes':
        return fit_hubs_het(k, np.full(len(k), T.mean()), group, rates, alpha, phi=phi, n_grid=ng, n_lam=nl, **kw)
    if method == 'nb_no_shared_rho':
        hubs, incl = [], np.zeros(len(k))
        lbf = 0.0
        for g in sorted(set(group)):
            idx = np.flatnonzero(group == g)
            f = fit_hubs_het(k[idx], T[idx], group[idx], rates, alpha, phi=phi, n_grid=ng, n_lam=nl, **kw)
            hubs += [int(idx[i]) for i in f.hubs]
            incl[idx] = f.incl_prob
            lbf = max(lbf, f.log_bf)
        return _Simple(sorted(hubs), incl, lbf)
    raise ValueError(method)


class _Simple:
    def __init__(self, hubs, incl_prob, log_bf):
        self.hubs, self.incl_prob, self.log_bf = hubs, incl_prob, log_bf


def run_methods(methods, k, T, group, layer, pr=None, **kw):
    """-> {method: sorted hub index list}; Bayesian methods also give incl_prob and log_bf."""
    out, extra = {}, {}
    for m in methods:
        if m in BAYES:
            f = run_bayes(m, k, T, group, layer, **kw)
            out[m] = sorted(int(i) for i in f.hubs)
            extra[m] = {'incl_prob': np.asarray(f.incl_prob), 'log_bf': float(f.log_bf)}
        elif m == 'pagerank_top':
            out[m] = baseline_hubs(m, pr=pr) if pr is not None else []
        else:
            kb = np.asarray(k, float)
            if layer in VALUE_LAYERS:                     # MDL needs integer counts: use effective counts
                kb = kb / constants()[1][layer]
            out[m] = baseline_hubs(m, kb, T)
    return out, extra


def team_units():
    """(team, season) units for real-data experiments: 20 teams of 2015/16 + Barcelona's other seasons."""
    matches = load_table('matches')
    league = sorted(set(matches[matches.season == CFG['league_season']].home))
    units = [(t, CFG['league_season']) for t in league]
    units += [(CFG['focus_team'], s) for s in CFG['seasons'].values() if s != CFG['league_season']]
    return units


def unit_data(pp, team, label, layer, ed=None, min_minutes=None, groups=None):
    """Candidates (outfield, >= min minutes) of one team-period: k, T, group, names, ids, PageRank."""
    min_minutes = CFG['min_minutes'] if min_minutes is None else min_minutes
    d = pp[(pp.team == team) & (pp.label == label)]
    cand = d[d.group.isin(groups or GROUPS) & (d.minutes >= min_minutes)].reset_index(drop=True)
    pr = None
    if ed is not None:
        e = ed[(ed.team == team) & (ed.label == label)]
        wcol = f'w_{layer}' if f'w_{layer}' in e.columns else 'w_org'
        scores = pagerank_scores(list(d.player_id), zip(e.i, e.j, e[wcol]))
        pr = [scores.get(p, 0.0) for p in cand.player_id]
    kc, tc = LAYER_SPEC[layer]
    return (cand[kc].to_numpy(float), cand[tc].to_numpy(float), cand['group'].to_numpy(),
            list(cand['name']), list(cand['player_id']), pr)


def dump(obj, name):
    RESULTS.mkdir(exist_ok=True)
    json.dump(obj, open(RESULTS / name, 'w'), ensure_ascii=False, indent=1, default=_np)


def _np(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))
