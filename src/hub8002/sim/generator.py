"""Simulated teams calibrated on real 2015/16 squads.

A simulated team copies a real team-season's squad structure: who played, how many minutes in each
match, which position group. Only the pass counts are regenerated:

    k_i ~ NB(mean = T_i * lambda_g(i) * u_i * rho^{s_i},  Var = phi * mean)        (season total)

with u_i ~ Gamma(alpha_g, alpha_g) the player's own style multiplier (alpha = inf: no individual
differences, the plan's original generator). Summing per-match NB counts with a common Fano factor
phi gives exactly this season-level NB, so we draw season totals directly.
Edges for PageRank: each player's passes are split over team-mates in proportion to co-presence minutes.
"""
import numpy as np


def nb_draw(rng, mean, phi):
    mean = np.asarray(mean, float)
    if phi <= 1.0 + 1e-9:
        return rng.poisson(mean)
    r = mean / (phi - 1.0)
    p = 1.0 / phi
    out = np.zeros_like(mean)
    pos = mean > 0
    out[pos] = rng.negative_binomial(r[pos], p)
    return out


def rotation_minutes(min_matrix, mode, rng, rest_frac=0.3):
    """Season exposure T (90-min units) per player from a (matches x players) minutes matrix.

    mode: 'equal' (everyone gets the squad-average exposure), 'real', 'rest' (real, then each
    appearance is dropped with probability rest_frac).
    """
    M = np.asarray(min_matrix, float)
    if mode == 'real':
        T = M.sum(axis=0) / 90.0
    elif mode == 'rest':
        keep = rng.random(M.shape) >= rest_frac
        T = (M * keep).sum(axis=0) / 90.0
    elif mode == 'equal':
        T = np.full(M.shape[1], M.sum() / M.shape[1] / 90.0)
    else:
        raise ValueError(mode)
    return T


def simulate_team(T, group, base_rate, hubs, rho, phi, rng, alpha=None, copresence=None):
    """Returns (k, edges) for one simulated season.

    T: exposure per player; group: group label per player; base_rate: {group: rate per 90};
    hubs: indices of planted hubs; alpha: {group: gamma shape} or None (no individual differences);
    copresence: optional (n x n) co-presence minutes for splitting passes into edges.
    """
    T = np.asarray(T, float)
    group = np.asarray(group)
    lam = np.array([base_rate[g] for g in group])
    u = np.ones(len(T))
    if alpha is not None:
        a = np.array([alpha[g] for g in group], float)
        finite = np.isfinite(a)
        u[finite] = rng.gamma(a[finite], 1.0 / a[finite])
    mult = np.ones(len(T))
    mult[list(hubs)] = rho
    k = nb_draw(rng, T * lam * u * mult, phi)
    edges = None
    if copresence is not None:
        C = np.asarray(copresence, float).copy()
        np.fill_diagonal(C, 0.0)
        edges = np.zeros_like(C)
        for i in range(len(T)):
            s = C[i].sum()
            if k[i] > 0 and s > 0:
                edges[i] = rng.multinomial(int(k[i]), C[i] / s)
    return k.astype(int), edges


EXPOSURE_KEY = {'att': 'own', 'prog': 'own', 'def': 'opp', 'org': 'minutes', 'box': 'minutes'}


def simulate_layer(T, group, layer, hubs, rho, rng, *, rates, phi, alpha, value_layer):
    """Season totals for one layer. Count layers: NB with Fano phi in pass units. Value layers
    (expected-goal units): quasi-Poisson, i.e. Poisson in effective counts of size phi, then rescaled."""
    if value_layer:
        k, _ = simulate_team(T, group, {g: r / phi for g, r in rates.items()}, hubs, rho, 1.0, rng, alpha=alpha)
        return k * phi
    k, _ = simulate_team(T, group, rates, hubs, rho, phi, rng, alpha=alpha)
    return k.astype(float)
