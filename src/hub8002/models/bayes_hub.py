"""Exposure- and position-aware Bayesian hub model.

    k_i ~ Poisson(T_i * lambda_g(i) * rho^{s_i}),   lambda_g ~ Gamma(a, b_g)  (mean a / b_g = prior_mean[g]),
    log rho ~ U(0, log rho_max],   P(S) = 1/(s_max+1) * C(N, |S|)^{-1},   |S| <= s_max.

lambda_g is integrated out analytically; log rho on a midpoint grid; every S with |S| <= s_max is
enumerated. Quasi-Poisson correction for over-dispersion phi: (k, T) -> (k / phi, T / phi).

`fit_hubs_het` is the heterogeneity-aware variant: non-hub players of a group have their own rates
lambda_g * u_i with u_i ~ Gamma(alpha_g, alpha_g) (mean 1), so each count is negative binomial;
hubs are players whose rate is rho times that. lambda_g is integrated on a log grid.
"""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.special import gammaln, logsumexp


@dataclass
class HubFit:
    hubs: list                      # indices of the maximum-posterior hub set
    incl_prob: np.ndarray           # posterior inclusion probability per player
    log_bf: float                   # log BF "at least one hub" vs "no hub"
    log_bf_map: float               # log p(k | S_MAP) - log p(k | empty)
    rho_mean: float                 # posterior mean of rho given at least one hub
    lam_mean: dict                  # posterior mean of each group's base rate
    p_none: float                   # posterior probability of no hub
    set_post: list = field(default_factory=list)   # top sets: (indices, posterior prob)
    log_ev: float = 0.0             # log marginal likelihood log sum_S P(S) p(k | S) (scaled data)


def enumerate_sets(n, s_max):
    sets = [()]
    for s in range(1, min(s_max, n) + 1):
        sets.extend(combinations(range(n), s))
    return sets


def _log_prior_sets(sets, n, s_max):
    sizes = np.array([len(s) for s in sets])
    s_eff = min(s_max, n)
    return -np.log(s_eff + 1) - (gammaln(n + 1) - gammaln(sizes + 1) - gammaln(n - sizes + 1))


def _membership(sets, n):
    m = np.zeros((len(sets), n))
    for r, s in enumerate(sets):
        m[r, list(s)] = 1.0
    return m


def _finish(sets, log_ml, log_prior, member, rho_post_mean, lam_post, groups):
    lp = log_ml + log_prior
    post = np.exp(lp - logsumexp(lp))
    best = int(np.argmax(post))
    p_none = float(post[0])
    lm_hub = logsumexp(log_ml[1:] + log_prior[1:]) - logsumexp(log_prior[1:]) if len(sets) > 1 else -np.inf
    order = np.argsort(-post)[:10]
    w_hub = post[1:] / max(1 - p_none, 1e-300)
    return HubFit(
        hubs=list(sets[best]), incl_prob=post @ member,
        log_bf=float(lm_hub - log_ml[0]), log_bf_map=float(log_ml[best] - log_ml[0]),
        rho_mean=float(w_hub @ rho_post_mean[1:]) if len(sets) > 1 else 1.0,
        lam_mean={g: float(post @ lam_post[:, gi]) for gi, g in enumerate(groups)},
        p_none=p_none, set_post=[(list(sets[i]), float(post[i])) for i in order],
        log_ev=float(logsumexp(lp)))


def fit_hubs(k, T, group, prior_mean, *, a=1.0, rho_max=4.0, s_max=3, phi=1.0, n_grid=400,
             candidates=None):
    """Poisson-Gamma hub model (the plan's model). k, T, group: per player; prior_mean: {group: rate per T}.

    candidates: optional boolean mask of players allowed in S (default: all).
    """
    k = np.asarray(k, float) / phi
    T = np.asarray(T, float) / phi
    group = np.asarray(group)
    n = len(k)
    groups = sorted(set(group))
    G = np.array([[group[i] == g for g in groups] for i in range(n)], float)  # n x G
    b = np.array([a / prior_mean[g] for g in groups])
    K = k @ G
    U = T @ G
    cand = np.arange(n) if candidates is None else np.flatnonzero(candidates)
    sets_c = enumerate_sets(len(cand), s_max)
    sets = [tuple(int(cand[i]) for i in s) for s in sets_c]
    member = _membership(sets, n)
    H = (member * T) @ G                      # hub exposure per set and group: S x G
    KS = member @ k                           # passes made by hubs: S
    logr = (np.arange(n_grid) + 0.5) / n_grid * np.log(rho_max)
    rho = np.exp(logr)
    E = (b + U)[None, :, None] + (rho[None, None, :] - 1.0) * H[:, :, None]      # S x G x R
    const = np.sum(a * np.log(b) - gammaln(a) + gammaln(a + K))
    data = np.sum(np.where(k > 0, k * np.log(np.where(T > 0, T, 1.0)), 0.0) - gammaln(k + 1))
    ll = const + data - np.einsum('g,sgr->sr', a + K, np.log(E)) + KS[:, None] * logr[None, :]
    log_ml = logsumexp(ll, axis=1) - np.log(n_grid)
    log_ml[0] = ll[0, 0]                      # empty set does not depend on rho
    w = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
    rho_post = w @ rho
    lam_post = np.einsum('sr,sgr->sg', w, (a + K)[None, :, None] / E)
    log_prior = _log_prior_sets(sets_c, len(cand), s_max)
    return _finish(sets, log_ml, log_prior, member, rho_post, lam_post, groups)


def _nb_logpmf(k, mu, alpha):
    """log NB(k | mean mu, shape alpha), broadcasting; Var = mu + mu^2 / alpha."""
    return (gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1)
            + alpha * np.log(alpha / (alpha + mu)) + k * np.log(mu / (alpha + mu)))


def fit_hubs_het(k, T, group, prior_mean, alpha, *, a=1.0, rho_max=4.0, s_max=3, phi=1.0, n_grid=60,
                 n_lam=50, candidates=None, lam_mean=False, delta_range=None, fixed_lambda=False):
    """Heterogeneity-aware hub model. alpha: {group: gamma shape of individual rate multipliers}.

    For each group the base rate lambda_g is integrated numerically on an adaptive log grid; the
    integrand for any hub subset A of the group factorises into a no-hub part plus per-hub
    corrections, so all sets are scored with array sums. Set lam_mean=True to also get the
    posterior mean of each lambda_g (slower).

    delta_range=(lo, hi): additive excess instead of a multiplier. A hub's rate is lambda_g + delta
    (in the same units as the rates, e.g. expected goals per 90), with log delta ~ U(log lo, log hi]
    shared by all hubs: a hub adds at least `lo` per 90 beyond what is normal for his position
    (value above a typical player in his role). The returned rho_mean is then the posterior mean of delta.

    fixed_lambda=True: lambda_g is not estimated from the team but fixed at prior_mean[g] (the league-typical
    rate of the position = replacement level), so "normal" means "a typical La Liga player in that position".
    """
    k = np.asarray(k, float) / phi
    T = np.asarray(T, float) / phi
    group = np.asarray(group)
    n = len(k)
    groups = sorted(set(group))
    if delta_range is None:
        logr = (np.arange(n_grid) + 0.5) / n_grid * np.log(rho_max)
    else:
        lo, hi = np.log(delta_range[0]), np.log(delta_range[1])   # rates k/T are unchanged by the phi scaling
        logr = lo + (np.arange(n_grid) + 0.5) / n_grid * (hi - lo)
    rho = np.exp(logr)
    cand_mask = np.ones(n, bool) if candidates is None else np.asarray(candidates, bool)
    sub_scores = {}
    for g in groups:
        idx = np.flatnonzero(group == g)
        al = float(alpha[g])
        kg, Tg = k[idx], T[idx]
        bg = a / prior_mean[g]
        r_hat = (kg.sum() + a) / (Tg.sum() + bg)
        sd = np.sqrt(1.0 / (kg.sum() + a) + 1.0 / (al * len(idx)))
        if fixed_lambda:
            ll_grid = np.array([np.log(prior_mean[g])])
            lam = np.exp(ll_grid)
            lprior = np.zeros(1)
        else:
            ll_grid = np.linspace(np.log(r_hat) - 1.6 - 5 * sd, np.log(r_hat) + 5 * sd, n_lam)
            lam = np.exp(ll_grid)
            dlog = ll_grid[1] - ll_grid[0]
            lprior = a * np.log(bg) - gammaln(a) + a * ll_grid - bg * lam + np.log(dlog)     # L
        c = gammaln(kg + al) - gammaln(al) - gammaln(kg + 1) + al * np.log(al)        # n_g
        logmu0 = np.log(np.where(Tg > 0, Tg, 1e-300))[:, None] + ll_grid[None, :]      # n_g x L
        f0 = c[:, None] - (al + kg[:, None]) * np.logaddexp(np.log(al), logmu0) + kg[:, None] * logmu0
        if delta_range is None:
            logmu1 = logmu0[:, :, None] + logr[None, None, :]                          # n_g x L x R
        else:   # mean T * (lambda + delta); lambda grid is in scaled units (k/phi per T/phi), delta too
            logmu1 = (np.log(np.where(Tg > 0, Tg, 1e-300))[:, None, None]
                      + np.logaddexp(ll_grid[None, :, None], logr[None, None, :]))
        f1 = (c[:, None, None] - (al + kg[:, None, None]) * np.logaddexp(np.log(al), logmu1)
              + kg[:, None, None] * logmu1)
        base = lprior + f0.sum(axis=0)                                                  # L
        corr = f1 - f0[:, :, None]                                                      # n_g x L x R
        local = [i for i in range(len(idx)) if cand_mask[idx[i]]]
        subs = [()]
        for s in range(1, min(s_max, len(local)) + 1):
            subs.extend(combinations(local, s))
        sc = np.empty((len(subs), n_grid))
        lam_m = np.zeros((len(subs), n_grid))
        # score subsets in batches of equal size: tot = base + sum of the members' corrections
        start = 0
        for size in range(0, min(s_max, len(local)) + 1):
            n_size = sum(1 for x in subs[start:] if len(x) == size)
            idx_arr = np.array(subs[start:start + n_size], dtype=int).reshape(n_size, size)
            for b0 in range(0, n_size, 64):
                ia = idx_arr[b0:b0 + 64]
                tot = np.broadcast_to(base[None, :, None], (len(ia), len(ll_grid), n_grid)).copy()
                for c_ in range(size):
                    tot += corr[ia[:, c_]]
                rows_ = slice(start + b0, start + b0 + len(ia))
                mx = tot.max(axis=1)
                sc[rows_] = np.log(np.exp(tot - mx[:, None, :]).sum(axis=1)) + mx
                if lam_mean:
                    lam_m[rows_] = np.exp(logsumexp(tot + ll_grid[None, :, None], axis=1) - sc[rows_])
            start += n_size
        sub_scores[g] = ([tuple(int(idx[i]) for i in sub) for sub in subs], sc, lam_m)
    # combine groups: all combinations of subsets with total size <= s_max
    sets, rows, lam_rows = [()], [np.zeros(n_grid)], [np.zeros((len(groups), n_grid))]
    for gi, g in enumerate(groups):
        subs, sc, lam_m = sub_scores[g]
        new_sets, new_rows, new_lam = [], [], []
        for s0, v0, l0 in zip(sets, rows, lam_rows):
            for r, sub in enumerate(subs):
                if len(s0) + len(sub) <= s_max:
                    new_sets.append(s0 + sub)
                    new_rows.append(v0 + sc[r])
                    if lam_mean:
                        l1 = l0.copy()
                        l1[gi] = lam_m[r]
                        new_lam.append(l1)
                    else:
                        new_lam.append(l0)
        sets, rows, lam_rows = new_sets, new_rows, new_lam
    sets = [tuple(sorted(s)) for s in sets]
    ll = np.array(rows)
    log_ml = logsumexp(ll, axis=1) - np.log(n_grid)
    i0 = sets.index(())
    log_ml[i0] = ll[i0, 0]
    w = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
    rho_post = w @ rho
    lam_post = np.einsum('sr,sgr->sg', w, np.array(lam_rows)) if lam_mean else np.full((len(sets), len(groups)), np.nan)
    order = [i0] + [i for i in range(len(sets)) if i != i0]
    sets = [sets[i] for i in order]
    log_ml, rho_post, lam_post = log_ml[order], rho_post[order], lam_post[order]
    member = _membership(sets, n)
    n_cand = int(cand_mask.sum())
    sizes = np.array([len(s) for s in sets])
    s_eff = min(s_max, n_cand)
    log_prior = -np.log(s_eff + 1) - (gammaln(n_cand + 1) - gammaln(sizes + 1) - gammaln(n_cand - sizes + 1))
    return _finish(sets, log_ml, log_prior, member, rho_post, lam_post, groups)


def log_ml_het_set(k, T, group, prior_mean, alpha, hub_mask, *, a=1.0, rho_max=4.0, phi=1.0, n_grid=100,
                   n_lam=80):
    """log p(k | S) of the heterogeneity-aware model for one given hub set (any size)."""
    k = np.asarray(k, float) / phi
    T = np.asarray(T, float) / phi
    group = np.asarray(group)
    hub_mask = np.asarray(hub_mask, bool)
    logr = (np.arange(n_grid) + 0.5) / n_grid * np.log(rho_max)
    tot = np.zeros(n_grid)
    for g in sorted(set(group)):
        idx = np.flatnonzero(group == g)
        al = float(alpha[g])
        kg, Tg, hg = k[idx], T[idx], hub_mask[idx]
        bg = a / prior_mean[g]
        r_hat = (kg.sum() + a) / (Tg.sum() + bg)
        sd = np.sqrt(1.0 / (kg.sum() + a) + 1.0 / (al * len(idx)))
        ll_grid = np.linspace(np.log(r_hat) - 1.6 - 5 * sd, np.log(r_hat) + 5 * sd, n_lam)
        lam = np.exp(ll_grid)
        lprior = a * np.log(bg) - gammaln(a) + a * ll_grid - bg * lam + np.log(ll_grid[1] - ll_grid[0])
        c = gammaln(kg + al) - gammaln(al) - gammaln(kg + 1) + al * np.log(al)
        logmu = (np.log(np.where(Tg > 0, Tg, 1e-300))[:, None, None] + ll_grid[None, :, None]
                 + hg[:, None, None] * logr[None, None, :])                                  # n_g x L x R
        f = c[:, None, None] - (al + kg[:, None, None]) * np.logaddexp(np.log(al), logmu) + kg[:, None, None] * logmu
        tot += logsumexp(lprior[:, None] + f.sum(axis=0), axis=0)
    if not hub_mask.any():
        return float(tot[0])
    return float(logsumexp(tot) - np.log(n_grid))


def log_pred_individual(k1, T1, k2, T2, group, prior_mean, alpha, *, phi=1.0):
    """Saturated reference: each player has his own rate lambda_i ~ Gamma(alpha_g, alpha_g / prior_mean_g);
    returns log p(k2 | k1) summed over players (quasi-Poisson scaled)."""
    k1, T1, k2, T2 = (np.asarray(x, float) / phi for x in (k1, T1, k2, T2))
    al = np.array([alpha[g] for g in group], float)
    b = al / np.array([prior_mean[g] for g in group], float)
    A, B = al + k1, b + T1
    return float(np.sum(A * np.log(B) - gammaln(A) + gammaln(A + k2) - (A + k2) * np.log(B + T2)
                        + k2 * np.log(np.where(T2 > 0, T2, 1.0)) - gammaln(k2 + 1)))
