"""v2 core model: who is most valuable to the team in attack, organisation and defence?

For one team-period and one dimension, player i has value k_i (expected goals) and exposure T_i
(minutes of the relevant possession phase on the pitch, in units of 90). His true rate is
r_i = lam_g * u_i, where lam_g is the league-average rate of his position (2015/16 La Liga) and
u_i ~ Gamma(alpha_g, alpha_g) the normal spread of players in that position (empirical Bayes).
Values are quasi-Poisson, (k, T) -> (k / phi, T / phi), so the posterior is conjugate:

    r_i | data ~ Gamma(alpha_g + k_i / phi,  alpha_g / lam_g + T_i / phi)

Value over an average player of his position (VOA): VOA_i = T_i * (r_i - lam_g), in expected goals over
the period. (Like baseball's WAR it is a value over a baseline, but WAR's baseline is a replacement-level
player; the replacement-level version below is the closer analogue.) Variant (2026-11-08):
value over a replacement-level player, T_i * (r_i - lam_rep_g), lam_rep_g = the rate of the league's
bench players of that position (`voa_base`); the prior is unchanged. Shrinkage matters:
a player with few minutes and a lucky streak is pulled back towards the positional average.

Decision (all from posterior draws):
  top core   : the player with the largest posterior mean VOA; p_top = P(he has the team's largest VOA)
  core set   : the smallest set of players whose probabilities of being the team's #1 add up to >= cred
               (default 0.9); reported only if every member has P(VOA > 0) >= pos_prob (default 0.9) and
               it has at most max_set players. One name = a clear core; two or three = a shared core;
               otherwise no clear core. (Rule revised 2026-11-08: dropping the members that fail the
               P(VOA > 0) test, as before, could leave one name whose P(#1) is far below 0.9.)
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar


@dataclass
class ValueCores:
    voa_mean: np.ndarray
    voa_lo: np.ndarray         # 5% posterior quantile
    voa_hi: np.ndarray         # 95%
    p_pos: np.ndarray          # P(VOA > 0)
    p_top: np.ndarray          # P(largest VOA in the team)
    rate_mean: np.ndarray      # posterior mean rate per 90
    core_set: list             # indices (empty if no clear core)
    top: int
    clear: bool                # core set has exactly one player


def posterior_params(k, T, group, base_rate, alpha, phi):
    k = np.asarray(k, float) / phi
    T = np.asarray(T, float) / phi
    lam = np.array([base_rate[g] for g in group], float)
    al = np.array([alpha[g] for g in group], float)
    return al + k, al / lam + T, lam


def value_cores(k, T, group, base_rate, alpha, *, phi=1.0, cred=0.9, pos_prob=0.9, max_set=3,
                n_draws=4000, seed=0, voa_base=None):
    """voa_base: rates of the player one compares with ({group: rate}); default = base_rate (the league
    average of the position, which is also the prior mean). E.g. the replacement level (bench players)."""
    k, T, group = np.asarray(k, float), np.asarray(T, float), np.asarray(group)
    A, B, lam = posterior_params(k, T, group, base_rate, alpha, phi)
    if voa_base is not None:
        lam = np.array([voa_base[g] for g in group], float)
    rng = np.random.default_rng(seed)
    r = rng.gamma(A[:, None], 1.0 / B[:, None], size=(len(k), n_draws))
    V = T[:, None] * (r - lam[:, None])
    winners = np.argmax(V, axis=0)
    p_top = np.bincount(winners, minlength=len(k)) / n_draws
    p_pos = (V > 0).mean(axis=1)
    order = np.argsort(-p_top)
    core, acc = [], 0.0
    for i in order:
        if acc >= cred:
            break
        core.append(int(i))
        acc += p_top[i]
    if len(core) > max_set or any(p_pos[i] < pos_prob for i in core):
        core = []          # a possible #1 whose value is not credibly positive: no clear core
    voa_mean = T * (A / B - lam)
    return ValueCores(voa_mean=voa_mean, voa_lo=np.percentile(V, 5, axis=1), voa_hi=np.percentile(V, 95, axis=1),
                      p_pos=p_pos, p_top=p_top, rate_mean=A / B, core_set=sorted(core, key=lambda i: -p_top[i]),
                      top=int(np.argmax(voa_mean)), clear=len(core) == 1)


def nb_loglik(k, T, lam, alpha):
    """Marginal NB log-likelihood of scaled counts k with mean T * lam and shape alpha."""
    mu = T * lam
    return np.sum(gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1)
                  + alpha * np.log(alpha / (alpha + mu)) + k * np.log(np.where(mu > 0, mu, 1e-300) / (alpha + mu)))


def eb_alpha(k, T, lam, phi):
    """Empirical-Bayes shape alpha (maximum marginal likelihood) for one position group."""
    k, T = np.asarray(k, float) / phi, np.asarray(T, float) / phi
    res = minimize_scalar(lambda la: -nb_loglik(k, T, lam, np.exp(la)), bounds=(np.log(0.1), np.log(1e4)),
                          method='bounded')
    return float(np.exp(res.x))
