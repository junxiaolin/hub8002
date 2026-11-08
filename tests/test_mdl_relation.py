"""Numerical check of the proposition in docs/mdl_relation.md:

Kirkley's weighted ER description length, as a function of the hub count h, equals
    log N + log M + log C(N, h) - sum_G log DirMult(k_G | K_G; alpha = N)  + const
and the CM length equals the same with alpha = 1 for the hub group, where G runs over
{hubs, non-hubs} and the constant (sum_i log MS(N, k_i)) does not depend on h.
"""
import numpy as np
from scipy.special import gammaln

from hub8002.models.mdl import logchoose, logmultiset


def log_dirmult(k, alpha):
    k = np.asarray(k, float)
    K, g = k.sum(), len(k)
    return (gammaln(K + 1) + gammaln(g * alpha) - gammaln(K + g * alpha)
            + np.sum(gammaln(k + alpha) - gammaln(k + 1) - gammaln(alpha)))


def kirkley_lengths(deg):
    """ER and CM description lengths for every h (hubs = top-h degrees), as in the official code."""
    N, M = len(deg), int(sum(deg))
    order = np.argsort(-np.asarray(deg), kind='stable')
    er, cm = {}, {}
    for h in range(1, N):
        hub = [deg[i] for i in order[:h]]
        Mh = sum(hub)
        base = np.log(N) + np.log(M) + logchoose(N, h)
        er[h] = base + logmultiset(h * N, Mh) + logmultiset((N - h) * N, M - Mh)
        cm[h] = base + logmultiset(h, Mh) + sum(logmultiset(N, k) for k in hub) + logmultiset((N - h) * N, M - Mh)
    return er, cm, order


def test_er_and_cm_are_dirichlet_multinomial_codes():
    rng = np.random.default_rng(0)
    for _ in range(50):
        N = int(rng.integers(6, 25))
        deg = [int(x) for x in rng.negative_binomial(3, 0.01, N) + 1]
        er, cm, order = kirkley_lengths(deg)
        const = sum(logmultiset(N, k) for k in deg)
        M = sum(deg)
        d_er, d_cm = [], []
        for h in er:
            hub = [deg[i] for i in order[:h]]
            non = [deg[i] for i in order[h:]]
            base = np.log(N) + np.log(M) + logchoose(N, h)
            d_er.append(er[h] - (base - log_dirmult(hub, N) - log_dirmult(non, N) + const))
            d_cm.append(cm[h] - (base - log_dirmult(hub, 1.0) - log_dirmult(non, N) + const))
        assert np.allclose(d_er, 0.0, atol=1e-7)
        assert np.allclose(d_cm, 0.0, atol=1e-7)
