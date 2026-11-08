import numpy as np
from scipy import integrate
from scipy.special import gammaln

from hub8002.models.bayes_hub import fit_hubs, fit_hubs_het, enumerate_sets


def _log_ml_numeric(k, T, group, prior_mean, S, rho, a=1.0):
    """log p(k | S, rho) by numerically integrating each group's lambda."""
    tot = 0.0
    for g in sorted(set(group)):
        idx = [i for i in range(len(k)) if group[i] == g]
        b = a / prior_mean[g]

        def f(lam):
            mu = np.array([T[i] * lam * (rho if i in S else 1.0) for i in idx])
            kk = np.array([k[i] for i in idx])
            return np.exp(np.sum(kk * np.log(mu) - mu - gammaln(kk + 1))
                          + a * np.log(b) - gammaln(a) + (a - 1) * np.log(lam) - b * lam - shift)
        K = sum(k[i] for i in idx)
        mode = (K + a - 1) / (sum(T[i] for i in idx) + b)
        shift = 0.0
        shift = np.log(f(mode)) if f(mode) > 0 else 0.0
        val, _ = integrate.quad(f, 0, mode * 10, points=[mode], limit=200, epsabs=0, epsrel=1e-11)
        tot += np.log(val) + shift
    return tot


def test_closed_form_matches_numeric():
    rng = np.random.default_rng(0)
    k = rng.poisson(30, 7)
    T = rng.uniform(2, 10, 7)
    group = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'B'])
    prior = {'A': 4.0, 'B': 2.0}
    from scipy.special import logsumexp
    for S, rho in [((), 1.0), ((1,), 1.7), ((0, 4), 2.3)]:
        # closed form, written out
        a = 1.0
        cf = 0.0
        for g in prior:
            idx = [i for i in range(7) if group[i] == g]
            b = a / prior[g]
            K = sum(k[i] for i in idx)
            E = sum(T[i] * (rho if i in S else 1.0) for i in idx)
            cf += a * np.log(b) - gammaln(a) + gammaln(a + K) - (a + K) * np.log(b + E)
        cf += sum(k[i] * np.log(T[i] * (rho if i in S else 1.0)) - gammaln(k[i] + 1) for i in range(7))
        num = _log_ml_numeric(k, T, group, prior, S, rho)
        assert abs(cf - num) / abs(num) < 1e-6


def test_time_unit_invariance():
    rng = np.random.default_rng(1)
    k = rng.poisson(40, 12)
    T = rng.uniform(5, 30, 12)
    group = np.array(['DEF'] * 4 + ['MID'] * 4 + ['FWD'] * 4)
    prior = {'DEF': 30.0, 'MID': 40.0, 'FWD': 20.0}
    f1 = fit_hubs(k, T, group, prior)
    f2 = fit_hubs(k, T * 90, group, {g: v / 90 for g, v in prior.items()})
    assert f1.hubs == f2.hubs
    assert np.allclose(f1.incl_prob, f2.incl_prob, atol=1e-9)
    assert abs(f1.log_bf - f2.log_bf) < 1e-8


def _sim(rng, hub=None, rho=2.0, n=18):
    group = np.array(['DEF'] * 6 + ['MID'] * 6 + ['FWD'] * 6)
    base = {'DEF': 40.0, 'MID': 50.0, 'FWD': 25.0}
    T = rng.uniform(3, 38, n)
    lam = np.array([base[g] for g in group]) * T
    if hub is not None:
        lam[hub] *= rho
    return rng.poisson(lam), T, group, base


def test_null_false_positive_rate():
    rng = np.random.default_rng(2)
    fp = 0
    for _ in range(200):
        k, T, group, base = _sim(rng)
        fp += len(fit_hubs(k, T, group, base).hubs) > 0
    assert fp / 200 <= 0.05


def test_recovers_planted_hub():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(50):
        k, T, group, base = _sim(rng, hub=8, rho=2.0)
        T[8] = 30.0
        k[8] = rng.poisson(base['MID'] * 30 * 2.0)
        hits += fit_hubs(k, T, group, base).hubs == [8]
    assert hits >= 48


def test_het_model_reduces_to_poisson_for_large_alpha():
    rng = np.random.default_rng(4)
    k, T, group, base = _sim(rng, hub=3, rho=1.5)
    f1 = fit_hubs(k, T, group, base, n_grid=200)
    f2 = fit_hubs_het(k, T, group, base, {g: 1e7 for g in base}, n_grid=200, n_lam=400)
    assert f1.hubs == f2.hubs
    assert np.max(np.abs(f1.incl_prob - f2.incl_prob)) < 1e-3
    assert abs(f1.log_bf - f2.log_bf) < 0.05


def test_set_enumeration_size():
    assert len(enumerate_sets(20, 3)) == 1 + 20 + 190 + 1140
