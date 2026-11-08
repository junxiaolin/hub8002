import numpy as np

from hub8002.models.core_model import eb_alpha, value_cores
from hub8002.value import fit_xt, xt_counts


def test_eb_alpha_recovers_spread():
    rng = np.random.default_rng(0)
    T = rng.uniform(5, 35, 4000)
    u = rng.gamma(8.0, 1 / 8.0, 4000)
    k = rng.poisson(T * 2.0 * u)
    a = eb_alpha(k, T, 2.0, 1.0)
    assert 6.5 < a < 10.0


def test_value_cores_picks_clear_star_and_is_scale_free():
    rng = np.random.default_rng(1)
    g = np.array(['FWD'] * 4 + ['MID'] * 4)
    T = np.full(8, 30.0)
    rate = {'FWD': 0.8, 'MID': 0.3}
    k = np.array([0.8, 0.8, 0.8, 2.0, 0.3, 0.3, 0.3, 0.3]) * T
    f = value_cores(k, T, g, rate, {'FWD': 10.0, 'MID': 10.0}, phi=0.25)
    assert f.top == 3 and f.core_set == [3] and f.clear
    # rescaling the value unit (and phi with it) leaves the decision unchanged
    f2 = value_cores(k * 10, T, g, {kk: v * 10 for kk, v in rate.items()}, {'FWD': 10.0, 'MID': 10.0}, phi=2.5)
    assert f2.core_set == [3]
    assert np.allclose(f2.voa_mean, 10 * f.voa_mean)


def test_xt_increases_towards_goal():
    events = []
    xs = (20, 60, 100, 110)
    for x, nxt in zip(xs, xs[1:] + (110,)):
        for _ in range(50):
            events.append({'type': {'name': 'Pass'}, 'location': [x, 40], 'pass': {'end_location': [nxt, 40]}})
        events.append({'type': {'name': 'Shot'}, 'location': [x, 40],
                       'shot': {'type': {'name': 'Open Play'}, 'statsbomb_xg': {20: 0.01, 60: 0.02, 100: 0.15, 110: 0.3}[x]}})
    xt = fit_xt([xt_counts(events)])
    assert xt[14, 6] > xt[13, 6] > xt[8, 6] > xt[2, 6] > 0
