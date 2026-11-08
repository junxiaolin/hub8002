import numpy as np

from hub8002.chains import END, SHOT, START, chain_counts
from hub8002.models.net_model import NetUnit, chain_values, net_cores, replacement_from_w, replacement_rows


def _toy(rng, n=6):
    group = np.array(['GK', 'DEF', 'DEF', 'MID', 'MID', 'FWD'])[:n]
    C = rng.integers(0, 30, size=(n, n)).astype(float)
    np.fill_diagonal(C, 0)
    S = rng.integers(0, 8, size=n).astype(float)
    X = S * rng.uniform(0.05, 0.2, size=n)
    E = rng.integers(5, 20, size=n).astype(float)
    N0 = rng.integers(5, 40, size=n).astype(float)
    return NetUnit(np.arange(n), group, C, S, X, E, N0)


def test_sherman_morrison_matches_brute_force():
    rng = np.random.default_rng(1)
    u = _toy(rng)
    n = len(u.ids)
    q = {g: {'GK': .05, 'DEF': .3, 'MID': .35, 'FWD': .1, 'SHOT': .03, 'END': .17} for g in ('GK', 'DEF', 'MID', 'FWD')}
    gbar_i = np.full(n, 0.1)
    c = np.concatenate([u.C, u.S[:, None], u.E[:, None]], 1)
    rows = c / c.sum(1, keepdims=True)
    P, s = rows[:, :n], rows[:, n]
    g = np.where(u.S > 0, u.X / np.maximum(u.S, 1), 0.1)
    for mode in ('net', 'full'):
        V, m, _, _, rep = chain_values(P[None], s[None], g, u.N0, q, u.group, gbar_i, mode)
        rep = rep[0]
        # the average player's receiving weights are the expected receptions of the factual chain
        assert np.allclose(rep, replacement_from_w(m[0] @ P, u.group, q))
        base = u.N0 @ np.linalg.solve(np.eye(n) - P, s * g)
        for i in range(n):
            P2, s2, g2 = P.copy(), s.copy(), g.copy()
            if mode == 'net':
                P2[i] = (1 - s[i]) * rep[i, :n] / (1 - rep[i, n])
            else:
                P2[i], s2[i], g2[i] = rep[i, :n], rep[i, n], gbar_i[i]
            alt = u.N0 @ np.linalg.solve(np.eye(n) - P2, s2 * g2)
            assert np.isclose(V[0, i], base - alt, rtol=1e-8, atol=1e-10)


def test_replacement_from_w_matches_loop():
    rng = np.random.default_rng(4)
    u = _toy(rng)
    q = {g: {'GK': .05, 'DEF': .3, 'MID': .35, 'FWD': .1, 'SHOT': .03, 'END': .17} for g in ('GK', 'DEF', 'MID', 'FWD')}
    w = u.C.sum(0)
    R = replacement_rows(u, q)
    n = len(w)
    for i in range(n):
        ref = np.zeros(n + 2)
        for h in ('GK', 'DEF', 'MID', 'FWD'):
            mask = (u.group == h) & (np.arange(n) != i) & (w > 0)
            if mask.any():
                ref[:n][mask] += q[u.group[i]][h] * w[mask] / w[mask].sum()
        ref[n], ref[n + 1] = .03, .17
        assert np.allclose(R[i], ref / ref.sum())


def test_net_cores_finds_planted_linker():
    rng = np.random.default_rng(2)
    u = _toy(rng)
    # player 3 (MID) never loses the ball and feeds the forward a lot: large positive linking value
    u.E[3] = 0
    u.C[3] = 0
    u.C[3, 5] = 400
    q = {g: {'GK': .05, 'DEF': .3, 'MID': .35, 'FWD': .1, 'SHOT': .03, 'END': .17} for g in ('GK', 'DEF', 'MID', 'FWD')}
    gbar = {g: 0.1 for g in ('GK', 'DEF', 'MID', 'FWD')}
    alpha = {g: 50.0 for g in ('GK', 'DEF', 'MID', 'FWD')}
    f = net_cores(u, q, gbar, alpha, cand=u.group != 'GK', n_draws=400)
    assert f.top == 3 and f.p_top[3] == f.p_top.max() and f.voa_lo[3] > 0


def test_gamma_shape_zero_is_zero():
    assert np.random.default_rng(0).gamma(np.zeros(5)).sum() == 0.0


def _ev(idx, team, ptype, pid=None, poss_team=None, period=1, **extra):
    e = {'index': idx, 'period': period, 'team': {'name': team}, 'type': {'name': ptype},
         'possession_team': {'name': poss_team or team}}
    if pid is not None:
        e['player'] = {'id': pid}
    e.update(extra)
    return e


def test_chain_counts_spells_and_shots():
    ev = [_ev(1, 'A', 'Pass', 1), _ev(2, 'A', 'Ball Receipt*', 2), _ev(3, 'A', 'Carry', 2),
          _ev(4, 'B', 'Foul Committed', 9, poss_team='A'), _ev(5, 'A', 'Foul Won', 2),
          _ev(6, 'A', 'Pass', 3),                                      # free kick: same spell continues
          _ev(7, 'A', 'Shot', 4, shot={'type': {'name': 'Open Play'}, 'statsbomb_xg': 0.3}),
          _ev(8, 'A', 'Ball Recovery', 2),                             # rebound: a new chain, same spell
          _ev(9, 'A', 'Pass', 2), _ev(10, 'B', 'Interception', 7),      # B now in possession
          _ev(11, 'B', 'Pass', 7), _ev(12, 'B', 'Ball Receipt*', 8, ball_receipt={'outcome': {'name': 'Incomplete'}})]
    for e in ev[9:]:
        e['possession_team'] = {'name': 'B'}
    c = chain_counts(ev)
    assert c[('A', 1, START)][0] == 1 and c[('A', 1, 2)][0] == 1 and c[('A', 2, 3)][0] == 1
    assert c[('A', 3, 4)][0] == 1 and c[('A', 4, SHOT)] == [1, 0.3]
    assert c[('A', 2, START)][0] == 1 and c[('A', 2, END)][0] == 1
    assert c[('B', 7, START)][0] == 1 and c[('B', 7, END)][0] == 1 and ('B', 7, 8) not in c


def test_plugin_reproduces_observed_touches_and_xg_and_decomposition_adds_up():
    rng = np.random.default_rng(3)
    u = _toy(rng)
    inflow = u.C.sum(0)                     # make the toy flow-consistent, as real chains are:
    u.E += np.maximum(0, inflow - u.C.sum(1) - u.S) + 5      # touches = starts + receptions
    u.N0 = u.C.sum(1) + u.S + u.E - inflow
    q = {g: {'GK': .05, 'DEF': .3, 'MID': .35, 'FWD': .1, 'SHOT': .03, 'END': .17} for g in ('GK', 'DEF', 'MID', 'FWD')}
    gbar = {g: 0.1 for g in ('GK', 'DEF', 'MID', 'FWD')}
    f = net_cores(u, q, gbar, {g: 50.0 for g in gbar}, cand=u.group != 'GK', plugin=True, kappa=0.0)
    assert np.allclose(f.visits, u.touches)
    assert np.isclose(f.team_xg, u.X.sum())
    assert np.allclose(f.retain + f.choice, f.edge)


def test_baseline_option_defaults_and_shift():
    """voa_base / q_base = the prior centre reproduce the default; a lower node baseline shifts VOA by T*(lam-lam_b)."""
    from hub8002.models.core_model import value_cores
    k, T, g = np.array([3.0, 5.0, 1.0]), np.array([10.0, 12.0, 4.0]), np.array(['MID', 'FWD', 'MID'])
    rate, al = {'MID': 0.3, 'FWD': 0.5}, {'MID': 5.0, 'FWD': 5.0}
    a = value_cores(k, T, g, rate, al, n_draws=500)
    b = value_cores(k, T, g, rate, al, n_draws=500, voa_base=rate)
    assert np.allclose(a.voa_mean, b.voa_mean) and np.allclose(a.p_top, b.p_top)
    low = {'MID': 0.25, 'FWD': 0.4}
    c = value_cores(k, T, g, rate, al, n_draws=500, voa_base=low)
    assert np.allclose(c.voa_mean - a.voa_mean, T * np.array([0.05, 0.1, 0.05]))
    u = _toy(np.random.default_rng(3))
    q = {x: {'GK': .05, 'DEF': .3, 'MID': .35, 'FWD': .1, 'SHOT': .03, 'END': .17} for x in ('GK', 'DEF', 'MID', 'FWD')}
    gbar, al, cand = {x: 0.1 for x in q}, {x: 50.0 for x in q}, u.group != 'GK'
    f1 = net_cores(u, q, gbar, al, cand, n_draws=200)
    f2 = net_cores(u, q, gbar, al, cand, n_draws=200, q_base=q)
    assert np.allclose(f1.voa_mean, f2.voa_mean)
    worse = {x: {**r, 'END': .27, 'DEF': .2} for x, r in q.items()}  # replacement loses the ball more often
    f3 = net_cores(u, q, gbar, al, cand, n_draws=200, q_base=worse)
    assert np.all(f3.voa_mean[cand] > f1.voa_mean[cand])


def test_separate_receiving_blocks():
    """dst = group reproduces the default; with coarser receiving blocks each row puts q_g(h) on block h."""
    u = _toy(np.random.default_rng(4))
    q = {x: {'GK': .05, 'DEF': .3, 'MID': .35, 'FWD': .1, 'SHOT': .03, 'END': .17} for x in ('GK', 'DEF', 'MID', 'FWD')}
    a = replacement_rows(u, q)
    u2 = NetUnit(u.ids, u.group, u.C, u.S, u.X, u.E, u.N0, dst=u.group.copy())
    assert np.allclose(a, replacement_rows(u2, q))
    # fine comparison groups (CB vs FB) but receiving blocks GK / DEF / MID / FWD
    fine = np.array(['GK', 'CB', 'FB', 'MID', 'MID', 'FWD'])
    qf = {**{x: q['DEF'] for x in ('CB', 'FB')}, 'GK': q['GK'], 'MID': q['MID'], 'FWD': q['FWD']}
    qf['FB'] = {'GK': .02, 'DEF': .2, 'MID': .45, 'FWD': .1, 'SHOT': .03, 'END': .2}
    u3 = NetUnit(u.ids, fine, u.C, u.S, u.X, u.E, u.N0, dst=u.group.copy())
    r = replacement_rows(u3, qf)
    assert np.allclose(r.sum(1), 1)
    assert np.isclose(r[2, 3:5].sum(), 0.45)                    # FB -> MID block gets q_FB(MID)
