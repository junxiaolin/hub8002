"""Network extension of v2: value over an average player of the same position, computed on the
possession network ("network VOA").

Markov chain over ball holders of one team-period (chains.py). Holder i's next step is one of:
pass/move to team-mate j (prob P_ij), shot (s_i, worth g_i expected goals on average) or chain end.
With M = (I - P)^-1:

    v = M (s * g)      expected goals of a chain that is now at each player
    m = N0^T M         expected touches (visits) of each player, N0 = chain starts
    team total         = N0^T v

Counterfactual: replace player i's behaviour on the ball by that of a league-average player of his
position (same slot: team-mates keep passing to him as often). Row i of P changes by d_i = p'_i - p_i,
a rank-one change, so by Sherman-Morrison

    VOA_i = total - total' = m_i * ((p_i - p'_i) . v + shot term) / (1 + (p_i - p'_i) . M[:, i])

i.e. touches x expected goals added per touch, with a feedback correction. Two versions:

  'net'  (linking core) : keep his own shooting, replace only what he does with the ball when he does
                          not shoot (keep it or lose it, and to whom he gives it). Shot term = 0.
  'full'                : replace everything incl. shooting (shot term = s_i g_i - s'_i gbar_g).

Average player of position g (from La Liga 2015/16): probabilities q_g over {pass to a GK / DEF / MID /
FWD team-mate, shoot, end}; within a destination group the mass is shared among the team's players in
proportion to how often they receive the ball (expected receptions under the chain, so that estimate
and estimand use the same definition). Uncertainty: row i ~ Dirichlet(alpha_g * pbar_i) a priori
(pbar_i = the average-player row in this team), alpha_g by empirical Bayes (Dirichlet-multinomial
marginal likelihood, the model family of Kirkley's MDL encoding, see docs/mdl_relation.md); counts are
quasi-multinomial (divided by phi). xG per shot: each shot's xG ~ Gamma(a, a / g_i) with the shape a
estimated from all shots, prior worth kappa shots at the positional mean. Posterior draws of all rows
and g give P(#1) and the core set exactly as in core_model.value_cores.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

DEST = ('GK', 'DEF', 'MID', 'FWD')
SHOT, END, START = -1, -2, -3


@dataclass
class NetUnit:
    ids: np.ndarray       # (n,) player ids (everyone who touched the ball)
    group: np.ndarray     # (n,) position group: whom each holder is compared with (GK / DEF / MID / FWD ...)
    C: np.ndarray         # (n, n) transitions i -> j
    S: np.ndarray         # (n,) shots
    X: np.ndarray         # (n,) xG of those shots
    E: np.ndarray         # (n,) chain ends
    N0: np.ndarray        # (n,) chain starts
    dst: np.ndarray = None  # (n,) receiving block of each player in the null model (default: group)

    @property
    def blocks(self):
        return self.group if self.dst is None else self.dst

    @property
    def touches(self):
        return self.C.sum(1) + self.S + self.E


def build_unit(chains, team, match_ids, groups, dst_groups=None):
    """chains: table from scripts/build_chains.py; groups: {player_id: group}; dst_groups: {player_id:
    receiving block} if the null model's receiving blocks differ from the comparison groups."""
    d = chains[(chains.team == team) & chains.match_id.isin(set(match_ids))]
    g = d.groupby(['i', 'j'])[['n', 'xg']].sum().reset_index()
    ids = np.array(sorted(set(g.i) | set(g.j[g.j >= 0])))
    idx = {p: k for k, p in enumerate(ids)}
    n = len(ids)
    C, S, X, E, N0 = np.zeros((n, n)), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i, j, c, x in zip(g.i, g.j, g.n, g.xg):
        a = idx[i]
        if j >= 0:
            C[a, idx[j]] += c
        elif j == SHOT:
            S[a] += c
            X[a] += x
        elif j == END:
            E[a] += c
        else:
            N0[a] += c
    grp = np.array([groups.get(int(p), 'MID') for p in ids])
    dst = None if dst_groups is None else np.array([dst_groups.get(int(p), 'MID') for p in ids])
    return NetUnit(ids, grp, C, S, X, E, N0, dst)


def category_counts(u, dest=DEST):
    """(n, len(dest) + 2) counts per holder over dest + (SHOT, END): team-mates collapsed to groups."""
    out = np.zeros((len(u.ids), len(dest) + 2))
    for h, name in enumerate(dest):
        out[:, h] = u.C[:, u.blocks == name].sum(1)
    out[:, -2], out[:, -1] = u.S, u.E
    return out


def league_profiles(units, dest=DEST, src=None):
    """q[g] (dict over dest + SHOT, END) and gbar[g] (xG per shot) pooled over league units, for every
    comparison group g in src (default: the same names as the receiving blocks dest)."""
    q, gbar = {}, {}
    dest = tuple(dest)
    for gname in (dest if src is None else tuple(src)):
        tot, s, x = np.zeros(len(dest) + 2), 0.0, 0.0
        for u in units:
            m = u.group == gname
            tot += category_counts(u, dest)[m].sum(0)
            s += u.S[m].sum()
            x += u.X[m].sum()
        q[gname] = dict(zip(dest + ('SHOT', 'END'), (tot / tot.sum()).tolist()))
        gbar[gname] = float(x / s) if s > 0 else 0.0
    return q, gbar


def _dest_setup(group, q, dst=None):
    dests = [k for k in q[group[0]] if k not in ('SHOT', 'END')]
    hidx = np.array([dests.index(x) for x in (group if dst is None else dst)])
    Q = np.array([[q[x][h] for h in dests] for x in group])
    qs = np.array([q[x]['SHOT'] for x in group])
    qe = np.array([q[x]['END'] for x in group])
    return hidx, Q, qs, qe, len(dests)


def replacement_from_w(w, group, q, dst=None):
    """Average-player rows (..., n, n + 2) given reception weights w (..., n): the pass mass q_g(h) of each
    destination group h is shared among the team's players of h (other than i) in proportion to w.
    group: comparison group of each holder (row of q); dst: receiving block of each player (default group)."""
    hidx, Q, qs, qe, H = _dest_setup(group, q, dst)
    n = len(group)
    w = np.asarray(w, float)
    T = w @ np.eye(H)[hidx]                                   # (..., H) group totals
    same = (hidx[:, None] == hidx[None, :]).astype(float)
    denom = T[..., None, hidx] - w[..., :, None] * same       # sum over k in h(j), k != i
    frac = np.divide(np.broadcast_to(w[..., None, :], denom.shape), denom, out=np.zeros_like(denom),
                     where=denom > 1e-12) * (1.0 - np.eye(n))
    dest = Q[:, hidx] * frac
    shp = dest.shape[:-1] + (1,)
    out = np.concatenate([dest, np.broadcast_to(qs[:, None], shp), np.broadcast_to(qe[:, None], shp)], -1)
    return out / out.sum(-1, keepdims=True)


def replacement_rows(u, q, weight='receptions'):
    """(n, n + 2) rows of an average player of each holder's position, placed in this team, using the
    observed receptions ('receptions') or equal shares among players who received the ball ('uniform')."""
    w = u.C.sum(0)
    return replacement_from_w(w if weight == 'receptions' else (w > 0).astype(float), u.group, q, u.dst)


def _counts(u):
    return np.concatenate([u.C, u.S[:, None], u.E[:, None]], axis=1)


def dm_loglik(c, a):
    """Dirichlet-multinomial log marginal likelihood of count rows c under parameters a (same shape)."""
    pos = a > 0
    A, N = a.sum(1), c.sum(1)
    t = np.where(pos, gammaln(np.where(pos, a + c, 1.0)) - gammaln(np.where(pos, a, 1.0)), 0.0)
    return float(np.sum(gammaln(A) - gammaln(A + N) + t.sum(1)))


def eb_alpha(units, q, group, phi=1.0):
    """Empirical-Bayes concentration alpha for holders of one position group."""
    rows = []
    for u in units:
        m = u.group == group
        if m.any():
            rows.append((_counts(u)[m] / phi, replacement_rows(u, q)[m]))
    res = minimize_scalar(lambda la: -sum(dm_loglik(c, np.exp(la) * p) for c, p in rows),
                          bounds=(np.log(1.0), np.log(1e5)), method='bounded')
    return float(np.exp(res.x))


def pearson_phi(match_units, q):
    """Quasi-multinomial scale: Pearson chi-square of each player's per-match category counts around his
    season proportions, divided by its degrees of freedom. match_units: list of lists of per-match
    category-count arrays {player: [(6,), ...]}."""
    chi, df = 0.0, 0
    for per_player in match_units:
        for rows in per_player.values():
            R = np.array(rows)
            R = R[R.sum(1) > 0]
            if len(R) < 2:
                continue
            p = R.sum(0) / R.sum()
            k = p > 0
            exp = R.sum(1, keepdims=True) * p[None, :]
            chi += float(np.sum((R[:, k] - exp[:, k]) ** 2 / exp[:, k]))
            df += (len(R) - 1) * (int(k.sum()) - 1)
    return chi / df if df else 1.0


@dataclass
class NetCores:
    voa_mean: np.ndarray
    voa_lo: np.ndarray
    voa_hi: np.ndarray
    p_pos: np.ndarray
    p_top: np.ndarray       # over candidates only (0 for others)
    visits: np.ndarray      # posterior mean expected touches
    edge: np.ndarray        # posterior mean expected goals added per touch (before feedback correction)
    core_set: list
    top: int
    clear: bool
    team_xg: float          # posterior mean expected goals of all chains (N0 . v)
    retain: np.ndarray      # per-touch value from keeping the ball (vs the average player), posterior-mean rows
    choice: np.ndarray      # per-touch value from whom he gives it to


def chain_values(P, s, g, N0, q, group, gbar_i, mode, w_fixed=None, dst=None):
    """VOA of every holder for a batch of chains. P (d, n, n), s (d, n), g (d, n) or (n,).
    q: profile of the player one compares with (league average of the position, or the replacement level).
    The average player's receiving weights are the expected receptions under each chain (w_fixed: use
    these weights instead). Returns VOA (d, n), expected touches m, per-touch edge, team xG, p' rows."""
    n = P.shape[-1]
    g = np.broadcast_to(g, s.shape)
    M = np.linalg.inv(np.eye(n)[None] - P)
    v = (M @ (s * g)[..., None])[..., 0]
    m = np.einsum('dk,dki->di', np.broadcast_to(N0, s.shape), M)
    w = np.einsum('di,dij->dj', m, P) if w_fixed is None else np.broadcast_to(w_fixed, s.shape)
    rep = replacement_from_w(w, group, q, dst)                # (d, n, n + 2)
    if mode == 'net':
        D = P - (1.0 - s)[..., None] * rep[..., :n] / (1.0 - rep[..., n:n + 1])
        shot = 0.0
    else:
        D = P - rep[..., :n]
        shot = s * g - rep[..., n] * gbar_i[None]
    num = np.einsum('dik,dk->di', D, v) + shot
    den = 1.0 + np.einsum('dik,dki->di', D, M)
    return m * num / den, m, num, (np.broadcast_to(N0, s.shape) * v).sum(1), rep


def decompose(rows, rep, g):
    """Per-touch linking value split into keeping the ball and choosing the receiver (rows, rep: (n, n+2)).
    p_i - p'_i = (1-s)[(1-e)(pi - pibar) + ((1-e) - (1-ebar)) pibar], e = end rate given no shot,
    pi = receiver distribution given a pass; bars = the average player of his position."""
    n = rows.shape[0]
    P, s, E = rows[:, :n], rows[:, n], rows[:, n + 1]
    v = np.linalg.solve(np.eye(n) - P, s * g)
    e = np.divide(E, 1.0 - s, out=np.zeros_like(E), where=(1.0 - s) > 0)
    pi = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    ebar = rep[:, n + 1] / (1.0 - rep[:, n])
    pib = rep[:, :n] / rep[:, :n].sum(1, keepdims=True)
    retain = (1.0 - s) * (ebar - e) * (pib @ v)
    choice = (1.0 - s) * (1.0 - e) * ((pi - pib) @ v)
    return retain, choice


def net_cores(u, q, gbar, alpha, cand, *, phi=1.0, mode='net', alpha_mult=1.0, kappa=5.0, shot_shape=0.6,
              cred=0.9, pos_prob=0.9, max_set=3, n_draws=2000, chunk=250, seed=0, plugin=False, repl='receptions',
              q_base=None):
    """Posterior network VOA for every holder; P(#1) and core set among `cand` (bool mask).
    q is the league-average profile (prior centre); q_base the profile one compares with (default q;
    e.g. the replacement level = the league's bench players of the position)."""
    qb = q if q_base is None else q_base
    n = len(u.ids)
    cand = np.asarray(cand, bool)
    ci = np.flatnonzero(cand)
    w_fixed = None if repl == 'receptions' else (u.C.sum(0) > 0).astype(float)
    pbar = replacement_rows(u, q, repl)
    gbar_i = np.array([gbar[x] for x in u.group])
    ga, gb = shot_shape * (u.S + kappa) + 1.0, u.X + kappa * gbar_i       # posterior of a / g: Gamma(ga, gb)
    g_mean = np.divide(shot_shape * gb, ga - 1.0, out=np.zeros(n), where=ga > 1.0)
    c = _counts(u) / phi
    if plugin:                                      # raw rows (no shrinkage); unseen players get pbar
        tot = c.sum(1, keepdims=True)
        rows = np.where(tot > 0, c / np.where(tot > 0, tot, 1.0), pbar)
        V, m, num, tx, rep = chain_values(rows[None, :, :n], rows[None, :, n], g_mean, u.N0, qb, u.group, gbar_i,
                                          mode, w_fixed, u.dst)
        voa = V[0]
        top = int(ci[np.argmax(voa[ci])])
        z = np.zeros(n)
        z[top] = 1.0
        ret, cho = decompose(rows, rep[0], g_mean)
        return NetCores(voa, voa, voa, (voa > 0).astype(float), z, m[0], num[0], [top], top, True, float(tx[0]),
                        ret, cho)
    al = np.array([alpha[x] for x in u.group]) * alpha_mult
    A = al[:, None] * pbar + c
    rng = np.random.default_rng(seed)
    Vs, ms, nums, txs = [], [], [], []
    for start in range(0, n_draws, chunk):
        d = min(chunk, n_draws - start)
        G = rng.gamma(A[None], size=(d, n, n + 2))
        rows = G / G.sum(-1, keepdims=True)
        beta = rng.gamma(ga[None], size=(d, n)) / np.where(gb > 0, gb, 1.0)[None]
        g = np.where(gb[None] > 0, shot_shape / np.maximum(beta, 1e-300), 0.0)
        V, m, num, tx, _ = chain_values(rows[..., :n], rows[..., n], g, u.N0, qb, u.group, gbar_i, mode, w_fixed, u.dst)
        Vs.append(V)
        ms.append(m)
        nums.append(num)
        txs.append(tx)
    V, m, num, tx = (np.concatenate(x) for x in (Vs, ms, nums, txs))
    winners = ci[np.argmax(V[:, ci], axis=1)]
    p_top = np.bincount(winners, minlength=n) / len(V)
    p_pos = (V > 0).mean(0)
    core, acc = [], 0.0
    for i in np.argsort(-p_top):
        if acc >= cred or p_top[i] == 0:
            break
        core.append(int(i))
        acc += p_top[i]
    if len(core) > max_set or any(p_pos[i] < pos_prob for i in core):
        core = []          # a possible #1 whose value is not credibly positive: no clear core
    vm = V.mean(0)
    rmean = A / A.sum(1, keepdims=True)
    _, _, _, _, rep = chain_values(rmean[None, :, :n], rmean[None, :, n], g_mean, u.N0, qb, u.group, gbar_i, mode,
                                   w_fixed, u.dst)
    ret, cho = decompose(rmean, rep[0], g_mean)
    return NetCores(vm, np.percentile(V, 5, 0), np.percentile(V, 95, 0), p_pos, p_top, m.mean(0), num.mean(0),
                    sorted(core, key=lambda i: -p_top[i]), int(ci[np.argmax(vm[ci])]), len(core) == 1,
                    float(tx.mean()), ret, cho)
