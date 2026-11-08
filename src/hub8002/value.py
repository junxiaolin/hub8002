"""Value of actions (v2 of the plan): three kinds of contribution, all in expected-goal units.

att  (attacking core)  : non-penalty xG of own shots + xA (xG of the shots set up by the player's pass)
prog (organising core) : xT added by successful open-play passes and carries that start before the
                         final third (x < 80): moving the ball from build-up into dangerous areas
def  (defensive core)  : danger removed by ball-winning / danger-clearing actions (successful tackles
                         and interceptions, ball recoveries, blocks, clearances), valued at the
                         opponent's xT at that spot
Exposure: att and prog are rated per minute the player's team has the ball while he is on the pitch,
def per minute the opponent has the ball (possession-adjusted, as is standard for defensive stats).

xT (expected threat, Singh 2018) is fitted on all 866 matches: a 16 x 12 grid, shot and move
probabilities, mean StatsBomb xG of shots per cell and the transition matrix of successful moves.
StatsBomb gives every event in the acting team's frame (attacking towards x = 120), so the
opponent's position for a defensive action at (x, y) is (120 - x, 80 - y).
"""
import numpy as np

NX, NY = 16, 12
SET_PIECES = {'Corner', 'Free Kick', 'Throw-in', 'Goal Kick', 'Kick Off'}
WIN_OUTCOMES = {'Won', 'Success', 'Success In Play', 'Success Out'}


def cell(x, y):
    i = min(max(int(x / 120.0 * NX), 0), NX - 1)
    j = min(max(int(y / 80.0 * NY), 0), NY - 1)
    return i, j


def xt_counts(events):
    """Per-cell counts for fitting xT from one match (both teams)."""
    shots = np.zeros((NX, NY))
    xg = np.zeros((NX, NY))
    moves = np.zeros((NX, NY))
    trans = np.zeros((NX, NY, NX, NY))
    for e in events:
        t = e['type']['name']
        if 'location' not in e:
            continue
        i, j = cell(*e['location'][:2])
        if t == 'Shot':
            if e['shot']['type']['name'] == 'Penalty':
                continue
            shots[i, j] += 1
            xg[i, j] += e['shot'].get('statsbomb_xg', 0.0)
        elif t == 'Pass':
            p = e['pass']
            if (p.get('type') or {}).get('name') in SET_PIECES:
                continue
            moves[i, j] += 1
            if 'outcome' not in p:
                a, b = cell(*p['end_location'][:2])
                trans[i, j, a, b] += 1
        elif t == 'Carry':
            moves[i, j] += 1
            a, b = cell(*e['carry']['end_location'][:2])
            trans[i, j, a, b] += 1
    return shots, xg, moves, trans


def fit_xt(counts, n_iter=50):
    shots = sum(c[0] for c in counts)
    xg = sum(c[1] for c in counts)
    moves = sum(c[2] for c in counts)
    trans = sum(c[3] for c in counts)
    tot = shots + moves
    s = np.divide(shots, tot, out=np.zeros_like(tot), where=tot > 0)
    m = np.divide(moves, tot, out=np.zeros_like(tot), where=tot > 0)
    g = np.divide(xg, shots, out=np.zeros_like(xg), where=shots > 0)
    T = np.divide(trans, moves[:, :, None, None], out=np.zeros_like(trans), where=moves[:, :, None, None] > 0)
    xt = np.zeros((NX, NY))
    for _ in range(n_iter):
        xt = s * g + m * np.einsum('ijab,ab->ij', T, xt)
    return xt


def xt_at(xt, x, y):
    return float(xt[cell(x, y)])


def possession_segments(events, clock):
    """[(start, end, team)] in elapsed minutes: each event's time up to the next event, owned by possession_team."""
    seg = []
    evs = sorted((e for e in events if e['period'] <= 4 and 'possession_team' in e),
                 key=lambda e: (e['period'], e['timestamp'], e['index']))
    for k, e in enumerate(evs):
        t0 = clock.from_timestamp(e['timestamp'], e['period'])
        if k + 1 < len(evs) and evs[k + 1]['period'] == e['period']:
            t1 = clock.from_timestamp(evs[k + 1]['timestamp'], e['period'])
        else:
            p = e['period']
            t1 = clock.offset[p] + clock.lengths[p]
        if t1 > t0:
            seg.append((t0, t1, e['possession_team']['name']))
    return seg


def possession_exposure(intervals, segments, team):
    """Minutes the player's team (own) and the opponent (opp) had the ball while he was on the pitch.
    `segments` may be the list from possession_segments or a pre-built (s0, s1, team_names) array triple."""
    if isinstance(segments, list):
        segments = (np.array([s[0] for s in segments]), np.array([s[1] for s in segments]),
                    np.array([s[2] for s in segments]))
    s0, s1, tm = segments
    mine = tm == team
    own = opp = 0.0
    for a, b in intervals:
        ov = np.clip(np.minimum(s1, b) - np.maximum(s0, a), 0.0, None)
        own += float(ov[mine].sum())
        opp += float(ov[~mine].sum())
    return own, opp


def action_values(events, xt):
    """{(team, player_id): {'att': .., 'prog': .., 'def': .., 'npxg': .., 'xa': ..}} for one match."""
    out = {}
    shot_by_keypass = {}
    for e in events:
        if e['type']['name'] == 'Shot' and e['shot']['type']['name'] != 'Penalty' and 'key_pass_id' in e['shot']:
            shot_by_keypass[e['shot']['key_pass_id']] = e['shot'].get('statsbomb_xg', 0.0)

    def add(e, key, v):
        d = out.setdefault((e['team']['name'], e['player']['id']),
                           {'att': 0.0, 'prog': 0.0, 'def': 0.0, 'npxg': 0.0, 'xa': 0.0})
        d[key] += v

    for e in events:
        t = e['type']['name']
        if 'player' not in e or 'location' not in e:
            continue
        x, y = e['location'][:2]
        if t == 'Shot':
            if e['shot']['type']['name'] != 'Penalty':
                v = e['shot'].get('statsbomb_xg', 0.0)
                add(e, 'att', v)
                add(e, 'npxg', v)
        elif t == 'Pass':
            p = e['pass']
            if e['id'] in shot_by_keypass:
                add(e, 'att', shot_by_keypass[e['id']])
                add(e, 'xa', shot_by_keypass[e['id']])
            if 'outcome' not in p and (p.get('type') or {}).get('name') not in SET_PIECES and x < 80:
                gain = xt_at(xt, *p['end_location'][:2]) - xt_at(xt, x, y)
                if gain > 0:
                    add(e, 'prog', gain)
        elif t == 'Carry':
            if x < 80:
                gain = xt_at(xt, *e['carry']['end_location'][:2]) - xt_at(xt, x, y)
                if gain > 0:
                    add(e, 'prog', gain)
        else:
            won = False
            if t == 'Interception':
                won = (e['interception'].get('outcome') or {}).get('name') in WIN_OUTCOMES
            elif t == 'Duel':
                d = e['duel']
                won = (d.get('type') or {}).get('name') == 'Tackle' and (d.get('outcome') or {}).get('name') in WIN_OUTCOMES
            elif t == 'Ball Recovery':
                won = not (e.get('ball_recovery') or {}).get('recovery_failure', False)
            elif t in ('Block', 'Clearance'):
                won = True
            if won:
                add(e, 'def', xt_at(xt, 120.0 - x, 80.0 - y))
    return out
