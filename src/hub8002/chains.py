"""Possession chains as a Markov chain over ball holders (network extension of v2).

A *control spell* is a maximal run of events in which the same team has the ball (StatsBomb starts a
new `possession` at every restart, e.g. a free kick after a foul won, so consecutive possessions of
the same team are merged). Within a spell, the team's on-ball events are collapsed into the sequence
of distinct ball holders h1 -> h2 -> ... Each step from one holder to the next is a transition
(usually a completed pass, sometimes a loose ball won back by a team-mate). A chain ends

  - with a SHOT (non-penalty; carries its StatsBomb xG); events after the shot that are still in the
    same spell (rebound won back) start a new chain;
  - with END when the spell ends (the opponent gets the ball, or the half ends).

Per match and team we count, for every holder i: chain starts at i, transitions i -> j, shots by i
(and their xG) and chain ends at i. Summed over any set of matches these are the sufficient
statistics of the Markov chain used in models/net_model.py.
"""
from collections import defaultdict

HOLD = {'Pass', 'Ball Receipt*', 'Carry', 'Dribble', 'Shot', 'Ball Recovery', 'Dispossessed',
        'Miscontrol', 'Foul Won', 'Goal Keeper', 'Clearance', 'Interception', '50/50', 'Shield'}
SHOT, END, START = -1, -2, -3          # codes in column j of the chain table


def is_holding(e):
    """Is this event a touch of the ball by a player of the team in possession?"""
    if 'player' not in e or 'possession_team' not in e:
        return False
    if e['team']['name'] != e['possession_team']['name']:
        return False
    t = e['type']['name']
    if t not in HOLD:
        return False
    if t == 'Ball Receipt*' and 'outcome' in (e.get('ball_receipt') or {}):
        return False                                   # failed reception
    if t == 'Ball Recovery' and (e.get('ball_recovery') or {}).get('recovery_failure', False):
        return False
    if t == 'Shot' and e['shot']['type']['name'] == 'Penalty':
        return False                                   # v2 works with non-penalty xG
    return True


def chain_counts(events):
    """-> dict {(team, i, j): [n, xg]} with j a player id or SHOT / END / START."""
    out = defaultdict(lambda: [0, 0.0])
    evs = sorted((e for e in events if e['period'] <= 4 and 'possession_team' in e),
                 key=lambda e: (e['period'], e['index']))
    team = per = prev = None

    def close():
        if prev is not None:
            out[(team, prev, END)][0] += 1

    for e in evs:
        pt = e['possession_team']['name']
        if pt != team or e['period'] != per:
            close()
            team, per, prev = pt, e['period'], None
        if not is_holding(e):
            continue
        pid = e['player']['id']
        if prev is None:
            out[(team, pid, START)][0] += 1
        elif pid != prev:
            out[(team, prev, pid)][0] += 1
        prev = pid
        if e['type']['name'] == 'Shot':
            c = out[(team, pid, SHOT)]
            c[0] += 1
            c[1] += float(e['shot'].get('statsbomb_xg', 0.0))
            prev = None
    close()
    return dict(out)
