"""Playing intervals, minutes and positions from StatsBomb lineup files.

Time axis: "elapsed minutes" from kick-off, with each half's real length (stoppage time included),
taken from the Half End events. Lineup times are match-clock strings ("MM:SS"); the clock restarts
at 45:00 in the 2nd half, 90:00 in extra time period 3 and 105:00 in period 4.

Red cards are NOT reflected in the lineup position spells (the sent-off player's spell runs to the
final whistle), so every player's intervals are cut at the time of a 'Red Card' / 'Second Yellow'
in the lineup's `cards` field.
"""
from collections import defaultdict

PERIOD_START_CLOCK = {1: 0.0, 2: 45.0, 3: 90.0, 4: 105.0, 5: 120.0}
SENDING_OFF = ('Red Card', 'Second Yellow')


def clock_to_min(s):
    mm, ss = s.split(':')
    return int(mm) + int(ss) / 60.0


def period_lengths(events):
    """Length (minutes) of each period, from Half End events (max over the two teams)."""
    ends = defaultdict(float)
    for e in events:
        if e['type']['name'] == 'Half End':
            p = e['period']
            t = e['minute'] + e['second'] / 60.0 - PERIOD_START_CLOCK[p]
            ends[p] = max(ends[p], t)
    if not ends:  # fall back on the last event of each period
        for e in events:
            p = e['period']
            ends[p] = max(ends[p], e['minute'] + e['second'] / 60.0 - PERIOD_START_CLOCK[p])
    return {p: ends[p] for p in sorted(ends) if p <= 4}


class Clock:
    """Converts (match clock, period) or (timestamp within period, period) into elapsed minutes."""

    def __init__(self, lengths):
        self.lengths = lengths
        self.offset, acc = {}, 0.0
        for p in sorted(lengths):
            self.offset[p] = acc
            acc += lengths[p]
        self.duration = acc
        self.last_period = max(lengths)

    def from_clock(self, clock, period):
        p = period if period in self.offset else self.last_period
        t = clock_to_min(clock) - PERIOD_START_CLOCK[p]
        return self.offset[p] + min(max(t, 0.0), self.lengths[p])

    def from_timestamp(self, ts, period):
        h, m, s = ts.split(':')
        t = int(h) * 60 + int(m) + float(s) / 60.0
        return self.offset[period] + min(t, self.lengths[period])


def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap(iv1, iv2):
    """Total length of the intersection of two merged interval lists."""
    i = j = 0
    tot = 0.0
    while i < len(iv1) and j < len(iv2):
        a = max(iv1[i][0], iv2[j][0])
        b = min(iv1[i][1], iv2[j][1])
        if b > a:
            tot += b - a
        if iv1[i][1] < iv2[j][1]:
            i += 1
        else:
            j += 1
    return tot


def _spell_times(p, clock):
    """[(start, end, spell)] for one player, cut at a sending-off."""
    cut = None
    for c in p.get('cards') or []:
        if c.get('card_type') in SENDING_OFF:
            t = clock.from_clock(c['time'], c['period'])
            cut = t if cut is None else min(cut, t)
    out = []
    for s in p.get('positions') or []:
        a = clock.from_clock(s['from'], s['from_period'])
        b = clock.duration if s['to'] is None else clock.from_clock(s['to'], s['to_period'] or s['from_period'])
        if cut is not None:
            b = min(b, cut)
        out.append((a, b, s))
    return out, cut


def _drop_phantom_returns(team_spells, duration):
    """Fix a StatsBomb quirk: a player goes 'Player Off' (treatment), is in fact replaced, but the
    lineup still shows a 'Player On' spell, so 12 players are on the pitch. While any moment has
    more than 11, drop the latest 'Player On' spell that overlaps it (and that player's later spells).
    Returns the number of spells dropped."""
    dropped = 0
    for _ in range(10):
        events = []
        for pid, sp in team_spells.items():
            for a, b, _s in sp:
                if b > a:
                    events += [(a, 1), (b, -1)]
        events.sort(key=lambda x: (x[0], x[1]))
        n, over = 0, None
        for t, d in events:
            n += d
            if n > 11:
                over = t
                break
        if over is None:
            return dropped
        cands = [(a, pid, i) for pid, sp in team_spells.items() for i, (a, b, s) in enumerate(sp)
                 if 'Player On' in (s.get('start_reason') or '') and a <= over + 1e-9 < b]
        if not cands:
            return dropped
        _, pid, i = max(cands)
        dropped += len(team_spells[pid]) - i
        team_spells[pid] = team_spells[pid][:i]
    return dropped


def appearances(lineups, clock, match_id, pos_group, pos_role=None):
    """One row per player who was on the pitch; also returns {(team, player_id): intervals}."""
    rows, ivs = [], {}
    for team in lineups:
        tname = team['team_name']
        spells_of, cut_of = {}, {}
        for p in team['lineup']:
            if p.get('positions'):
                spells_of[p['player_id']], cut_of[p['player_id']] = _spell_times(p, clock)
        n_orig = {pid: len(sp) for pid, sp in spells_of.items()}
        _drop_phantom_returns(spells_of, clock.duration)
        for p in team['lineup']:
            if p['player_id'] not in spells_of:
                continue
            times = spells_of[p['player_id']]
            spells = [s for _, _, s in times]
            if not spells:
                continue
            cut = cut_of[p['player_id']]
            by_pos = defaultdict(float)
            intervals = []
            for a, b, s in times:
                if b > a:
                    by_pos[s['position']] += b - a
                    intervals.append((a, b))
            intervals = merge(intervals)
            if not intervals:
                continue
            by_grp = defaultdict(float)
            for pos, m in by_pos.items():
                by_grp[pos_group[pos]] += m
            minutes = sum(b - a for a, b in intervals)
            last = spells[-1]
            if cut is not None and cut <= intervals[-1][1] + 1e-9:
                off_reason = 'Red Card'
            elif last['to'] is None:
                off_reason = 'Final Whistle'
            elif 'Injury' in (last.get('end_reason') or ''):
                off_reason = 'Injury'
            elif len(spells) < n_orig[p['player_id']]:
                off_reason = 'Replaced after Player Off'
            elif 'Player Off' in (last.get('end_reason') or ''):
                off_reason = 'Player Off'
            else:
                off_reason = 'Substitution'
            rows.append({
                'match_id': match_id, 'team': tname, 'player_id': p['player_id'],
                'name': p.get('player_nickname') or p['player_name'], 'full_name': p['player_name'],
                'started': spells[0].get('start_reason') == 'Starting XI',
                'on_min': intervals[0][0], 'off_min': intervals[-1][1], 'minutes': minutes,
                'off_reason': off_reason, 'red_card_min': cut if off_reason == 'Red Card' else None,
                'position': max(by_pos, key=by_pos.get), 'group': max(by_grp, key=by_grp.get),
                **{f'min_{g}': by_grp.get(g, 0.0) for g in ('GK', 'DEF', 'MID', 'FWD')},
                **({f'rmin_{r}': sum(m for pos, m in by_pos.items() if pos_role[pos] == r)
                    for r in sorted(set(pos_role.values()))} if pos_role else {}),
                'intervals': ';'.join(f'{a:.3f}-{b:.3f}' for a, b in intervals),
            })
            ivs[(tname, p['player_id'])] = intervals
    return rows, ivs


def parse_intervals(s):
    return [[float(x) for x in part.split('-')] for part in s.split(';') if part]
