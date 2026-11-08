from hub8002.minutes import Clock, appearances, overlap

PG = {'Goalkeeper': 'GK', 'Center Back': 'DEF', 'Center Midfield': 'MID', 'Center Forward': 'FWD'}


def spell(pos, a, b, pa, pb, sr='Starting XI', er='Final Whistle'):
    return {'position': pos, 'from': a, 'to': b, 'from_period': pa, 'to_period': pb, 'start_reason': sr, 'end_reason': er}


def player(pid, spells, cards=()):
    return {'player_id': pid, 'player_name': f'P{pid}', 'player_nickname': None, 'positions': spells, 'cards': list(cards)}


def test_clock_uses_real_half_lengths():
    c = Clock({1: 47.0, 2: 49.0})
    assert c.duration == 96.0
    assert c.from_clock('46:30', 1) == 46.5          # first-half stoppage time
    assert c.from_clock('45:00', 2) == 47.0          # second half restarts at 45:00
    assert c.from_clock('60:00', 2) == 62.0
    assert abs(c.from_timestamp('00:10:30.000', 2) - 57.5) < 1e-9


def test_red_card_cuts_spell_and_sub_minutes():
    c = Clock({1: 45.0, 2: 45.0})
    team = {'team_name': 'A', 'lineup': [
        player(1, [spell('Center Back', '00:00', None, 1, None)],
               cards=[{'time': '60:00', 'period': 2, 'card_type': 'Red Card'}]),
        player(2, [spell('Center Midfield', '00:00', '70:00', 1, 2, er='Substitution - Off (Tactical)')]),
        player(3, [spell('Center Midfield', '70:00', None, 2, None, sr='Substitution - On (Tactical)')]),
    ]}
    rows, ivs = appearances([team], c, 1, PG)
    m = {r['player_id']: r for r in rows}
    assert abs(m[1]['minutes'] - 60.0) < 1e-9 and m[1]['off_reason'] == 'Red Card'
    assert abs(m[2]['minutes'] - 70.0) < 1e-9 and m[2]['off_reason'] == 'Substitution'
    assert abs(m[3]['minutes'] - 20.0) < 1e-9
    assert abs(overlap(ivs[('A', 1)], ivs[('A', 3)]) - 0.0) < 1e-9
    assert abs(overlap(ivs[('A', 1)], ivs[('A', 2)]) - 60.0) < 1e-9


def test_phantom_return_is_dropped_when_twelve_on_pitch():
    c = Clock({1: 45.0, 2: 45.0})
    lineup = [player(100 + i, [spell('Center Back', '00:00', None, 1, None)]) for i in range(10)]
    lineup.append(player(1, [spell('Center Midfield', '00:00', '30:00', 1, 1, er='Player Off'),
                             spell('Center Midfield', '33:00', None, 1, None, sr='Player On')]))
    lineup.append(player(2, [spell('Center Midfield', '33:00', None, 1, None, sr='Substitution - On (Injury)')]))
    rows, _ = appearances([{'team_name': 'A', 'lineup': lineup}], c, 1, PG)
    m = {r['player_id']: r for r in rows}
    assert abs(m[1]['minutes'] - 30.0) < 1e-9 and m[1]['off_reason'] == 'Replaced after Player Off'
    assert abs(m[2]['minutes'] - 57.0) < 1e-9
    assert abs(sum(r['minutes'] for r in rows) - (11 * 90 - 3)) < 1e-9   # 3 minutes with 10 men during treatment
