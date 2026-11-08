"""Pass table and the two layers (decisions 3 and 4).

is_org : completed open-play pass with a recipient (corners, free kicks, throw-ins, goal kicks
         and kick-offs removed).
is_box : completed open-play pass from outside the opponent's box to inside it
         (x >= 102, 18 <= y <= 62 on the 120 x 80 pitch).
is_box_pilot : the pilot's definition (any completed pass ending in the box, set pieces included),
         kept to measure how much the new definition changes Messi's share.
A pass is completed when StatsBomb gives no pass.outcome.
"""


def in_box(x, y, box):
    return x >= box['x_min'] and box['y_min'] <= y <= box['y_max']


def pass_rows(events, clock, match_id, cfg):
    box = cfg['box']
    set_pieces = set(cfg['set_piece_types'])
    box_excl_sp = cfg.get('box_layer_excludes_set_pieces', True)
    rows = []
    for e in events:
        if e['type']['name'] != 'Pass' or e['period'] > 4:
            continue
        p = e['pass']
        x0, y0 = e['location'][:2]
        x1, y1 = p['end_location'][:2]
        completed = 'outcome' not in p
        recipient = p.get('recipient', {}).get('id')
        ptype = (p.get('type') or {}).get('name')
        open_play = ptype not in set_pieces
        ok = completed and recipient is not None and recipient != e['player']['id']
        start_out, end_in = not in_box(x0, y0, box), in_box(x1, y1, box)
        rows.append({
            'match_id': match_id, 'team': e['team']['name'], 'period': e['period'],
            't_min': clock.from_timestamp(e['timestamp'], e['period']),
            'passer_id': e['player']['id'], 'recipient_id': recipient,
            'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'completed': completed,
            'pass_type': ptype, 'play_pattern': e['play_pattern']['name'], 'possession': e['possession'],
            'is_org': ok and open_play,
            'is_box': ok and start_out and end_in and (open_play or not box_excl_sp),
            'is_box_pilot': ok and end_in,
        })
    return rows
