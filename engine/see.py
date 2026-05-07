"""see.py — Static Exchange Evaluation (SEE) for capture sanity."""

from __future__ import annotations

from position import Position, Move, WHITE, BLACK, type_of, color_of


PIECE_VALUE = [100, 320, 330, 500, 900, 20000]


def _attacker_list(pos: Position, target_sq: int, side: int) -> list[tuple[int, int]]:
    """Return list of (value, from_sq) for all attackers of target_sq by side."""
    at = pos.attackers_to(target_sq, by_side=side)
    out: list[tuple[int, int]] = []
    x = at
    while x:
        lsb = x & -x
        sq = lsb.bit_length() - 1
        piece = pos.board[sq]
        if piece >= 0 and color_of(piece) == side:
            out.append((PIECE_VALUE[type_of(piece)], sq))
        x ^= lsb
    out.sort()
    return out


def see(pos: Position, mv: Move) -> int:
    """
    Approximate net material gain for a capture sequence starting with mv.
    Positive means the capture is likely safe.
    """
    if not (mv.is_ep or pos.board[mv.to_sq] >= 0):
        return 0

    # Determine captured piece value.
    if mv.is_ep:
        captured_val = PIECE_VALUE[0]
    else:
        captured_val = PIECE_VALUE[type_of(pos.board[mv.to_sq])]

    gain = [0] * 32
    gain[0] = captured_val

    # We'll simulate attackers on the destination square without fully updating rays;
    # this is a conservative heuristic used for ordering/filtering, not exact.
    side = pos.side
    target = mv.to_sq

    # First attacker is the moving piece.
    moving_piece = pos.board[mv.from_sq]
    if moving_piece < 0:
        return 0
    gain_idx = 0
    attackers = {WHITE: _attacker_list(pos, target, WHITE), BLACK: _attacker_list(pos, target, BLACK)}

    # remove the moving attacker from its side list
    attackers[side] = [(v, s) for (v, s) in attackers[side] if s != mv.from_sq]

    # material of the piece that moved onto target
    cur_val = PIECE_VALUE[type_of(moving_piece)]
    gain_idx += 1
    gain[gain_idx] = gain[gain_idx - 1] - cur_val

    side ^= 1
    while attackers[side]:
        # least valuable attacker captures
        v, s = attackers[side].pop(0)
        cur_val = v
        gain_idx += 1
        gain[gain_idx] = gain[gain_idx - 1] - cur_val
        side ^= 1

    # minimax backward
    for i in range(gain_idx - 1, -1, -1):
        gain[i] = max(-gain[i + 1], gain[i])
    return gain[0]

