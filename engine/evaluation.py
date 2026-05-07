"""evaluation.py — submission v3 evaluation (PST + basic opening terms)."""

from __future__ import annotations

from bitboard import file_of, rank_of
from position import Position, WHITE, BLACK, type_of, color_of


# type 0..5: pawn, knight, bishop, rook, queen, king
PIECE_VALUE = [100, 320, 330, 500, 900, 0]


# PSTs from v1 (row0=rank8 .. row7=rank1), white perspective
_PAWN_MID_PST = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
_PAWN_END_PST = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [80, 80, 80, 80, 80, 80, 80, 80],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [30, 30, 30, 30, 30, 30, 30, 30],
    [20, 20, 20, 20, 20, 20, 20, 20],
    [10, 10, 10, 10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10, 10, 10, 10],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
_KNIGHT_PST = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
]
_BISHOP_PST = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
]
_ROOK_PST = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0],
]
_QUEEN_PST = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20],
]
_KING_MID_PST = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [20, 30, 10, 0, 0, 10, 30, 20],
]
_KING_END_PST = [
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10, 0, 0, -10, -20, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -30, 0, 0, 0, 0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50],
]

_PST = {
    1: _KNIGHT_PST,
    2: _BISHOP_PST,
    3: _ROOK_PST,
    4: _QUEEN_PST,
}


def _endgame_t(pos: Position) -> float:
    # Smoothish phase based on remaining non-pawn material
    w = b = 0
    for p in pos.board:
        if p < 0:
            continue
        pt = type_of(p)
        if pt in (5, 0):
            continue
        val = PIECE_VALUE[pt]
        if color_of(p) == WHITE:
            w += val
        else:
            b += val
    # 0..1
    max_side = max(w, b)
    return 1.0 - min(1.0, max_side / 3200.0)


def _pst_value(pt: int, sq: int, color: int, et: float) -> int:
    r = rank_of(sq)
    c = file_of(sq)
    idx = 7 - r if color == WHITE else r
    if pt == 0:
        return int(_PAWN_MID_PST[idx][c] * (1 - et) + _PAWN_END_PST[idx][c] * et)
    if pt == 5:
        return int(_KING_MID_PST[idx][c] * (1 - et) + _KING_END_PST[idx][c] * et)
    return _PST[pt][idx][c]


def _pawn_structure_score(pos: Position) -> int:
    # Very small/cheap pawn terms (kept conservative to avoid tactical distortion).
    # Passed pawns: reward advancement if no enemy pawn blocks adjacent files.
    white_pawns = []
    black_pawns = []
    for sq, p in enumerate(pos.board):
        if p < 0:
            continue
        if type_of(p) == 0:
            (white_pawns if color_of(p) == WHITE else black_pawns).append((rank_of(sq), file_of(sq)))

    score = 0
    passed_bonus = [0, 10, 15, 25, 40, 60, 90, 0]

    for r, f in white_pawns:
        if not any(bf in (f - 1, f, f + 1) and br > r for br, bf in black_pawns):
            score += passed_bonus[r]
    for r, f in black_pawns:
        if not any(wf in (f - 1, f, f + 1) and wr < r for wr, wf in white_pawns):
            score -= passed_bonus[7 - r]
    return score


def evaluate(pos: Position) -> int:
    """White-centric evaluation."""
    et = _endgame_t(pos)
    score = 0
    w_bishops = b_bishops = 0
    w_rooks = []
    b_rooks = []
    w_knights = []
    b_knights = []
    w_king = b_king = -1
    for sq, p in enumerate(pos.board):
        if p < 0:
            continue
        pt = type_of(p)
        val = PIECE_VALUE[pt]
        c = color_of(p)
        score += val if c == WHITE else -val
        # PST
        pst = _pst_value(pt, sq, c, et)
        score += pst if c == WHITE else -pst

    # Collect piece locations
    for sq, p in enumerate(pos.board):
        if p < 0:
            continue
        c = color_of(p)
        pt = type_of(p)
        if pt == 2:  # bishop
            if c == WHITE:
                w_bishops += 1
            else:
                b_bishops += 1
        elif pt == 3:  # rook
            (w_rooks if c == WHITE else b_rooks).append(sq)
        elif pt == 1:  # knight
            (w_knights if c == WHITE else b_knights).append(sq)
        elif pt == 5:  # king
            if c == WHITE:
                w_king = sq
            else:
                b_king = sq

    score += _pawn_structure_score(pos)

    # Opening bias: prefer center pawns + development early
    if pos.fullmove <= 10:
        center = {("d", 4), ("e", 4), ("c", 4), ("f", 4), ("d", 5), ("e", 5)}
        # center pawns
        for sq, p in enumerate(pos.board):
            if p < 0 or type_of(p) != 0:
                continue
            f = "abcdefgh"[file_of(sq)]
            r = rank_of(sq) + 1
            if (f, r) in center:
                score += 45 if color_of(p) == WHITE else -45
            # extra bonus for classic center pawns advanced (e4/d4 and ...e5/...d5)
            if (f, r) in {("e", 4), ("d", 4), ("e", 5), ("d", 5)}:
                score += 25 if color_of(p) == WHITE else -25
        # development
        # knights/bishops off back rank get a bonus
        for sq, p in enumerate(pos.board):
            if p < 0:
                continue
            c = color_of(p)
            pt = type_of(p)
            if pt in (1, 2):
                home_rank = 0 if c == WHITE else 7
                if rank_of(sq) != home_rank:
                    score += 8 if c == WHITE else -8

        # discourage anti-development pawn pushes early (prevents a3/c3 style openings)
        bad_pawn_squares = {
            ("a", 3), ("h", 3), ("a", 6), ("h", 6),
            ("c", 3), ("f", 3), ("c", 6), ("f", 6),
        }
        for sq, p in enumerate(pos.board):
            if p < 0 or type_of(p) != 0:
                continue
            f = "abcdefgh"[file_of(sq)]
            r = rank_of(sq) + 1
            if (f, r) in bad_pawn_squares:
                score -= 35 if color_of(p) == WHITE else -35

    # Bishop pair
    if w_bishops >= 2:
        score += 30
    if b_bishops >= 2:
        score -= 30

    # Rooks on open/semi-open files
    def file_pawns(file_c: int) -> tuple[bool, bool]:
        w = b = False
        for r in range(8):
            p = pos.board[r * 8 + file_c]
            if p < 0:
                continue
            if type_of(p) == 0:
                if color_of(p) == WHITE:
                    w = True
                else:
                    b = True
        return w, b

    for sq in w_rooks:
        w, b = file_pawns(file_of(sq))
        if not w:
            score += 10
            if not b:
                score += 10
    for sq in b_rooks:
        w, b = file_pawns(file_of(sq))
        if not b:
            score -= 10
            if not w:
                score -= 10

    # Knight outposts (simple)
    def pawn_attacks_square(color: int, target_sq: int) -> bool:
        tr, tc = rank_of(target_sq), file_of(target_sq)
        pr = tr - 1 if color == WHITE else tr + 1
        if 0 <= pr < 8:
            for dc in (-1, 1):
                fc = tc + dc
                if 0 <= fc < 8:
                    p = pos.board[pr * 8 + fc]
                    if p >= 0 and type_of(p) == 0 and color_of(p) == color:
                        return True
        return False

    def enemy_pawn_can_attack(enemy_color: int, target_sq: int) -> bool:
        tr, tc = rank_of(target_sq), file_of(target_sq)
        pr = tr + 1 if enemy_color == WHITE else tr - 1
        if 0 <= pr < 8:
            for dc in (-1, 1):
                fc = tc + dc
                if 0 <= fc < 8:
                    p = pos.board[pr * 8 + fc]
                    if p >= 0 and type_of(p) == 0 and color_of(p) == enemy_color:
                        return True
        return False

    for sq in w_knights:
        if rank_of(sq) >= 3 and pawn_attacks_square(WHITE, sq) and not enemy_pawn_can_attack(BLACK, sq):
            score += 15
    for sq in b_knights:
        if rank_of(sq) <= 4 and pawn_attacks_square(BLACK, sq) and not enemy_pawn_can_attack(WHITE, sq):
            score -= 15

    # King safety (very light pawn shield)
    def king_shield(king_sq: int, color: int) -> int:
        if king_sq < 0:
            return 0
        r, c = rank_of(king_sq), file_of(king_sq)
        dr = 1 if color == WHITE else -1
        rr = r + dr
        if not (0 <= rr < 8):
            return 0
        cnt = 0
        for dc in (-1, 0, 1):
            cc = c + dc
            if 0 <= cc < 8:
                p = pos.board[rr * 8 + cc]
                if p >= 0 and type_of(p) == 0 and color_of(p) == color:
                    cnt += 1
        return cnt

    score += 8 * (king_shield(w_king, WHITE) - king_shield(b_king, BLACK))

    return score

