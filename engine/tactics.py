"""
tactics.py — Game-phase tactics: opening book, move ordering, and
             quiescence search.

Opening   : Position-keyed book built from 15 common opening lines.
Middlegame: MVV-LVA + TT-move + killer moves + history heuristic + PST-delta.
Endgame   : Lone-king detection; quiescence search over captures.

order_moves() extended to accept optional search-state hints:
  tt_move  — best move from the transposition table (tried first).
  killers  — two quiet beta-cutoff moves per ply (tried before other quiets).
  history  — history heuristic scores indexed by (from_sq, to_sq).
  ply      — current ply from root (needed to index killers correctly).
"""

import random

from board import (
    GameState,
    EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    WHITE, BLACK,
    piece_type, piece_color,
    STARTING_FEN,
)
from evaluation import evaluate, _PST, _KNIGHT_PST, _BISHOP_PST, _ROOK_PST, _QUEEN_PST


# ── Opening book ───────────────────────────────────────────────────────────────

_OPENING_LINES = [
    # Italian Game
    ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","c2c3","g8f6","d2d4","e5d4"],
    # Italian Game (Giuoco Pianissimo)
    ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","d2d3","g8f6","c2c3","d7d6"],
    # Two Knights Defense
    ["e2e4","e7e5","g1f3","b8c6","f1c4","g8f6","d2d4","e5d4","e1g1","f6e4"],
    # Ruy Lopez (Spanish)
    ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1","f8e7"],
    # Ruy Lopez (Berlin)
    ["e2e4","e7e5","g1f3","b8c6","f1b5","g8f6","e1g1","f6e4","d2d4","e4d6"],
    # Ruy Lopez (Open)
    ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1","f6e4"],
    # Petrov Defense
    ["e2e4","e7e5","g1f3","g8f6","f3e5","d7d6","e5f3","f6e4","d2d4","d6d5"],
    # Sicilian – Najdorf
    ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","a7a6"],
    # Sicilian – Classical
    ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","b8c6"],
    # Sicilian – Kan
    ["e2e4","c7c5","g1f3","e7e6","d2d4","c5d4","f3d4","a7a6","b1c3","d7d6"],
    # Sicilian – Dragon
    ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","g7g6"],
    # Sicilian – Accelerated Dragon
    ["e2e4","c7c5","g1f3","b8c6","d2d4","c5d4","f3d4","g7g6","b1c3","f8g7"],
    # Sicilian – Sveshnikov-ish setup
    ["e2e4","c7c5","g1f3","b8c6","d2d4","c5d4","f3d4","g8f6","b1c3","e7e5"],
    # Sicilian – Alapin
    ["e2e4","c7c5","c2c3","d7d5","e4d5","d8d5","d2d4","g8f6","g1f3","e7e6"],
    # Queen's Gambit Declined
    ["d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","c1g5","f8e7","e2e3","e8g8"],
    # Slav Defense
    ["d2d4","d7d5","c2c4","c7c6","b1c3","g8f6","g1f3","d5c4","a2a4","c8f5"],
    # Semi-Slav
    ["d2d4","d7d5","c2c4","c7c6","g1f3","g8f6","b1c3","e7e6","c1g5","d5c4"],
    # Queen's Gambit (mainline developing)
    ["d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","g1f3","f8e7","c1g5","e8g8"],
    # King's Indian Defense
    ["d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6","g1f3","e8g8"],
    # Grunfeld Defense
    ["d2d4","g8f6","c2c4","g7g6","b1c3","d7d5","c4d5","f6d5","e2e4","d5c3"],
    # Nimzo-Indian
    ["d2d4","g8f6","c2c4","e7e6","b1c3","f8b4","e2e3","e8g8","g1f3","d7d5"],
    # Scotch Game
    ["e2e4","e7e5","g1f3","b8c6","d2d4","e5d4","f3d4","f8c5","c1e3","g8f6"],
    # Scotch Four Knights
    ["e2e4","e7e5","g1f3","b8c6","b1c3","g8f6","d2d4","e5d4","f3d4","f8b4"],
    # French Defense
    ["e2e4","e7e6","d2d4","d7d5","b1c3","g8f6","c1g5","f8e7","e4e5","e8g8"],
    # French Advance
    ["e2e4","e7e6","d2d4","d7d5","e4e5","c7c5","g1f3","b8c6","f1e2","c5d4"],
    # French Tarrasch
    ["e2e4","e7e6","d2d4","d7d5","b1d2","c7c5","g1f3","b8c6","f1b5","c5d4"],
    # Caro-Kann Classical
    ["e2e4","c7c6","d2d4","d7d5","b1c3","d5e4","c3e4","c8f5","e4g3","f5g6"],
    # Caro-Kann Advance
    ["e2e4","c7c6","d2d4","d7d5","e4e5","c8f5","g1f3","e7e6","f1e2","c6c5"],
    # Four Knights Game
    ["e2e4","e7e5","g1f3","b8c6","b1c3","g8f6","f1b5","f8b4","e1g1","e8g8"],
    # Vienna Game
    ["e2e4","e7e5","b1c3","g8f6","g1f3","b8c6","f1c4","f8c5","d2d3","d7d6"],
    # London System
    ["d2d4","d7d5","g1f3","g8f6","c1f4","e7e6","e2e3","f8d6","f1d3","e8g8"],
    # Colle System
    ["d2d4","d7d5","g1f3","g8f6","e2e3","e7e6","f1d3","c7c5","b1d2","b8c6"],
    # Pirc Defense
    ["e2e4","d7d6","d2d4","g8f6","b1c3","g7g6","g1f3","f8g7","c1e3","e8g8"],
    # Modern Defense
    ["e2e4","g7g6","d2d4","f8g7","b1c3","d7d6","g1f3","b8c6","f1e2","e7e5"],
    # Queen's Gambit Accepted
    ["d2d4","d7d5","c2c4","d5c4","g1f3","g8f6","e2e3","e7e6","f1c4","c7c5"],
    # QGA mainline with e3
    ["d2d4","d7d5","c2c4","d5c4","e2e3","g8f6","f1c4","e7e6","g1f3","c7c5"],
    # English Opening
    ["c2c4","e7e5","b1c3","g8f6","g1f3","b8c6","g2g3","d7d5","c4d5","f6d5"],
    # English Symmetrical
    ["c2c4","c7c5","b1c3","b8c6","g1f3","g8f6","g2g3","g7g6","f1g2","f8g7"],
    # Réti Opening
    ["g1f3","d7d5","c2c4","g8f6","g2g3","e7e6","f1g2","f8e7","e1g1","e8g8"],
    # Catalan setup
    ["d2d4","g8f6","c2c4","e7e6","g2g3","d7d5","f1g2","f8e7","g1f3","e8g8"],
]


def _build_opening_book() -> dict:
    # position_key -> {uci: count}
    book: dict[str, dict[str, int]] = {}

    def _find(state, uci):
        for m in state.get_legal_moves():
            if m.uci() == uci:
                return m
        return None

    for line in _OPENING_LINES:
        state = GameState.from_fen(STARTING_FEN)
        for uci in line:
            key = state.position_key()
            book.setdefault(key, {})
            book[key][uci] = book[key].get(uci, 0) + 1
            move = _find(state, uci)
            if move is None:
                break
            state = state.make_move(move)
    return book


OPENING_BOOK: dict = _build_opening_book()


def book_move(state: GameState):
    """
    Return a book-move UCI string for the position, or None.

    Prefers the most-supported move in the book (best_move),
    falling back to weighted randomness among known lines.
    """
    candidates = OPENING_BOOK.get(state.position_key())
    if not candidates:
        return None
    legal_set = {m.uci() for m in state.get_legal_moves()}
    valid = [(uci, cnt) for (uci, cnt) in candidates.items() if uci in legal_set]
    if not valid:
        return None
    # Prefer the highest-count move; break ties randomly.
    best_cnt = max(cnt for _, cnt in valid)
    best = [uci for (uci, cnt) in valid if cnt == best_cnt]
    return random.choice(best)


# ── Endgame helpers ────────────────────────────────────────────────────────────

def opponent_king_only(state: GameState) -> bool:
    """True when the opponent has no pieces other than their king."""
    opp = -state.turn
    return not any(
        p != EMPTY and piece_color(p) == opp and piece_type(p) != KING
        for p in state.board
    )


# ── Move ordering ──────────────────────────────────────────────────────────────

_MVV_LVA = {PAWN: 1, KNIGHT: 2, BISHOP: 3, ROOK: 4, QUEEN: 5, KING: 6}

# Mid-game PST for non-king/pawn pieces (used for quiet-move PST-delta ordering).
_ORD_PST = {
    KNIGHT: _KNIGHT_PST,
    BISHOP: _BISHOP_PST,
    ROOK:   _ROOK_PST,
    QUEEN:  _QUEEN_PST,
}


def _gives_direct_check(board, from_sq: int, to_sq: int,
                         pt: int, color: int, king_sq: int) -> bool:
    """
    True if a piece of type pt/color moving from_sq→to_sq directly attacks
    king_sq.  Discovered checks are not detected.
    """
    if to_sq == king_sq:
        return False
    r,  c  = to_sq  >> 3, to_sq  & 7
    kr, kc = king_sq >> 3, king_sq & 7
    dr, dc = kr - r, kc - c

    if pt == KNIGHT:
        return abs(dr) * abs(dc) == 2 and abs(dr) + abs(dc) == 3

    if pt == PAWN:
        fwd = -1 if color == WHITE else 1
        return dr == fwd and abs(dc) == 1

    if pt == KING:
        return False

    is_diag     = abs(dr) == abs(dc) and dr != 0
    is_straight = (dr == 0) != (dc == 0)

    if pt == BISHOP and not is_diag:                    return False
    if pt == ROOK   and not is_straight:                return False
    if pt == QUEEN  and not (is_diag or is_straight):   return False

    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    cr, cc = r + step_r, c + step_c
    while (cr, cc) != (kr, kc):
        s = cr * 8 + cc
        if s != from_sq and board[s] != EMPTY:
            return False
        cr += step_r; cc += step_c
    return True


def order_moves(moves: list, state: GameState,
                tt_move=None,
                killers=None, history=None, ply: int = 0) -> list:
    """
    Score and sort moves for alpha-beta ordering.

    Priority tiers (descending):
      10 000 000  TT / hash move  (best move from a prior search)
       8 000 000+ Winning captures (MVV-LVA)
       6 000 000  Promotions
       4 000 000  Losing captures  (MVV-LVA)
         850 000  Killer moves     (quiet β-cutoff moves from same ply)
         800 000  Direct checks    (quiet moves that give check)
             0+   Quiet moves      (history heuristic + PST delta)
    """
    board       = state.board
    opp         = -state.turn
    opp_king_sq = next((s for s in range(64) if board[s] == KING * opp), -1)

    for m in moves:
        from_sq = m.from_sq
        to_sq   = m.to_sq

        # ── TT / hash move ────────────────────────────────────────────────────
        if (tt_move is not None
                and from_sq == tt_move.from_sq
                and to_sq   == tt_move.to_sq):
            m.score = 10_000_000
            continue

        cap = piece_type(m.captured) if m.captured else 0
        att = piece_type(board[from_sq])

        # ── Captures (MVV-LVA) ────────────────────────────────────────────────
        if cap:
            delta = 10 * _MVV_LVA.get(cap, 0) - _MVV_LVA.get(att, 0)
            # Separate winning/even captures from losing ones.
            m.score = (8_000_000 if delta >= 0 else 4_000_000) + delta
            continue

        # ── Promotions ────────────────────────────────────────────────────────
        if m.promotion:
            m.score = 6_000_000
            continue

        # ── Killer moves ──────────────────────────────────────────────────────
        if killers is not None and ply < len(killers):
            mk = (from_sq, to_sq)
            if mk == killers[ply][0] or mk == killers[ply][1]:
                m.score = 850_000
                continue

        # ── Direct checks ─────────────────────────────────────────────────────
        if opp_king_sq >= 0 and _gives_direct_check(
                board, from_sq, to_sq, att, state.turn, opp_king_sq):
            m.score = 800_000
            continue

        # ── Quiet moves: history + PST-delta ──────────────────────────────────
        h = history.get((from_sq, to_sq), 0) if history else 0

        # PST delta: prefer moves that improve the piece's position.
        pst_table = _ORD_PST.get(att)
        if pst_table:
            fr, fc = from_sq >> 3, from_sq & 7
            tr_f   = fr if state.turn == WHITE else 7 - fr
            tr_t_r, tr_t_c = to_sq >> 3, to_sq & 7
            tr_t   = tr_t_r if state.turn == WHITE else 7 - tr_t_r
            h += pst_table[tr_t][tr_t_c] - pst_table[tr_f][fc]

        m.score = min(h, 700_000)   # cap below check/killer tiers

    moves.sort(key=lambda mv: mv.score, reverse=True)
    return moves


# ── Quiescence search ──────────────────────────────────────────────────────────

def _from_current(state: GameState, white_score: float) -> float:
    """Convert a White-centric score to the current player's perspective."""
    return white_score if state.turn == WHITE else -white_score


def quiescence(state: GameState, alpha: float, beta: float,
               use_pst: bool,
               deadline: float | None = None,
               nodes_ref: list | None = None) -> float:
    """
    Extend the search on captures until the position is quiet.
    Prevents the horizon effect on tactical sequences.
    """
    if deadline is not None and nodes_ref is not None:
        # Cheap periodic check (same cadence as agents.py). If time is up,
        # treat this node as "quiet enough" and return the static eval bound.
        if (nodes_ref[0] & 2047) == 0:
            import time
            if time.perf_counter() >= deadline:
                stand_pat = _from_current(state, evaluate(state, use_pst=use_pst))
                return max(alpha, stand_pat)

    stand_pat = _from_current(state, evaluate(state, use_pst=use_pst))
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    captures = [m for m in state._gen_pseudo_moves()
                if (m.captured or m.is_ep) and state._is_legal_fast(m)]
    order_moves(captures, state)

    for move in captures:
        ns    = state._apply(move)
        score = -quiescence(ns, -beta, -alpha, use_pst, deadline=deadline, nodes_ref=nodes_ref)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha
