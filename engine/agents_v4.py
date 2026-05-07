"""agents_v3_nnue.py — same search as submission v3, plus optional NNUE (v4/nnue_weights.npz)."""

from __future__ import annotations

import os
import sys

_V5_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENGINE_DIR = os.path.abspath(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_V5_ROOT, "data")
# Stockfish NNUE network on disk (used when SF_NNUE_FILE is unset or empty in env)
_STOCKFISH_NNUE_NET = os.path.join(_DATA_DIR, "nnue", "nn-c288c895ea92.nnue")
# Defaults if environment variables are not set (export SF_NNUE_* to override)
_DEFAULT_SF_NNUE_FILE = ""  # empty -> _STOCKFISH_NNUE_NET
_DEFAULT_SF_NNUE_BLEND = "1.0"
# Ensure local engine modules and data-backed modules are importable.
for _p in (_ENGINE_DIR, _DATA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import random
import time
from contextvars import ContextVar
from dataclasses import dataclass

from evaluation import evaluate
from position import Position, Move, WHITE
from tt import TranspositionTable, TT_EXACT, TT_LOWER, TT_UPPER
from see import see
from syzygy_345 import probe_bestmove, probe_bestmoves
# from opening_book import pick_book_move
from opening_book_v4 import pick_book_move
from bitboard import file_of, rank_of

try:
    import nnue_infer as _nnue  # noqa: E402  # submission v4
except ImportError:
    _nnue = None

try:
    # Optional: Stockfish NNUE bridge (DLL) for v4 only.
    from sf_nnue_bridge import sf_nnue as _sf_nnue  # type: ignore
except Exception:
    _sf_nnue = None

_SF_NNUE_CACHE: dict[int, int] = {}
_SF_NNUE_CACHE_ORDER: list[int] = []
_SF_NNUE_CACHE_MAX = 50_000

_EXP_DB_CONN = None

# When True (per-search), NNUE layers are skipped — used for handicap levels (no NNUE).
_EVAL_NO_NNUE: ContextVar[bool] = ContextVar("_eval_no_nnue", default=False)


@dataclass(frozen=True)
class SkillConfig:
    """GUI levels 2–10: behaviour presets on top of the v4 search stack."""

    no_opening_book: bool = False
    no_exp_book: bool = False
    no_syzygy: bool = False
    shallow_one_prob: float = 0.0           # fraction of moves that use depth‑1 search only
    second_best_prob: float = 0.0           # chance to deliberately pick score‑rank‑2 root move
    disable_nnue: bool = False
    quick_depth_cap: int | None = None       # clamp iterative deepening (e.g. 3)
    always_second_best: bool = False         # always take score‑rank‑2 root move after search


def _skill_for_level(level: int) -> SkillConfig:
    # 10 = full v4; presets per assignment spec for 9 … 2 (1 = RandomAgent elsewhere).
    m = {
        10: SkillConfig(),
        9: SkillConfig(no_opening_book=True, no_exp_book=True, no_syzygy=True),
        8: SkillConfig(shallow_one_prob=0.50),
        7: SkillConfig(second_best_prob=0.10),
        6: SkillConfig(disable_nnue=True),
        5: SkillConfig(disable_nnue=True, quick_depth_cap=3),
        4: SkillConfig(quick_depth_cap=3),
        3: SkillConfig(always_second_best=True),
        2: SkillConfig(disable_nnue=True, always_second_best=True),
    }
    return m.get(level, SkillConfig())


def _sf_nnue_eval_white_cp(pos: Position) -> int | None:
    if _sf_nnue is None:
        return None
    if not getattr(_sf_nnue, "enabled", lambda: False)():
        return None
    k = getattr(pos, "key", None)
    if isinstance(k, int) and k in _SF_NNUE_CACHE:
        return _SF_NNUE_CACHE[k]
    # Default big net; override with SF_NNUE_FILE or _DEFAULT_SF_NNUE_FILE (above).
    default_file = os.path.abspath(_DEFAULT_SF_NNUE_FILE.strip() or _STOCKFISH_NNUE_NET)
    evalfile = os.environ.get("SF_NNUE_FILE", "").strip() or default_file
    try:
        cp = int(_sf_nnue.eval_white_cp_from_pos(pos, evalfile))
    except Exception:
        return None
    if isinstance(k, int):
        _SF_NNUE_CACHE[k] = cp
        _SF_NNUE_CACHE_ORDER.append(k)
        if len(_SF_NNUE_CACHE_ORDER) > _SF_NNUE_CACHE_MAX:
            old = _SF_NNUE_CACHE_ORDER.pop(0)
            _SF_NNUE_CACHE.pop(old, None)
    return cp


def sf_nnue_status() -> str:
    """
    Human-readable status for the optional Stockfish NNUE bridge.
    Intended for logging / diagnostics (UCI info string).
    """
    if _sf_nnue is None:
        return "sf_nnue=unavailable"
    try:
        en = bool(getattr(_sf_nnue, "enabled", lambda: False)())
    except Exception:
        en = False
    try:
        default_file = os.path.abspath(_DEFAULT_SF_NNUE_FILE.strip() or _STOCKFISH_NNUE_NET)
        evalfile = os.environ.get("SF_NNUE_FILE", "").strip() or default_file
    except Exception:
        evalfile = ""
    try:
        blend = float(os.environ.get("SF_NNUE_BLEND", _DEFAULT_SF_NNUE_BLEND))
    except Exception:
        blend = None
    exists = None
    if evalfile:
        try:
            exists = os.path.isfile(evalfile)
        except Exception:
            exists = None
    return f"sf_nnue={'on' if en else 'off'} blend={blend} file={evalfile} file_exists={exists}"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _root_avoid_immediate_threefold(
    pos: Position,
    moves: list[Move],
    *,
    repetition_base: dict[int, int] | None,
) -> list[Move]:
    """
    If we are materially ahead and repetition history is known, avoid root moves
    that would make the current position occur for the 3rd time (immediate draw),
    *when* that can be detected from `repetition_base`.

    This complements alpha-beta's threefold handling: a 1-ply "finish the game" draw
    can be missed as a "best move" if the search doesn't see the termination clearly.
    """
    if repetition_base is None:
        return moves
    try:
        bal = _material_balance_cp(pos)
    except Exception:
        bal = 0
    min_cp = _env_int("V4_ROOT_REP_AVOID_MIN_CP", 200)
    if bal < min_cp:
        return moves

    kept: list[Move] = []
    for mv in moves:
        u = pos.push(mv)
        try:
            nxt = int(repetition_base.get(int(pos.key), 0) or 0) + 1
        finally:
            pos.pop(u)
        if nxt < 3:
            kept.append(mv)
    return kept if kept else moves


class _SearchTimeout(Exception):
    pass


_MAX_PLY = 64


def _mv_key(m: Move) -> tuple[int, int, int]:
    return (m.from_sq, m.to_sq, -1 if m.promo is None else m.promo)


def _update_killers(killers: list[list[tuple[int, int, int] | None]], ply: int, mv: Move) -> None:
    k = _mv_key(mv)
    if killers[ply][0] != k:
        killers[ply][1] = killers[ply][0]
        killers[ply][0] = k


def _raise_if_time_up(deadline: float | None, nodes: int) -> None:
    if deadline is None:
        return
    # Check more frequently when close to the deadline. This reduces overshoot
    # (important for online play / low time).
    now = time.perf_counter()
    rem = deadline - now
    mask = 2047
    if rem <= 0.20:
        mask = 511
    if rem <= 0.05:
        mask = 127
    if rem <= 0.015:
        mask = 31
    if (nodes & mask) != 0:
        return
    if now >= deadline:
        raise _SearchTimeout()


def _is_capture(pos: Position, mv: Move) -> bool:
    if mv.is_ep:
        return True
    return pos.board[mv.to_sq] >= 0


def _gives_check(pos: Position, mv: Move) -> bool:
    """True if making mv gives check to the opponent king."""
    u = pos.push(mv)
    try:
        # After pushing, it's opponent to move; if they are in check, mv is a checking move.
        return pos.in_check(pos.side)
    finally:
        pos.pop(u)


def _pawn_storm_bonus(pos: Position, mv: Move, depth: int) -> int:
    """Cheap king-attack heuristic for quiet pawn pushes near the enemy king."""
    if depth < 2:
        return 0
    if mv.promo is not None or mv.is_castle or mv.is_ep:
        return 0
    piece = pos.board[mv.from_sq]
    if piece < 0 or (piece % 6) != 0:  # pawn (type 0)
        return 0
    if pos.board[mv.to_sq] >= 0:
        return 0  # captures are scored elsewhere
    try:
        ksq = pos.king_sq(1 - pos.side)
        df = abs(file_of(mv.to_sq) - file_of(ksq))
        dr = abs(rank_of(mv.to_sq) - rank_of(ksq))
    except Exception:
        return 0
    d = df + dr
    if d <= 1:
        return 650_000
    if d == 2:
        return 350_000
    return 0


def _piece_value(piece_idx: int) -> int:
    # piece_idx: 0..11, type = idx % 6: 0=P,1=N,2=B,3=R,4=Q,5=K
    t = piece_idx % 6
    if t == 0:
        return 100
    if t == 1:
        return 320
    if t == 2:
        return 330
    if t == 3:
        return 500
    if t == 4:
        return 900
    return 0


def _material_balance_cp(pos: Position) -> int:
    """Material (centipawn) balance from side-to-move perspective."""
    w = 0
    b = 0
    for p in pos.board:
        if p < 0:
            continue
        v = _piece_value(p)
        if p < 6:
            w += v
        else:
            b += v
    # Positive = white up; convert to stm perspective at end.
    bal = w - b
    return bal if pos.side == WHITE else -bal


def _draw_score_stm(pos: Position) -> int:
    """
    Draw score (centipawns) from stm perspective; contempt when materially unbalanced.

    Tunable via V4_DRAW_CONTEMPT_DIV (default 20) and V4_DRAW_CONTEMPT_CAP (default 50).
    """
    try:
        bal = _material_balance_cp(pos)
    except Exception:
        bal = 0
    div = _env_int("V4_DRAW_CONTEMPT_DIV", 20)
    if div < 1:
        div = 1
    cap = _env_int("V4_DRAW_CONTEMPT_CAP", 50)
    if cap < 0:
        cap = 0
    c = min(cap, abs(int(bal)) // div)
    if bal > 0:
        return -c
    if bal < 0:
        return c
    return 0


def _upcoming_repetition_within_2ply(pos: Position, rep: dict[int, int]) -> bool:
    """
    True if some line from this node reaches a position key count >= 3 within 2 plies.

    Mirrors the idea of Stockfish's upcoming_repetition: clamp optimistic bounds when
    a third occurrence is reachable before the terminal threefold check fires.
    """
    max_br = _env_int("V4_UPCOMING_REP_MAX_MOVES", 16)
    moves = list(pos.legal_moves())
    if len(moves) > max_br:
        return False
    for m in moves:
        u = pos.push(m)
        try:
            r1 = dict(rep)
            k1 = int(pos.key)
            r1[k1] = r1.get(k1, 0) + 1
            if r1[k1] >= 3:
                return True
            moves2 = list(pos.legal_moves())
            if len(moves2) > max_br:
                continue
            for m2 in moves2:
                u2 = pos.push(m2)
                try:
                    r2 = dict(r1)
                    k2 = int(pos.key)
                    r2[k2] = r2.get(k2, 0) + 1
                    if r2[k2] >= 3:
                        return True
                finally:
                    pos.pop(u2)
        finally:
            pos.pop(u)
    return False


def _king_comfort_safe(pos: Position) -> bool:
    """Cheap 'king is safe enough' check for simplify heuristics."""
    try:
        if pos.in_check(pos.side):
            return False
        ksq = pos.king_sq(pos.side)
    except Exception:
        return False
    f = file_of(ksq)
    r = rank_of(ksq)
    # "Castled-ish" squares (or near): g1/c1 for white, g8/c8 for black.
    if pos.side == WHITE:
        return (r == 0 and (f <= 2 or f >= 5))
    return (r == 7 and (f <= 2 or f >= 5))


def _piece_count_including_kings(pos: Position) -> int:
    # Syzygy counts pieces including both kings.
    n = 0
    for p in pos.board:
        if p >= 0:
            n += 1
    return n


def _side_material_counts(pos: Position, side: int) -> tuple[int, int, int, int, int]:
    """Return counts (pawns, knights, bishops, rooks, queens) for `side`."""
    pawns = knights = bishops = rooks = queens = 0
    lo = 0 if side == WHITE else 6
    hi = lo + 6
    for p in pos.board:
        if p < lo or p >= hi:
            continue
        t = p % 6
        if t == 0:
            pawns += 1
        elif t == 1:
            knights += 1
        elif t == 2:
            bishops += 1
        elif t == 3:
            rooks += 1
        elif t == 4:
            queens += 1
    return pawns, knights, bishops, rooks, queens


def _has_basic_mating_material(pos: Position, side: int) -> bool:
    """
    Conservative "can mate without opponent help" material test.
    Used only to allow aggressive simplification into Syzygy.
    """
    pawns, n, b, r, q = _side_material_counts(pos, side)
    if q >= 1 or r >= 1:
        return True
    # Minor-piece mates:
    if b >= 2:
        return True
    if b >= 1 and n >= 1:
        return True
    if (b + n) >= 3:
        return True
    # Practical: pawn + minor can usually convert (promotion gives mating material).
    if pawns >= 1 and (b + n) >= 1:
        return True
    return False


def _tb_simplify_bonus(pos: Position, mv: Move, *, time_limit_s: float | None = None) -> int:
    """
    When clearly winning, allow *unequal* captures/trades that reduce total pieces to <= 5
    (your 3-4-5 Syzygy domain), as long as we keep basic mating material.
    This is only a move-order / tie-break heuristic.
    """
    if not _is_capture(pos, mv) or mv.is_ep:
        return 0
    if _material_balance_cp(pos) < 450:
        return 0
    if not _king_comfort_safe(pos):
        return 0

    before = _piece_count_including_kings(pos)
    if before <= 5:
        return 0

    u = pos.push(mv)
    try:
        after = _piece_count_including_kings(pos)
        if after > 5:
            return 0
        mover = 1 - pos.side  # after push, side toggles; mover is the previous side
        if not _has_basic_mating_material(pos, mover):
            return 0

        drop = before - after
        bonus = 250_000 + drop * 90_000
        if time_limit_s is not None and time_limit_s <= 0.20:
            bonus += 200_000
        return bonus
    finally:
        pos.pop(u)


def _queen_trade_bonus(pos: Position, mv: Move) -> int:
    """
    Prefer immediate queen trades when we're ahead, even if not "up huge".
    This helps convert advantages and reduces tactical/perpetual risks.
    """
    if not _is_capture(pos, mv) or mv.is_ep:
        return 0
    cap = pos.board[mv.to_sq]
    att = pos.board[mv.from_sq]
    if cap < 0 or att < 0:
        return 0
    # Queen-for-queen only.
    if (cap % 6) != 4 or (att % 6) != 4:
        return 0
    # Trigger at a modest advantage (e.g. exchange up).
    if _material_balance_cp(pos) < 200:
        return 0
    if not _king_comfort_safe(pos):
        return 0
    try:
        if see(pos, mv) < 0:
            return 0
    except Exception:
        # If SEE fails, don't block the trade.
        pass
    return 220_000


def _simplify_bonus(pos: Position, mv: Move) -> int:
    """
    When clearly winning and king is safe, prefer equal-ish trades that simplify.
    This is a move-order / tie-break heuristic (not part of evaluation).
    """
    # Only when we're up big.
    if _material_balance_cp(pos) < 600:
        return 0
    if not _king_comfort_safe(pos):
        return 0
    if not _is_capture(pos, mv):
        return 0
    if mv.is_ep:
        return 0
    cap = pos.board[mv.to_sq]
    att = pos.board[mv.from_sq]
    if cap < 0 or att < 0:
        return 0
    cap_v = _piece_value(cap)
    att_v = _piece_value(att)
    # Reward exchanges of non-pawns; prefer trading queens/rooks most.
    if cap_v < 200 or att_v < 200:
        return 0
    if cap_v < att_v:
        return 0
    bonus = 20_000
    if cap_v >= 900:
        bonus += 40_000
    elif cap_v >= 500:
        bonus += 25_000
    else:
        bonus += 10_000
    # Also slightly reward "clean" captures by SEE.
    try:
        s = see(pos, mv)
        if s > 0:
            bonus += min(30_000, s * 50)
    except Exception:
        pass
    return bonus


def _trade_offer_bonus(pos: Position, mv: Move) -> int:
    """
    When clearly winning and king is safe, reward quiet moves that *offer* an equal-ish
    trade (opponent can capture our moved piece with a similar/higher value piece and
    we can recapture without losing material).
    """
    if _material_balance_cp(pos) < 600:
        return 0
    if not _king_comfort_safe(pos):
        return 0
    if _is_capture(pos, mv) or mv.is_ep or mv.is_castle or mv.promo is not None:
        return 0

    moved = pos.board[mv.from_sq]
    if moved < 0:
        return 0
    moved_v = _piece_value(moved)
    if moved_v < 200:
        return 0

    u1 = pos.push(mv)
    try:
        replies = [r for r in pos.legal_captures() if r.to_sq == mv.to_sq]
        if not replies:
            return 0

        best = 0
        for r in replies:
            capper = pos.board[r.from_sq]
            if capper < 0:
                continue
            capper_v = _piece_value(capper)
            if capper_v + 50 < moved_v:
                continue

            u2 = pos.push(r)
            try:
                recaps = [c for c in pos.legal_captures() if c.to_sq == mv.to_sq]
                if not recaps:
                    continue
                best_see = -10**9
                for c in recaps:
                    s = see(pos, c)
                    if s > best_see:
                        best_see = s
                if best_see >= -50:
                    b = 12_000
                    if capper_v >= 900:
                        b += 35_000
                    elif capper_v >= 500:
                        b += 20_000
                    else:
                        b += 8_000
                    b += min(15_000, max(0, best_see) * 40)
                    if b > best:
                        best = b
            finally:
                pos.pop(u2)

        return best
    finally:
        pos.pop(u1)


def _mate_mode_enabled() -> bool:
    v = os.environ.get("V4_MATE_MODE", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _search_info_enabled() -> bool:
    v = os.environ.get("V4_SEARCH_INFO", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _experience_book_enabled() -> bool:
    v = os.environ.get("V4_EXPERIENCE_BOOK", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _keyhex(pos: Position) -> str:
    k = int(getattr(pos, "key", 0)) & ((1 << 64) - 1)
    return f"{k:016x}"


def _experience_book_lookup_uci(pos: Position) -> str | None:
    """
    Look up a forced move from experience_book.sqlite3 (if enabled).
    Returns UCI string or None.
    """
    if not _experience_book_enabled():
        return None
    try:
        import sqlite3  # local import: avoid cost unless enabled
    except Exception:
        return None

    global _EXP_DB_CONN
    try:
        if _EXP_DB_CONN is None:
            db_path = os.environ.get("V4_EXPERIENCE_BOOK_PATH", "").strip()
            if not db_path:
                db_path = os.path.abspath(os.path.join(_DATA_DIR, "experience_book.sqlite3"))
            if not os.path.exists(db_path):
                return None
            _EXP_DB_CONN = sqlite3.connect(db_path)
        k = _keyhex(pos)
        row = _EXP_DB_CONN.execute("SELECT uci FROM book WHERE key = ?", (k,)).fetchone()
        if row is None:
            return None
        uci = str(row[0]).strip()
        # Optional usage log (for selftrain summary). One JSON line per hit.
        try:
            logp = os.environ.get("V4_EXPERIENCE_USAGE_LOG", "").strip()
            if logp:
                gid = os.environ.get("V4_GAME_ID", "").strip() or None
                with open(logp, "a", encoding="utf-8", errors="replace") as f:
                    f.write(f"{gid or ''}\t{k}\t{uci}\n")
        except Exception:
            pass
        return uci or None
    except Exception:
        return None


def _mate_mode_max_ply() -> int:
    try:
        n = int(os.environ.get("V4_MATE_MODE_MAX_PLY", "9"))
    except Exception:
        n = 9
    return max(5, min(13, n))


def _mate_mode_budget_s(time_limit_s: float | None) -> float:
    # Mate proving is expensive; keep this bounded.
    if time_limit_s is None:
        return 0.30
    if time_limit_s <= 0:
        return 0.0
    return min(2.5, max(0.10, time_limit_s * 0.35))


def _endgame_tb_mate_mode_bestmove(pos: Position) -> Move | None:
    """
    Endgame mate/conversion mode:
    If opponent has only king+pawns and we have at least one heavy piece (rook/queen),
    prioritize moves that *immediately* reduce the position to Syzygy-usable piece count.

    This is meant for shallow-search situations (depth ~2-4) where "obvious conversion"
    moves like KxP can be missed even though they unlock perfect tablebase play.
    Enable/disable via V4_ENDGAME_TB_MATE_MODE (default on).
    """
    if _env_int("V4_ENDGAME_TB_MATE_MODE", 1) == 0:
        return None
    try:
        from syzygy_345 import max_syzygy_pieces, probe_info  # type: ignore
    except Exception:
        return None

    try:
        maxp = int(max_syzygy_pieces())
    except Exception:
        maxp = 5

    # Only useful if we're currently above tablebase size by a small margin.
    try:
        cur_pieces = sum(1 for p in pos.board if p >= 0)
    except Exception:
        return None
    if cur_pieces <= maxp or cur_pieces > maxp + 3:
        return None

    us = pos.side
    our_lo, our_hi = (0, 6) if us == WHITE else (6, 12)
    opp_lo, opp_hi = (6, 12) if us == WHITE else (0, 6)

    have_heavy = False
    opp_only_king_pawns = True
    opp_has_pawn = False
    for piece in pos.board:
        if piece < 0:
            continue
        if our_lo <= piece < our_hi:
            t = piece % 6
            if t in (3, 4):  # rook/queen
                have_heavy = True
        elif opp_lo <= piece < opp_hi:
            t = piece % 6
            if t == 0:
                opp_has_pawn = True
            elif t == 5:
                pass
            else:
                opp_only_king_pawns = False
                break
    if (not have_heavy) or (not opp_only_king_pawns) or (not opp_has_pawn):
        return None

    best: Move | None = None
    best_key: tuple[int, int, int, int, str] | None = None

    for mv in pos.legal_moves():
        # Capture metadata must be computed before push.
        try:
            is_cap = _is_capture(pos, mv)
        except Exception:
            is_cap = False
        cap_pawn = 0
        try:
            if is_cap:
                if mv.is_ep:
                    cap_pawn = 1
                else:
                    tgt = pos.board[mv.to_sq]
                    if tgt >= 0 and opp_lo <= tgt < opp_hi and (tgt % 6) == 0:
                        cap_pawn = 1
        except Exception:
            cap_pawn = 0

        # Evaluate resulting position after move.
        u = pos.push(mv)
        try:
            n_after = sum(1 for p in pos.board if p >= 0)
            if n_after > maxp:
                continue

            # Probe resulting TB outcome (opponent to move). If opponent is losing, it's good.
            inf = probe_info(pos.to_fen())
            if not isinstance(inf, dict):
                continue
            opp_wdl = int(inf.get("wdl") or 0)  # side-to-move (opponent) POV
            our_wdl = -opp_wdl
            if our_wdl <= 0:
                continue

            mover = pos.board[mv.to_sq]
            king_move = 1 if (mover >= 0 and (mover % 6) == 5) else 0

            # Higher is better:
            #  - our_wdl (win > draw),
            #  - prefer capturing pawn (heuristic: capture moves first),
            #  - prefer king moves for KxP conversions,
            #  - fewer pieces remaining,
            #  - stable UCI tie-break.
            key = (our_wdl, cap_pawn, king_move, -n_after, mv.uci())
            if best_key is None or key > best_key:
                best_key = key
                best = mv
        finally:
            pos.pop(u)

    return best


def _mate_mode_bestmove(
    pos: Position,
    time_limit_s: float | None,
    repetition_base: dict[int, int] | None,
    stop_event=None,
) -> Move | None:
    """
    Mate mode: try to prove a short forced mate (including those starting with a quiet setup move).
    Prints a line to stdout when it triggers so you can see it in the terminal.
    """
    if not _mate_mode_enabled():
        return None
    debug = os.environ.get("V4_MATE_MODE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    budget_s = _mate_mode_budget_s(time_limit_s)
    if budget_s <= 0:
        return None
    max_ply = _mate_mode_max_ply()

    deadline = time.perf_counter() + budget_s
    if debug:
        print(f"[mate-mode] start max_ply={max_ply} budget={budget_s:.2f}s")
    rep = dict(repetition_base) if repetition_base else {pos.key: 1}

    memo: dict[tuple[int, int, int], bool] = {}

    def _time_up() -> bool:
        return time.perf_counter() >= deadline

    def mate_in(ply_left: int) -> bool:
        # True if side-to-move can force delivering mate within ply_left plies.
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            return False
        if _time_up():
            raise _SearchTimeout()
        if ply_left <= 0:
            return False
        if pos.halfmove >= 100:
            return False
        if rep.get(pos.key, 0) >= 3:
            return False

        mkey = (pos.key, ply_left, pos.side)
        hit = memo.get(mkey)
        if hit is not None:
            return hit

        moves = list(pos.legal_moves())
        if not moves:
            memo[mkey] = False
            return False

        # Strong forcing bias to keep branching under control:
        # checks > captures > a few best quiets.
        try:
            checks = [m for m in moves if _gives_check(pos, m)]
        except Exception:
            checks = []
        # Build a limited candidate set with forcing bias but not forcing-only:
        # quiet mates often require a quiet follow-up even when a check exists.
        pawn_moves: list[Move] = []
        qn_moves: list[Move] = []
        caps: list[Move] = []
        other_quiet: list[Move] = []
        for m in moves:
            if _is_capture(pos, m):
                caps.append(m)
                continue
            p = pos.board[m.from_sq]
            if p >= 0 and (p % 6) == 0:
                pawn_moves.append(m)
            elif p >= 0 and (p % 6) in (1, 4):
                qn_moves.append(m)
            else:
                other_quiet.append(m)

        # Merge in priority order. Keep sets small but include key categories.
        seen = set()
        out: list[Move] = []
        for m in checks + caps[:12] + qn_moves[:18] + pawn_moves[:14] + other_quiet[:10]:
            uci = m.uci()
            if uci in seen:
                continue
            seen.add(uci)
            out.append(m)
        moves = out[:22]

        def order(m: Move) -> int:
            s = 0
            if _is_capture(pos, m):
                try:
                    s += 1_000_000 + see(pos, m)
                except Exception:
                    s += 1_000_000
            try:
                if _gives_check(pos, m):
                    s += 800_000
            except Exception:
                pass
            if not _is_capture(pos, m):
                p = pos.board[m.from_sq]
                if p >= 0 and (p % 6) in (1, 4):  # knight/queen quiet
                    s += 120_000
            return s

        moves.sort(key=order, reverse=True)

        for mv in moves:
            u1 = pos.push(mv)
            rep[pos.key] = rep.get(pos.key, 0) + 1
            try:
                # immediate mate?
                replies = list(pos.legal_moves())
                if not replies and pos.in_check(pos.side):
                    memo[mkey] = True
                    return True
                if ply_left - 1 <= 0:
                    continue

                ok = True
                for r in replies:
                    u2 = pos.push(r)
                    rep[pos.key] = rep.get(pos.key, 0) + 1
                    try:
                        if not mate_in(ply_left - 2):
                            ok = False
                            break
                    finally:
                        rep[pos.key] -= 1
                        if rep[pos.key] <= 0:
                            rep.pop(pos.key, None)
                        pos.pop(u2)

                if ok and replies:
                    memo[mkey] = True
                    return True
            finally:
                rep[pos.key] -= 1
                if rep[pos.key] <= 0:
                    rep.pop(pos.key, None)
                pos.pop(u1)

        memo[mkey] = False
        return False

    root_moves = list(pos.legal_moves())
    if not root_moves:
        return None

    def root_order(m: Move) -> int:
        s = 0
        try:
            if _gives_check(pos, m):
                s += 10_000_000
        except Exception:
            pass
        if _is_capture(pos, m):
            try:
                s += 5_000_000 + see(pos, m)
            except Exception:
                s += 5_000_000
        else:
            p = pos.board[m.from_sq]
            if p >= 0 and (p % 6) in (1, 4):
                s += 250_000
        u = pos.push(m)
        try:
            s += -_eval_current(pos)
        finally:
            pos.pop(u)
        return s

    root_moves.sort(key=root_order, reverse=True)
    # Always include common quiet-mate starters early if they are legal.
    starters: list[Move] = []
    for uci in ("e2e5", "b1d2", "b1c3"):
        for m in root_moves:
            if m.uci() == uci:
                starters.append(m)
                break
    if starters:
        sset = {m.uci() for m in starters}
        root_moves = starters + [m for m in root_moves if m.uci() not in sset]
    root_moves = root_moves[:20]

    try:
        for mv in root_moves:
            if _time_up():
                break
            u = pos.push(mv)
            rep[pos.key] = rep.get(pos.key, 0) + 1
            try:
                if not list(pos.legal_moves()) and pos.in_check(pos.side):
                    print(f"[mate-mode] immediate mate: {mv.uci()} (budget={budget_s:.2f}s)")
                    return mv
                ok = mate_in(max_ply - 1)
                if debug:
                    print(f"[mate-mode] try {mv.uci()} -> {ok}")
                if ok:
                    print(f"[mate-mode] mate found (<= {max_ply} ply): {mv.uci()} (budget={budget_s:.2f}s)")
                    return mv
            finally:
                rep[pos.key] -= 1
                if rep[pos.key] <= 0:
                    rep.pop(pos.key, None)
                pos.pop(u)
    except _SearchTimeout:
        return None
    except Exception:
        return None

    return None


def _pawn_deflection_bonus(pos: Position, mv: Move, depth: int) -> int:
    """
    Mini tactical probe for quiet pawn pushes.
    If opponent can capture the advanced pawn and that capture enables an immediate,
    profitable capture for us (via SEE), bump the move.
    """
    if depth < 3:
        return 0
    if mv.promo is not None or mv.is_castle or mv.is_ep:
        return 0
    piece = pos.board[mv.from_sq]
    if piece < 0 or (piece % 6) != 0:
        return 0
    if pos.board[mv.to_sq] >= 0:
        return 0  # captures are handled elsewhere
    # Keep this heuristic focused: advanced central pawn pushes are the common
    # deflection/threat source (prevents bogus triggers like rook-pawn pushes).
    f = file_of(mv.to_sq)
    r = rank_of(mv.to_sq)
    if not (2 <= f <= 5):  # c..f files
        return 0
    if pos.side == WHITE:
        if r < 4:  # must reach at least 5th rank
            return 0
    else:
        if r > 3:  # must reach at most 4th rank for black
            return 0

    u1 = pos.push(mv)
    try:
        # Opponent captures the pawn we just advanced.
        replies = [r for r in pos.legal_captures() if r.to_sq == mv.to_sq]
        if not replies:
            return 0

        # Only treat it as a real pawn-deflection tactic when the capture is
        # also performed by a pawn. Otherwise we get many false positives
        # (e.g. queen/knight captures that don't actually deflect/lose material).
        pawn_replies = []
        for r in replies:
            cap_piece = pos.board[r.from_sq]
            if cap_piece >= 0 and (cap_piece % 6) == 0:  # pawn
                pawn_replies.append(r)
        if not pawn_replies:
            return 0
        replies = pawn_replies

        best_see = 0
        for r in replies:
            u2 = pos.push(r)
            try:
                # Our best immediate capture response (SEE-based).
                for c in pos.legal_captures():
                    # King captures confuse our cheap SEE (king "value" artifacts and
                    # legality/safety not modeled here). Ignore them for this heuristic.
                    p = pos.board[c.from_sq]
                    if p >= 0 and (p % 6) == 5:
                        continue
                    s = see(pos, c)
                    if s > best_see:
                        best_see = s
            finally:
                pos.pop(u2)

        # If the best response is clearly profitable, treat it as a tactical threat.
        if best_see >= 150:
            # Cap to avoid pathological scaling from any SEE oddities.
            if best_see > 1200:
                best_see = 1200
            return 300_000 + best_see * 300
        return 0
    finally:
        pos.pop(u1)


def _eval_current(pos: Position) -> int:
    """Evaluation from the perspective of the side to move (negamax-compatible)."""
    s = evaluate(pos)  # white-centric baseline
    nn_off = bool(_EVAL_NO_NNUE.get())
    sf_cp = None if nn_off else _sf_nnue_eval_white_cp(pos)
    if sf_cp is not None:
        try:
            a = float(os.environ.get("SF_NNUE_BLEND", _DEFAULT_SF_NNUE_BLEND))
        except Exception:
            a = 1.0
        a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
        s = int((1.0 - a) * s + a * sf_cp)

    # Your self-trained NNUE delta head (npz) applies after baseline, unless SF NNUE is enabled.
    if not nn_off and sf_cp is None and _nnue is not None and _nnue.enabled():
        a = _nnue.blend()
        if _nnue.kind() == "delta":
            # Safer: NNUE learns a correction term over the baseline.
            s = int(s + a * _nnue.score_white(pos))
        else:
            # Back-compat: absolute cp head blended with baseline.
            n = _nnue.score_white(pos)
            s = int((1.0 - a) * s + a * n)
    return s if pos.side == WHITE else -s


def _alpha_beta(
    pos: Position,
    depth: int,
    alpha: int,
    beta: int,
    nodes_ref: list[int],
    tt: TranspositionTable,
    killers: list[list[tuple[int, int, int] | None]],
    history: dict[tuple[int, int, int], int],
    rep: dict[int, int],
    ply: int,
    deadline: float | None,
    pv: bool,
    stop_event=None,
) -> int:
    nodes_ref[0] += 1
    if stop_event is not None and (nodes_ref[0] & 2047) == 0 and getattr(stop_event, "is_set", lambda: False)():
        raise _SearchTimeout()
    _raise_if_time_up(deadline, nodes_ref[0])

    # 50-move rule: 100 halfmoves with no pawn move/capture.
    if pos.halfmove >= 100:
        return _draw_score_stm(pos)
    # Threefold repetition: same position (key) three times (game history + search path).
    if rep.get(pos.key, 0) >= 3:
        return _draw_score_stm(pos)

    # Stockfish-style: if a third repetition is reachable within 2 plies, raise a
    # pessimistic lower bound toward the draw score when materially ahead (reduces
    # aimless shuffles that search as if the draw were still far away).
    if (
        ply > 0
        and (not pv)
        and _env_int("V4_UPCOMING_REP_EN", 1) != 0
    ):
        try:
            bal_up = _material_balance_cp(pos)
        except Exception:
            bal_up = 0
        min_adv = _env_int("V4_UPCOMING_REP_MIN_CP", 150)
        if bal_up >= min_adv and _upcoming_repetition_within_2ply(pos, rep):
            ds = _draw_score_stm(pos)
            if alpha < ds:
                alpha = ds
                if alpha >= beta:
                    return alpha

    moves = list(pos.legal_moves())
    if not moves:
        if pos.in_check(pos.side):
            return -(100_000 - ply)
        return 0

    if depth <= 0:
        return _quiescence(pos, alpha, beta, nodes_ref, tt, ply, deadline, stop_event=stop_event)

    key = pos.key
    hit, tt_score, tt_best = tt.probe(key, depth, alpha, beta)
    if hit:
        return tt_score

    # Internal iterative reduction (IIR): if no TT guidance and not PV, reduce depth a bit.
    # Avoid IIR near the root, where we must accurately compare alternatives (mates often
    # start with a quiet move that would otherwise get reduced and missed).
    if ply > 1 and (not pv) and tt_best is None and depth >= 4:
        depth -= 1

    # Basic ordering: TT move first, then captures.
    tt_uci = None if tt_best is None else tt_best.uci()

    def order_score(m: Move) -> int:
        k = _mv_key(m)
        if tt_uci is not None and m.uci() == tt_uci:
            return 10_000_000
        if _is_capture(pos, m):
            return 5_000_000 + see(pos, m) + _simplify_bonus(pos, m) + _tb_simplify_bonus(pos, m) + _queen_trade_bonus(pos, m)
        # Forcing moves: checks often decide tactics/mates; search them early.
        # Keep this below captures to avoid wasting time on empty checks.
        if ply < _MAX_PLY and depth >= 2:
            try:
                if _gives_check(pos, m):
                    return 4_800_000
            except Exception:
                pass
        # "Threat" heuristic: quiet pawn pushes near the enemy king.
        if ply < _MAX_PLY:
            b = _pawn_storm_bonus(pos, m, depth)
            if b:
                return 4_200_000 + b
        # Quiet pawn deflections that win material a ply later.
        if ply < _MAX_PLY:
            b = _pawn_deflection_bonus(pos, m, depth)
            if b:
                # Must stay strictly below the capture tier (5_000_000 + see): an uncapped
                # bonus can exceed e.g. 5_000_320 for Re4-e3 and sort a losing pawn push first.
                return 4_900_000 + min(b, 95_000)
        # Simplify when clearly winning: reward "trade offers" too.
        if ply < _MAX_PLY:
            b = _trade_offer_bonus(pos, m)
            if b:
                return 4_150_000 + min(b, 90_000)
        if ply < _MAX_PLY:
            if killers[ply][0] == k:
                return 4_000_000
            if killers[ply][1] == k:
                return 3_900_000
        return history.get(k, 0)

    moves.sort(key=order_score, reverse=True)

    orig_alpha = alpha
    best: Move | None = None

    for i, mv in enumerate(moves):
        storm = _pawn_storm_bonus(pos, mv, depth)
        defl = _pawn_deflection_bonus(pos, mv, depth)
        u = pos.push(mv)
        rep[pos.key] = rep.get(pos.key, 0) + 1
        try:
            # Conservative extensions
            ext = 1 if pos.in_check(pos.side) else 0  # giving-check extension
            # Threat extension (v4-only): quiet pawn pushes near the enemy king can be
            # forcing (mate nets, deflections, forks). Give them one extra ply.
            if storm and depth >= 3 and not _is_capture(pos, mv):
                ext += 1
            # Deflection extension: forcing tactical pawn pushes.
            if defl and depth >= 3 and not _is_capture(pos, mv):
                ext += 1

            # Simple LMR: reduce late quiet moves outside PV
            new_depth = depth - 1 + ext
            # Avoid LMR near the root for the same reason as IIR.
            if ply > 1 and (not pv) and depth >= 3 and i >= 4 and not _is_capture(pos, mv) and (not storm) and (not defl):
                new_depth = max(0, new_depth - 1)

            score = -_alpha_beta(
                pos,
                new_depth,
                -beta,
                -alpha,
                nodes_ref,
                tt,
                killers,
                history,
                rep,
                ply + 1,
                deadline,
                pv and i == 0,
                stop_event=stop_event,
            )
        finally:
            rep[pos.key] -= 1
            if rep[pos.key] <= 0:
                rep.pop(pos.key, None)
            pos.pop(u)

        if score >= beta:
            # update killers/history for quiet β cutoffs
            if (not _is_capture(pos, mv)) and ply < _MAX_PLY:
                _update_killers(killers, ply, mv)
                k = _mv_key(mv)
                history[k] = history.get(k, 0) + depth * depth
            tt.store(key, depth, beta, TT_LOWER, mv)
            return beta
        if score > alpha:
            alpha = score
            best = mv

    flag = TT_EXACT if alpha > orig_alpha else TT_UPPER
    tt.store(key, depth, alpha, flag, best)
    return alpha


def _quiescence(
    pos: Position,
    alpha: int,
    beta: int,
    nodes_ref: list[int],
    tt: TranspositionTable,
    ply: int,
    deadline: float | None,
    qdepth: int = 8,
    check_depth: int = 4,
    stop_event=None,
) -> int:
    nodes_ref[0] += 1
    if stop_event is not None and (nodes_ref[0] & 2047) == 0 and getattr(stop_event, "is_set", lambda: False)():
        raise _SearchTimeout()
    _raise_if_time_up(deadline, nodes_ref[0])

    stand_pat = _eval_current(pos)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if qdepth <= 0:
        return alpha
    if deadline is not None and (deadline - time.perf_counter()) < 0.010:
        return alpha

    caps = list(pos.legal_captures())
    caps.sort(key=lambda m: see(pos, m), reverse=True)

    for mv in caps:
        # delta pruning: skip clearly losing captures
        if see(pos, mv) < 0 and stand_pat + see(pos, mv) + 50 < alpha:
            continue
        u = pos.push(mv)
        try:
            score = -_quiescence(
                pos,
                -beta,
                -alpha,
                nodes_ref,
                tt,
                ply + 1,
                deadline,
                qdepth - 1,
                check_depth=check_depth,
                stop_event=stop_event,
            )
        finally:
            pos.pop(u)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    # Quiet checks: a small extra extension to avoid missing forcing lines
    # right after book / in tactical positions.
    if check_depth > 0:
        try:
            moves = list(pos.legal_moves())
        except Exception:
            moves = []
        checks = []
        for mv in moves:
            if _is_capture(pos, mv):
                continue
            u = pos.push(mv)
            try:
                # After making mv, it's opponent to move; if they are in check, mv is a checking move.
                if pos.in_check(pos.side):
                    checks.append(mv)
            finally:
                pos.pop(u)

        for mv in checks:
            u = pos.push(mv)
            try:
                score = -_quiescence(
                    pos,
                    -beta,
                    -alpha,
                    nodes_ref,
                    tt,
                    ply + 1,
                    deadline,
                    qdepth - 1,
                    check_depth=check_depth - 1,
                    stop_event=stop_event,
                )
            finally:
                pos.pop(u)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    return alpha


def _score_root_moves_for_substitute(
    pos: Position,
    moves: list[Move],
    depth_full: int,
    deadline: float | None,
    repetition_base: dict[int, int] | None,
    tt: TranspositionTable,
    stop_event=None,
) -> list[tuple[int, Move]]:
    """Score root moves independently (for second-best substitutions). Highest score best for STM."""
    if not moves:
        return []
    if depth_full < 2:
        return [(0, mv) for mv in moves]

    nodes_ref = [0]
    killers: list[list[tuple[int, int, int] | None]] = [[None, None] for _ in range(_MAX_PLY)]
    history: dict[tuple[int, int, int], int] = {}
    d_child = depth_full - 1
    ranked: list[tuple[int, Move]] = []

    rep0 = dict(repetition_base) if repetition_base else {pos.key: 1}

    for mv in moves:
        _raise_if_time_up(deadline, nodes_ref[0])
        u = pos.push(mv)
        rep0[pos.key] = rep0.get(pos.key, 0) + 1
        try:
            sc = -_alpha_beta(
                pos,
                d_child,
                -10**9,
                10**9,
                nodes_ref,
                tt,
                killers,
                history,
                rep0,
                1,
                deadline,
                True,
                stop_event=stop_event,
            )
        except _SearchTimeout:
            sc = -10**18
        finally:
            rep0[pos.key] -= 1
            if rep0[pos.key] <= 0:
                rep0.pop(pos.key, None)
            pos.pop(u)
        ranked.append((sc, mv))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


class RandomAgent:
    """Level 1 — uniform random legal move."""

    level = 1
    name = "L1 Random"
    last_info_string: str | None = None

    def choose_move(
        self,
        pos,
        time_limit_s: float | None = None,
        stop_event=None,
        repetition_base: dict[int, int] | None = None,
    ):
        del time_limit_s, repetition_base
        self.last_info_string = None
        moves = list(pos.legal_moves())
        if not moves:
            return None
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            return None
        return random.choice(moves)

    def ponder(self, *args, **kwargs) -> None:
        return None


class MinimaxAgent:
    def __init__(self, level: int):
        self.level = max(2, min(10, level))
        self._skill = _skill_for_level(self.level)
        self.base_depth = 7  # nominal v4 deepening cap (levels use SkillConfig clamps)
        self._tt = TranspositionTable()
        self.last_info_string: str | None = None

    @property
    def name(self) -> str:
        return f"v4-L{self.level}"

    def ponder(self, pos: Position, time_limit_s: float | None = 0.5, stop_event=None) -> None:
        """Search-only warmup while the opponent is thinking (fills TT — thread-safe TT lock)."""
        tok_nn = None
        if self._skill.disable_nnue:
            tok_nn = _EVAL_NO_NNUE.set(True)
        try:
            self._choose_move_inner(
                pos, time_limit_s, stop_event, None, pondering=True
            )
        finally:
            if tok_nn is not None:
                _EVAL_NO_NNUE.reset(tok_nn)

    def choose_move(
        self,
        pos: Position,
        time_limit_s: float | None = None,
        stop_event=None,
        repetition_base: dict[int, int] | None = None,
    ) -> Move | None:
        tok_nn = None
        if self._skill.disable_nnue:
            tok_nn = _EVAL_NO_NNUE.set(True)
        try:
            return self._choose_move_inner(
                pos, time_limit_s, stop_event, repetition_base, pondering=False
            )
        finally:
            if tok_nn is not None:
                _EVAL_NO_NNUE.reset(tok_nn)

    def _choose_move_inner(
        self,
        pos: Position,
        time_limit_s: float | None = None,
        stop_event=None,
        repetition_base: dict[int, int] | None = None,
        *,
        pondering: bool = False,
    ) -> Move | None:
        self.last_info_string = None

        cfg = self._skill

        bm = (
            pick_book_move(pos)
            if (not pondering and not cfg.no_opening_book)
            else None
        )
        if bm is not None:
            return bm

        if not pondering and not cfg.no_exp_book:
            try:
                uci = _experience_book_lookup_uci(pos)
            except Exception:
                uci = None
            if uci:
                for m in pos.legal_moves():
                    if m.uci() == uci:
                        return m

        if not pondering:
            try:
                eg = _endgame_tb_mate_mode_bestmove(pos)
            except Exception:
                eg = None
            if eg is not None:
                self.last_info_string = (self.last_info_string + " " if self.last_info_string else "") + f"endgame_tb {eg.uci()}"
                return eg

        syzygy_candidates: set[str] | None = None
        if not pondering and not cfg.no_syzygy:
            try:
                force_syzygy = _env_int("V4_SYZYGY_FORCE", 1) != 0
            except Exception:
                force_syzygy = True
            if force_syzygy:
                try:
                    bm1 = probe_bestmove(pos.to_fen())
                except Exception:
                    bm1 = None
                if bm1:
                    for m in pos.legal_moves():
                        if m.uci() == bm1:
                            try:
                                from syzygy_345 import probe_info as _probe_info  # type: ignore

                                inf = _probe_info(pos.to_fen())
                            except Exception:
                                inf = None
                            if isinstance(inf, dict):
                                self.last_info_string = (
                                    f"syzygy force pieces={inf.get('pieces')} "
                                    f"wdl={inf.get('wdl')} best={inf.get('bestmove')} "
                                    f"best_wdl={inf.get('best_wdl')} best_dtz={inf.get('best_dtz')}"
                                )
                            else:
                                self.last_info_string = f"syzygy force best={bm1}"
                            return m
            try:
                cands = probe_bestmoves(pos.to_fen())
            except Exception:
                cands = None
            if cands:
                if len(cands) == 1:
                    bm_uci = cands[0]
                    for m in pos.legal_moves():
                        if m.uci() == bm_uci:
                            try:
                                from syzygy_345 import probe_info as _probe_info  # type: ignore

                                inf = _probe_info(pos.to_fen())
                            except Exception:
                                inf = None
                            if isinstance(inf, dict):
                                self.last_info_string = (
                                    f"syzygy hit pieces={inf.get('pieces')} "
                                    f"wdl={inf.get('wdl')} best={inf.get('bestmove')} "
                                    f"best_wdl={inf.get('best_wdl')} best_dtz={inf.get('best_dtz')}"
                                )
                            else:
                                self.last_info_string = f"syzygy hit best={bm_uci}"
                            return m
                else:
                    syzygy_candidates = set(cands)
                    self.last_info_string = f"syzygy tie n={len(cands)}"

        moves = list(pos.legal_moves())
        if not moves:
            return None
        if len(moves) == 1:
            if pondering:
                return None
            return moves[0]

        if syzygy_candidates:
            filtered = [m for m in moves if m.uci() in syzygy_candidates]
            if filtered:
                moves = filtered

        # Root safety: if repetition history is known, don't pick a 1-ply line that
        # would immediately end the game as a threefold when we're up material.
        try:
            before = len(moves)
            moves = _root_avoid_immediate_threefold(pos, moves, repetition_base=repetition_base)
            if before and len(moves) < before:
                self.last_info_string = (self.last_info_string + " " if self.last_info_string else "") + (
                    f"root_rep_filter {before}->{len(moves)}"
                )
        except Exception:
            pass

        # Simple conversion rule: if we're ahead and can immediately trade queens
        # safely, do it. This avoids perpetual-check resources and simplifies winning
        # rook endgames (e.g. exchange-up positions).
        if not pondering:
            try:
                if _material_balance_cp(pos) >= 200 and _king_comfort_safe(pos):
                    for mv in moves:
                        if _queen_trade_bonus(pos, mv) > 0:
                            return mv
            except Exception:
                pass

        # Time-trouble endgame shortcut: if we're winning and can capture into <= 5 pieces,
        # do it quickly so the remainder can be tablebase-driven (even if "unequal").
        if (not pondering) and time_limit_s is not None and time_limit_s <= 0.20:
            try:
                best_tb: Move | None = None
                best_tb_score = -10**18
                for mv in moves:
                    b = _tb_simplify_bonus(pos, mv, time_limit_s=time_limit_s)
                    if b <= 0:
                        continue
                    u = pos.push(mv)
                    try:
                        tbm = None
                        if not self._skill.no_syzygy:
                            try:
                                tbm = probe_bestmove(pos.to_fen())
                            except Exception:
                                tbm = None
                        s = b + (300_000 if tbm else 0)
                        if s > best_tb_score:
                            best_tb_score = s
                            best_tb = mv
                    finally:
                        pos.pop(u)
                if best_tb is not None:
                    return best_tb
            except Exception:
                pass

        # Optional mate mode (prints when it triggers).
        if not pondering:
            mm = _mate_mode_bestmove(pos, time_limit_s, repetition_base, stop_event=stop_event)
            if mm is not None:
                return mm

        def _apply_root_overrides(cur_best: Move) -> Move:
            # Only use these "cheap tactical" overrides in time trouble. Otherwise they
            # can dominate the real search in messy positions.
            if time_limit_s is None or time_limit_s > 0.25:
                return cur_best
            try:
                best_tb: Move | None = None
                best_tb_score = 0
                for mv in moves:
                    b = _tb_simplify_bonus(pos, mv, time_limit_s=time_limit_s)
                    if b > best_tb_score:
                        best_tb_score = b
                        best_tb = mv
                if best_tb is not None and best_tb_score >= 350_000:
                    return best_tb
            except Exception:
                pass
            # Root tactical override: if we strongly detect a pawn-deflection tactic
            # (opponent can capture the pawn and we have a clear profitable capture),
            # prefer it under time control even if shallow depth hasn't flipped yet.
            try:
                best_defl = 0
                best_defl_mv: Move | None = None
                for mv in moves:
                    b = _pawn_deflection_bonus(pos, mv, depth=4)
                    if b > best_defl:
                        best_defl = b
                        best_defl_mv = mv
                if best_defl_mv is not None and best_defl >= 500_000:
                    return best_defl_mv
            except Exception:
                pass
            return cur_best

        # Safety margin: aim to stop a bit before the external time limit.
        # This is important because the search only checks the deadline periodically.
        if time_limit_s is not None and time_limit_s > 0:
            try:
                overhead_ms = int(os.environ.get("V4_MOVE_OVERHEAD_MS", "120"))
            except Exception:
                overhead_ms = 120
            overhead_s = max(0.0, min(1.0, overhead_ms / 1000.0))
            budget_s = max(0.0, time_limit_s - overhead_s)
            # Additional fractional safety.
            budget_s = max(0.0, budget_s * 0.92)
            deadline = time.perf_counter() + budget_s
        else:
            deadline = None
        nodes_ref = [0]
        killers: list[list[tuple[int, int, int] | None]] = [[None, None] for _ in range(_MAX_PLY)]
        history: dict[tuple[int, int, int], int] = {}

        # If we ever hit a timeout early, prefer a reasonable "opening-ish" move
        # instead of the first movegen move.
        best = moves[0]
        best_static = -10**9
        # Seed history with shallow static ordering so quiet forcing moves (often mate starters)
        # are searched earlier under time pressure.
        root_hist: dict[tuple[int, int, int], int] = {}
        for mv in moves:
            defl = _pawn_deflection_bonus(pos, mv, depth=4)
            tb_simpl = _tb_simplify_bonus(pos, mv, time_limit_s=time_limit_s)
            qtrade = _queen_trade_bonus(pos, mv)
            u = pos.push(mv)
            # child position is from opponent perspective; negate to compare root moves
            sc = -_eval_current(pos)
            # If we time out early, prefer moves with obvious tactical payoffs.
            sc += defl // 3000
            sc += tb_simpl // 6000
            sc += qtrade // 6000
            pos.pop(u)
            root_hist[_mv_key(mv)] = sc
            if sc > best_static:
                best_static = sc
                best = mv
        history.update(root_hist)

        # Low-time strategy: don't start expensive ID/aspiration when time is tiny.
        # In 10+0 this matters in time trouble: better a decent move fast than a blunder late.
        if time_limit_s is not None and time_limit_s <= 0.05:
            return _apply_root_overrides(best)

        last_score = 0
        last_best_uci: str | None = None
        stable_iters = 0
        t_start = time.perf_counter()
        info = _search_info_enabled()

        # Don't "easy-stop" too early in tactical positions (exposed kings, forcing threats).
        # This prevents missing quiet deflections like e6 (threat + win material) that only
        # show up a ply later.
        tactical_root = False
        try:
            for m in moves:
                if _is_capture(pos, m):
                    continue
                if _gives_check(pos, m):
                    tactical_root = True
                    break
                if _pawn_storm_bonus(pos, m, depth=2):
                    tactical_root = True
                    break
        except Exception:
            tactical_root = False

        # Iterative deepening + aspiration windows
        # If we have a real clock, allow deeper search; mates in 4/5 need ~8–10 plies
        # and "easy-stop" can otherwise freeze too early.
        max_depth = self.base_depth + (12 if deadline is not None else 0)
        if cfg.quick_depth_cap is not None:
            max_depth = min(max_depth, cfg.quick_depth_cap)
        if cfg.shallow_one_prob > 0 and random.random() < cfg.shallow_one_prob:
            max_depth = min(max_depth, 1)

        low_time = time_limit_s is not None and time_limit_s <= 0.20
        last_finished_depth = 0
        for d in range(1, max_depth + 1):
            if low_time:
                # Avoid aspiration re-search loops under low time.
                alpha, beta = -10**9, 10**9
                try:
                    rep = dict(repetition_base) if repetition_base else {pos.key: 1}
                    score = _alpha_beta(pos, d, alpha, beta, nodes_ref, self._tt, killers, history, rep, 0, deadline, True, stop_event=stop_event)
                except _SearchTimeout:
                    return _apply_root_overrides(best)
                last_score = score
            else:
                alpha = last_score - 50
                beta = last_score + 50
                while True:
                    try:
                        rep = dict(repetition_base) if repetition_base else {pos.key: 1}
                        score = _alpha_beta(pos, d, alpha, beta, nodes_ref, self._tt, killers, history, rep, 0, deadline, True, stop_event=stop_event)
                    except _SearchTimeout:
                        return _apply_root_overrides(best)
                    if score <= alpha:
                        alpha -= 200
                        continue
                    if score >= beta:
                        beta += 200
                        continue
                    last_score = score
                    break

            _, _, tt_best = self._tt.probe(pos.key, 0, -10**9, 10**9)
            if tt_best is not None:
                # TT best might be stale (hash collision or key bug); only accept if legal now.
                tuci = tt_best.uci()
                for m in moves:
                    if m.uci() == tuci:
                        best = m
                        break

            if info:
                try:
                    elapsed = time.perf_counter() - t_start
                    nps = int(nodes_ref[0] / max(1e-9, elapsed))
                    print(
                        f"[v4-info] depth={d} score={last_score} best={best.uci()} "
                        f"nodes={nodes_ref[0]} nps={nps} elapsed={elapsed:.2f}s"
                    )
                except Exception:
                    pass

            last_finished_depth = d

            # Dynamic stopping ("easy move") — see submission v3/agents.py.
            if deadline is not None and time_limit_s and time_limit_s > 0:
                cur_uci = best.uci()
                if last_best_uci is not None and cur_uci == last_best_uci and abs(last_score - score) <= 15:
                    stable_iters += 1
                else:
                    stable_iters = 0
                last_best_uci = cur_uci
                last_score = score

                elapsed = time.perf_counter() - t_start
                min_spend = 0.05 if len(moves) <= 3 else 0.10
                frac = 0.18 if len(moves) <= 3 else (0.28 if len(moves) <= 8 else 0.38)
                # In tactical roots (any checking move exists), don't stop before we
                # could reasonably see short mates.
                min_depth_for_easy_stop = 9 if tactical_root else 4
                if d >= min_depth_for_easy_stop and stable_iters >= 2 and elapsed >= max(min_spend, time_limit_s * frac):
                    break

        if pondering:
            return None

        pick = best
        want_sub = cfg.always_second_best or (
            cfg.second_best_prob > 0 and random.random() < cfg.second_best_prob
        )
        if want_sub and len(moves) >= 2 and last_finished_depth > 0:
            rank_d = max(4, min(7, last_finished_depth))
            if deadline is None or (deadline - time.perf_counter()) > 0.06:
                try:
                    ranked = _score_root_moves_for_substitute(
                        pos, moves, rank_d, deadline, repetition_base, self._tt, stop_event
                    )
                    if len(ranked) >= 2:
                        pick = ranked[1][1]
                except (_SearchTimeout, Exception):
                    pass

        return _apply_root_overrides(pick)


def get_agent(level: int):
    lv = max(1, min(10, int(level)))
    if lv <= 1:
        return RandomAgent()
    return MinimaxAgent(lv)


def run_match(
    agent_w,
    agent_b,
    *,
    verbose: bool = False,
    max_moves: int = 600,
    time_limit_s: float = 2.0,
) -> str:
    """Play one game; returns ``'white'``, ``'black'``, or ``'draw'``."""
    from board import GameState, STARTING_FEN, WHITE

    gs = GameState.from_fen(STARTING_FEN)
    for ply in range(max_moves):
        pos = Position.from_fen(gs.to_fen())
        ag = agent_w if gs.turn == WHITE else agent_b
        pm = ag.choose_move(pos, time_limit_s=time_limit_s, repetition_base=None)
        if pm is None:
            if verbose:
                side = "w" if gs.turn == WHITE else "b"
                print(f"ply {ply}: no move ({side} to play)")
            return "draw"
        uci = pm.uci()
        bm = None
        for m in gs.get_legal_moves():
            if m.uci() == uci:
                bm = m
                break
        if bm is None:
            if verbose:
                print(f"ply {ply}: engine returned non-legal {uci!r}")
            return "draw"
        gs = gs.make_move(bm)
        if verbose:
            print(f"  {ply + 1}. {uci}")
        r = gs.game_result()
        if r == "white":
            return "white"
        if r == "black":
            return "black"
        if r == "draw":
            return "draw"
    return "draw"


def benchmark(
    level: int,
    games: int,
    *,
    verbose: bool = True,
    time_limit_s: float = 1.5,
) -> tuple[int, int, int]:
    """
    Run ``games`` with ``get_agent(level)`` as White vs L1 Random as Black.
    Returns ``(wins, draws, losses)`` from White's perspective.
    """
    wins = draws = losses = 0
    for gi in range(games):
        agent_w = get_agent(level)
        agent_b = get_agent(1)
        if verbose:
            print(f"--- game {gi + 1}/{games} ---")
        r = run_match(
            agent_w, agent_b, verbose=False, time_limit_s=time_limit_s
        )
        if r == "white":
            wins += 1
        elif r == "black":
            losses += 1
        else:
            draws += 1
        if verbose:
            print(f"result: {r}\n")
    return wins, draws, losses

