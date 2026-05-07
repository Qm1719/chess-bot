"""
analyze_fen.py (v5)

Quick analysis driver for your submission v5 engine from a single FEN.

Configure these at the top:
- FEN: position to analyze
- MODE: "time", "depth", or "both"
- TIME_LIMIT_S: seconds for choose_move()
- DEPTH: fixed depth for alpha-beta (ply depth)

Run:
  py analyze_fen.py
"""

from __future__ import annotations

import os
import sys
import time

# ──────────────────────────────────────────────────────────────────────────────
# Config (edit these)

# 5b2/8/r5k1/p4p2/P2R4/2PK4/1P4P1/8 b - - 0 40
# rn1qk2r/pQ2b3/2p2pp1/8/8/2N5/PPP2PPP/R3K2R b KQkq - 4 17
# rn1qkb1r/pp3pp1/2p1pnp1/8/2BP4/2N5/PPP1QPPP/R1B1K2R w KQkq - 0 10

FEN = "8/8/4K3/8/8/8/3Q4/k7 w - - 11 90"
MODE = "time"  # "time" | "depth" | "both"

TIME_LIMIT_S = 60
DEPTH = 4  # ply depth (>= 1). Depth 7-9 can be slow.
TOP_N = 3  # how many root moves to print for depth mode
PRINT_TOP_N_IN_TIME_MODE = True  # print top-N from time-bounded search (MultiPV-ish)
TIME_MODE_MAX_DEPTH = 99  # hard cap for time-bounded iterative deepening

LEVEL = 10  # engine level (affects choose_move max depth)
ENGINE_INFO = True  # set V4_SEARCH_INFO=1 to print depth/score/nodes during choose_move
USE_EXPERIENCE_BOOK = True  # use experience_book.sqlite3 if present

# ──────────────────────────────────────────────────────────────────────────────


def _add_paths() -> None:
    v5_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    engine = os.path.join(v5_root, "engine")
    data = os.path.join(v5_root, "data")
    for p in (engine, data, v5_root):
        if p not in sys.path:
            sys.path.insert(0, p)


def bestmove_by_time(fen: str, time_limit_s: float) -> str | None:
    from position import Position  # type: ignore
    from agents_v4 import get_agent  # type: ignore

    pos = Position.from_fen(fen)
    agent = get_agent(LEVEL)
    if ENGINE_INFO:
        os.environ["V4_SEARCH_INFO"] = "1"
    if USE_EXPERIENCE_BOOK:
        os.environ["V4_EXPERIENCE_BOOK"] = "1"
    t0 = time.perf_counter()
    mv = agent.choose_move(pos, time_limit_s=time_limit_s, repetition_base={pos.key: 1})
    dt = time.perf_counter() - t0
    print(f"[time]  think={time_limit_s:.2f}s elapsed={dt:.2f}s bestmove={mv.uci() if mv else None}")
    return mv.uci() if mv else None


def top_moves_by_time(fen: str, time_limit_s: float, top_n: int) -> list[tuple[str, int, int]]:
    """
    Time-bounded iterative deepening at the root using the engine's alpha-beta.
    Returns list of (uci, score, depth_reached) from the last fully-completed depth.
    Score is in centipawns from the side-to-move perspective.
    """
    from position import Position  # type: ignore
    from agents_v4 import _alpha_beta, _MAX_PLY, _SearchTimeout  # type: ignore
    from tt import TranspositionTable  # type: ignore

    pos = Position.from_fen(fen)
    root_moves = list(pos.legal_moves())
    if not root_moves:
        return []

    deadline = time.perf_counter() + max(0.0, float(time_limit_s))
    tt = TranspositionTable()

    # Start with a stable deterministic order.
    root_order = [m for m in root_moves]
    last_completed: list[tuple[str, int, int]] = []

    for depth in range(1, max(1, int(TIME_MODE_MAX_DEPTH)) + 1):
        if time.perf_counter() >= deadline:
            break

        scored: list[tuple[int, str]] = []
        # If we have previous results, search best-first for better cutoffs.
        if last_completed:
            prev_rank = {uci: i for i, (uci, _, _) in enumerate(last_completed)}
            root_order.sort(key=lambda m: prev_rank.get(m.uci(), 9999))

        try:
            for mv in root_order:
                nodes_ref = [0]
                killers = [[None, None] for _ in range(_MAX_PLY)]
                history: dict[tuple[int, int, int], int] = {}
                rep = {pos.key: 1}

                u = pos.push(mv)
                rep[pos.key] = rep.get(pos.key, 0) + 1
                try:
                    score = -_alpha_beta(
                        pos,
                        depth - 1,
                        -10**9,
                        10**9,
                        nodes_ref,
                        tt,
                        killers,
                        history,
                        rep,
                        1,
                        deadline,
                        False,
                    )
                finally:
                    rep[pos.key] -= 1
                    if rep[pos.key] <= 0:
                        rep.pop(pos.key, None)
                    pos.pop(u)

                scored.append((int(score), mv.uci()))
        except _SearchTimeout:
            # Don't use partial depth results; keep last completed.
            break

        scored.sort(reverse=True, key=lambda x: x[0])
        last_completed = [(uci, s, depth) for (s, uci) in scored[: max(1, int(top_n))]]

    return last_completed


def bestmove_by_depth(fen: str, depth: int) -> tuple[str | None, int | None]:
    """
    Fixed-depth root search using internal alpha-beta.
    Returns (uci, score_cp_from_side_to_move).
    """
    from position import Position  # type: ignore
    from agents_v4 import _alpha_beta, _MAX_PLY  # type: ignore
    from tt import TranspositionTable  # type: ignore

    if depth < 1:
        raise ValueError("DEPTH must be >= 1")

    pos = Position.from_fen(fen)
    moves = list(pos.legal_moves())
    if not moves:
        print("[depth] no legal moves")
        return None, None

    scored: list[tuple[int, str]] = []

    # Keep a TT across root moves to speed up analysis.
    tt = TranspositionTable()

    t0 = time.perf_counter()
    for mv in moves:
        nodes_ref = [0]
        killers = [[None, None] for _ in range(_MAX_PLY)]
        history: dict[tuple[int, int, int], int] = {}
        rep = {pos.key: 1}

        u = pos.push(mv)
        rep[pos.key] = rep.get(pos.key, 0) + 1
        try:
            # Score from root perspective.
            score = -_alpha_beta(
                pos,
                depth - 1,
                -10**9,
                10**9,
                nodes_ref,
                tt,
                killers,
                history,
                rep,
                1,
                None,
                False,
            )
        finally:
            rep[pos.key] -= 1
            if rep[pos.key] <= 0:
                rep.pop(pos.key, None)
            pos.pop(u)

        scored.append((int(score), mv.uci()))

    dt = time.perf_counter() - t0
    scored.sort(reverse=True, key=lambda x: x[0])
    best_score, best_uci = scored[0]
    print(f"[depth] depth={depth} elapsed={dt:.2f}s top{TOP_N}:")
    for i, (s, uci) in enumerate(scored[: max(1, TOP_N)], start=1):
        print(f"  {i}. {uci}  score={s}")
    return best_uci, int(best_score)


def main() -> None:
    _add_paths()

    mode = MODE.strip().lower()
    if mode not in {"time", "depth", "both"}:
        raise ValueError('MODE must be "time", "depth", or "both"')

    fen = FEN
    # Optional CLI override:
    #   python tools/analyze_fen.py "<fen>"
    if len(sys.argv) >= 2:
        fen = " ".join(sys.argv[1:]).strip()

    print("fen", fen)
    try:
        from position import Position, WHITE  # type: ignore
        from evaluation import evaluate  # type: ignore
        from agents_v4 import _eval_current  # type: ignore

        pos0 = Position.from_fen(fen)
        cp_white = int(evaluate(pos0))
        cp_stm = int(_eval_current(pos0))
        stm = "w" if pos0.side == WHITE else "b"
        print(f"[static] stm={stm} eval_stm_cp={cp_stm} eval_white_cp={cp_white}")
    except Exception:
        pass
    if mode in {"time", "both"}:
        bestmove_by_time(fen, TIME_LIMIT_S)
        if mode == "time" and PRINT_TOP_N_IN_TIME_MODE:
            t0 = time.perf_counter()
            top = top_moves_by_time(fen, TIME_LIMIT_S, TOP_N)
            dt = time.perf_counter() - t0
            if top:
                d = top[0][2]
                print(f"[time-top] elapsed={dt:.2f}s last_completed_depth={d} top{TOP_N}:")
                for i, (uci, s, _d) in enumerate(top, start=1):
                    print(f"  {i}. {uci}  score={s}")
            else:
                print("[time-top] no legal moves")
    if mode in {"depth", "both"}:
        bestmove_by_depth(fen, DEPTH)


if __name__ == "__main__":
    main()

