"""
tournament_train.py

Train the experience book from tournament PGNs.

Pipeline:
1) Read PGNs (default: submission v5/testbot/out/results.pgn)
2) Replay every move and use Stockfish depth 12 to flag played moves as:
   - mistake: best_cp - played_cp >= 80
   - blunder: best_cp - played_cp >= 200
3) For each flagged pre-move FEN, ask current agents_v4 for a move with 10s think time
4) Re-evaluate that agents_v4 move with Stockfish depth 18
5) If the agents_v4 move is still a mistake/blunder, add the Stockfish best move to experience_book

Outputs:
- text report: submission v5/tools/tournament_train_report.txt
- optional additions to data/experience_book.sqlite3
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# -------------------------
# Fast-edit config
# -------------------------

PGN_PATH = os.environ.get("TOURNAMENT_PGN_PATH", "").strip()
PASS1_DEPTH = int(os.environ.get("TOURNAMENT_TRAIN_PASS1_DEPTH", "12"))
PASS2_DEPTH = int(os.environ.get("TOURNAMENT_TRAIN_PASS2_DEPTH", "18"))
AGENT_THINK_S = float(os.environ.get("TOURNAMENT_TRAIN_AGENT_TIME_S", "10"))
MISTAKE_DROP_CP = int(os.environ.get("MISTAKE_DROP_CP", "80"))
BLUNDER_DROP_CP = int(os.environ.get("BLUNDER_DROP_CP", "200"))
WRITE_EXP_BOOK = os.environ.get("TOURNAMENT_TRAIN_WRITE", "1").strip().lower() in {"1", "true", "yes", "on"}
MAX_FLAGGED = int(os.environ.get("TOURNAMENT_TRAIN_MAX_FLAGGED", "0"))  # 0 = unlimited


def _add_paths() -> tuple[str, str]:
    here = os.path.abspath(os.path.dirname(__file__))
    v5 = os.path.abspath(os.path.join(here, ".."))
    engine = os.path.join(v5, "engine")
    data = os.path.join(v5, "data")
    for p in (engine, data, v5):
        if p not in sys.path:
            sys.path.insert(0, p)
    return here, v5


HERE, V5_ROOT = _add_paths()


def _detect_stockfish_path() -> str:
    p = os.environ.get("STOCKFISH_PATH", "").strip()
    if p:
        return p
    try:
        from stockfish_path import detect_stockfish_path  # type: ignore

        return detect_stockfish_path()
    except Exception:
        return "stockfish"


def _cp_pov_white(score, *, pov_white: bool, mate_cp: int = 100000) -> int:
    try:
        s = score.pov(True if pov_white else False)
    except Exception:
        s = score
    v = s.score(mate_score=mate_cp)
    return int(v if v is not None else 0)


def _classify(drop_cp: int) -> str:
    if drop_cp >= BLUNDER_DROP_CP:
        return "blunder"
    if drop_cp >= MISTAKE_DROP_CP:
        return "mistake"
    return "ok"


@dataclass
class FlaggedMove:
    game_idx: int
    round_tag: str
    ply: int
    side: str
    played_uci: str
    best_uci_d12: str
    best_cp_d12: int
    played_cp_d12: int
    drop_cp_d12: int
    label_d12: str
    fen_before: str


@dataclass
class AgentReview:
    flagged: FlaggedMove
    agent_uci: str
    best_uci_d18: str
    best_cp_d18: int
    agent_cp_d18: int
    drop_cp_d18: int
    label_d18: str
    added_to_exp_book: bool
    keyhex: str


def _load_pgn_games(pgn_path: str):
    import chess.pgn  # type: ignore

    out = []
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            out.append(g)
    return out


def _uci_from_move(move) -> str:
    return move.uci() if move is not None else "0000"


def _analyse_flagged_moves(pgn_path: str) -> list[FlaggedMove]:
    import chess  # type: ignore
    import chess.engine  # type: ignore

    games = _load_pgn_games(pgn_path)
    sf_path = _detect_stockfish_path()
    engine = chess.engine.SimpleEngine.popen_uci([sf_path])
    try:
        try:
            engine.configure({"Threads": 1, "Hash": 128})
        except Exception:
            pass

        flagged: list[FlaggedMove] = []
        for gi, game in enumerate(games, start=1):
            board = game.board()
            round_tag = str(game.headers.get("Round", "") or f"game{gi}")
            for ply, move in enumerate(game.mainline_moves(), start=1):
                fen_before = board.fen()
                side = "w" if board.turn == chess.WHITE else "b"

                info = engine.analyse(board, chess.engine.Limit(depth=PASS1_DEPTH))
                pv = info.get("pv", []) or []
                best_move = pv[0] if pv else None
                best_uci = _uci_from_move(best_move)
                best_cp = _cp_pov_white(info.get("score", None), pov_white=board.turn == chess.WHITE)

                san = board.san(move)
                board.push(move)
                played_cp = _cp_pov_white(engine.analyse(board, chess.engine.Limit(depth=PASS1_DEPTH)).get("score", None), pov_white=(side == "w"))
                played_uci = move.uci()
                drop = best_cp - played_cp
                label = _classify(drop)
                if label != "ok":
                    flagged.append(
                        FlaggedMove(
                            game_idx=gi,
                            round_tag=round_tag,
                            ply=ply,
                            side=side,
                            played_uci=played_uci,
                            best_uci_d12=best_uci,
                            best_cp_d12=best_cp,
                            played_cp_d12=played_cp,
                            drop_cp_d12=drop,
                            label_d12=label,
                            fen_before=fen_before,
                        )
                    )
                    if MAX_FLAGGED > 0 and len(flagged) >= MAX_FLAGGED:
                        return flagged
        return flagged
    finally:
        try:
            engine.quit()
        except Exception:
            pass


def _agent_move_for_fen(fen: str) -> tuple[str, str]:
    from position import Position  # type: ignore
    import agents_v4  # type: ignore

    pos = Position.from_fen(fen)
    keyhex = f"{int(pos.key) & ((1 << 64) - 1):016x}"
    agent = agents_v4.MinimaxAgent(10)
    old_book = getattr(agents_v4, "pick_book_move", None)
    try:
        # Avoid exp-book feedback loops while mining training positions.
        os.environ["V4_EXPERIENCE_BOOK"] = "0"
        mv = agent.choose_move(pos, time_limit_s=AGENT_THINK_S, repetition_base={pos.key: 1})
        return (mv.uci() if mv else "0000"), keyhex
    finally:
        if old_book is not None:
            setattr(agents_v4, "pick_book_move", old_book)


def _review_with_agent(flagged: list[FlaggedMove]) -> list[AgentReview]:
    import chess  # type: ignore
    import chess.engine  # type: ignore
    from experience_book import DB_PATH, _connect, add_entry  # type: ignore

    sf_path = _detect_stockfish_path()
    sf = chess.engine.SimpleEngine.popen_uci([sf_path])
    try:
        try:
            sf.configure({"Threads": 1, "Hash": 128})
        except Exception:
            pass

        con = _connect(DB_PATH)
        try:
            out: list[AgentReview] = []
            for fm in flagged:
                board = chess.Board(fm.fen_before)
                agent_uci, keyhex = _agent_move_for_fen(fm.fen_before)
                agent_mv = chess.Move.from_uci(agent_uci) if agent_uci != "0000" else None
                if agent_mv is None or agent_mv not in board.legal_moves:
                    agent_cp = -100000
                else:
                    board_after = board.copy(stack=False)
                    board_after.push(agent_mv)
                    agent_info = sf.analyse(board_after, chess.engine.Limit(depth=PASS2_DEPTH))
                    agent_cp = _cp_pov_white(agent_info.get("score", None), pov_white=board.turn == chess.WHITE)

                best_info = sf.analyse(board, chess.engine.Limit(depth=PASS2_DEPTH))
                best_pv = best_info.get("pv", []) or []
                best_move = best_pv[0] if best_pv else None
                best_uci = _uci_from_move(best_move)
                best_cp = _cp_pov_white(best_info.get("score", None), pov_white=board.turn == chess.WHITE)
                drop = best_cp - agent_cp
                label = _classify(drop)

                added = False
                if WRITE_EXP_BOOK and label != "ok" and best_uci != "0000":
                    add_entry(con, fm.fen_before, best_uci, depth=PASS2_DEPTH, score=best_cp)
                    added = True

                out.append(
                    AgentReview(
                        flagged=fm,
                        agent_uci=agent_uci,
                        best_uci_d18=best_uci,
                        best_cp_d18=best_cp,
                        agent_cp_d18=agent_cp,
                        drop_cp_d18=drop,
                        label_d18=label,
                        added_to_exp_book=added,
                        keyhex=keyhex,
                    )
                )
            return out
        finally:
            con.close()
    finally:
        try:
            sf.quit()
        except Exception:
            pass


def _default_pgn_path() -> str:
    return str(Path(V5_ROOT) / "testbot" / "out" / "results.pgn")


def _report_path() -> str:
    return str(Path(HERE) / "tournament_train_report.txt")


def main() -> int:
    pgn_path = PGN_PATH or _default_pgn_path()
    if not os.path.exists(pgn_path):
        print(f"missing pgn: {pgn_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    print(f"pgn={pgn_path}")
    print(f"stockfish={_detect_stockfish_path()} pass1_depth={PASS1_DEPTH} pass2_depth={PASS2_DEPTH}")
    print(f"thresholds: mistake>={MISTAKE_DROP_CP} blunder>={BLUNDER_DROP_CP} agent_think_s={AGENT_THINK_S}")
    print(f"write_exp_book={int(WRITE_EXP_BOOK)}")

    flagged = _analyse_flagged_moves(pgn_path)
    print(f"flagged_moves={len(flagged)}")

    reviews = _review_with_agent(flagged)
    added = sum(1 for r in reviews if r.added_to_exp_book)
    mistakes = sum(1 for r in reviews if r.label_d18 == "mistake")
    blunders = sum(1 for r in reviews if r.label_d18 == "blunder")

    lines: list[str] = []
    lines.append(f"pgn={pgn_path}")
    lines.append(f"stockfish={_detect_stockfish_path()} pass1_depth={PASS1_DEPTH} pass2_depth={PASS2_DEPTH}")
    lines.append(f"flagged_moves={len(flagged)} reviewed={len(reviews)} added_to_exp_book={added}")
    lines.append(f"agent_reply_mistakes={mistakes} agent_reply_blunders={blunders}")
    lines.append("")
    for i, r in enumerate(reviews, start=1):
        fm = r.flagged
        lines.append(
            f"[{i}] {fm.round_tag} ply={fm.ply} side={fm.side} "
            f"played={fm.played_uci} sf12_best={fm.best_uci_d12} drop12={fm.drop_cp_d12}({fm.label_d12})"
        )
        lines.append(
            f"    fen={fm.fen_before}"
        )
        lines.append(
            f"    agent_v4={r.agent_uci} sf18_best={r.best_uci_d18} "
            f"best18={r.best_cp_d18} agent18={r.agent_cp_d18} drop18={r.drop_cp_d18}({r.label_d18}) "
            f"key={r.keyhex} add={int(r.added_to_exp_book)}"
        )
        lines.append("")

    report = _report_path()
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"reviewed={len(reviews)} added={added} report={report} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

