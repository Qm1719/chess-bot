"""
two_pass_local_sf_test.py

Two-pass local Stockfish analysis for one Lichess game ID.

What it does:
- Fetches game JSON from Lichess (`/game/export/{id}` Accept: application/json).
- Replays moves from startpos.
- For *bot side only* plies:
  Pass 1 (depth=PASS1_DEPTH):
    - evaluate best line at the pre-move position
    - evaluate resulting position after the played move
    - compute centipawn drop and classify mistake/blunder by thresholds
  Pass 2 (depth=PASS2_DEPTH) on flagged plies only:
    - re-evaluate best vs played and print final recommended best move
    - print the exact (fen, best_uci) pairs you'd add to experience_book

Notes:
- Scores are normalized to "bot POV centipawns" using python-chess Score.pov().
- Uses optional `psutil` if installed to report RSS of Python + Stockfish process.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


# -------------------------
# Fast-edit config (change here)
# -------------------------

GAME_ID = os.environ.get("SMOKE_GAME_ID", "y6uQvMkg").strip()
BOT_USER = (
    os.environ.get("LICHESS_BOT_NAME", "").strip()
    or os.environ.get("SELFTRAIN_BOT_USER", "nmai252").strip()
)

# Lichess token is optional for many exports; set if you hit rate limits / private games.
_LICHESS_TOKEN_RAW = (
    os.environ.get("LICHESS_TOKEN", "").strip()
    or os.environ.get("LICHESS_BOT_TOKEN", "").strip()
    or os.environ.get("LICHESS_API_TOKEN", "").strip()
    or ""
).strip()
LICHESS_TOKEN = _LICHESS_TOKEN_RAW or None

LICHESS_BASE = os.environ.get("LICHESS_BASE", "https://lichess.org").rstrip("/")

# Stockfish UCI engine path.
# Paste a full path here if you want fastest iteration, e.g.
#   STOCKFISH_PATH = r"D:\engines\stockfish\stockfish-windows-x86-64-avx2.exe"
# If left empty, we'll auto-detect:
#   1) STOCKFISH_PATH env
#   2) submission v5/engine/stockfish/ (OS-specific)
#   3) repo_root/stockfish/ (older layout)
#   4) "stockfish" (PATH)
STOCKFISH_PATH = r""

# Pass depths (fixed depth for reproducibility)
PASS1_DEPTH = int(os.environ.get("PASS1_DEPTH", "12"))
PASS2_DEPTH = int(os.environ.get("PASS2_DEPTH", "18"))

# Classification thresholds based on (best_cp - played_cp), bot POV.
MISTAKE_DROP_CP = int(os.environ.get("MISTAKE_DROP_CP", "80"))
BLUNDER_DROP_CP = int(os.environ.get("BLUNDER_DROP_CP", "200"))

# If true, print every bot ply line in pass 1; else only print flagged.
PRINT_ALL_PASS1 = os.environ.get("PRINT_ALL_PASS1", "1").strip() not in {"0", "false", "False"}

# mate score used to convert mate to cp-ish number for reporting
MATE_CP = int(os.environ.get("MATE_CP", "100000"))


# -------------------------
# Implementation
# -------------------------


def _http_json(method: str, url: str, *, token: str | None) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, method=method.upper(), headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        body = resp.read()
        code = int(getattr(resp, "status", 200) or 200)
    if code // 100 != 2:
        raise RuntimeError(f"HTTP {code} for {url}")
    s = body.decode("utf-8", errors="replace").strip()
    if "\n" in s:
        s = s.splitlines()[0]
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise RuntimeError("Expected JSON object")
    return obj


def fetch_game(game_id: str) -> dict[str, Any]:
    q = urllib.parse.urlencode(
        {
            "moves": "true",
            "pgnInJson": "false",
            "tags": "false",
            "clocks": "false",
            "evals": "false",
            "accuracy": "false",
            "opening": "false",
            "division": "false",
            "literate": "false",
        }
    )
    url = f"{LICHESS_BASE}/game/export/{urllib.parse.quote(game_id)}?{q}"
    return _http_json("GET", url, token=LICHESS_TOKEN)


def bot_is_white(g: dict[str, Any], bot_user: str) -> bool:
    players = g.get("players") or {}
    w = (((players.get("white") or {}).get("user") or {})).get("name") or ""
    b = (((players.get("black") or {}).get("user") or {})).get("name") or ""
    u = bot_user.strip().lower()
    if u and str(w).strip().lower() == u:
        return True
    if u and str(b).strip().lower() == u:
        return False
    raise RuntimeError("Could not determine bot color from game JSON. Set BOT_USER correctly.")


def cp_pov(score, *, pov_white: bool) -> int:
    """Convert python-chess Score to centipawns from POV (mate => +/-MATE_CP)."""
    try:
        s = score.pov(True if pov_white else False)
    except Exception:
        s = score
    v = s.score(mate_score=MATE_CP)
    if v is None:
        # fallback: treat as 0 if score is missing
        return 0
    return int(v)


def _first_info(x):
    """python-chess may return dict or [dict] depending on multipv/version."""
    if isinstance(x, list):
        return x[0] if x else {}
    return x if isinstance(x, dict) else {}


@dataclass
class PlyEval:
    ply: int
    played_uci: str
    best_uci: str
    best_cp: int
    played_cp: int
    drop_cp: int
    label: str  # ok|mistake|blunder
    fen_before: str
    ply_san: str


def _rss_bytes(pid: int) -> int | None:
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except Exception:
        return None


def main() -> int:
    global STOCKFISH_PATH
    if not STOCKFISH_PATH:
        try:
            here = os.path.abspath(os.path.dirname(__file__))
            v5 = os.path.abspath(os.path.join(here, ".."))
            engine_dir = os.path.join(v5, "engine")
            for p in (engine_dir, v5):
                if p not in sys.path:
                    sys.path.insert(0, p)
            from stockfish_path import detect_stockfish_path  # type: ignore

            STOCKFISH_PATH = detect_stockfish_path()
        except Exception:
            STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "").strip() or "stockfish"

    game_id = GAME_ID
    # Optional CLI override:
    #   python tools/two_pass_local_sf_test.py <gameId-or-url>
    # Accept:
    #   - 8-char game id (export endpoint)
    #   - "analysis" url id like 7Bq5YXRjCPOJ (take first 8 chars)
    #   - full https://lichess.org/... URLs (extract first 8-char id segment)
    if len(sys.argv) >= 2:
        raw = " ".join(sys.argv[1:]).strip()
        s = raw.strip()
        if "lichess.org" in s:
            try:
                # Try to pull the first 8-char token from URL path segments.
                import re

                m = re.search(r"(?i)\b([a-z0-9]{8})\b", s)
                if m:
                    s = m.group(1)
            except Exception:
                pass
        # If user pasted an analysis id (12 chars), take the game id (first 8 chars).
        if len(s) >= 8:
            s = s[:8]
        game_id = s

    if not game_id or len(game_id) != 8:
        print("Set GAME_ID (8-char Lichess game ID).", file=sys.stderr)
        return 2
    if not BOT_USER:
        print("Set BOT_USER.", file=sys.stderr)
        return 2

    try:
        import chess  # type: ignore
        import chess.engine  # type: ignore
    except Exception as e:
        print("Missing python-chess. Install: py -m pip install python-chess", file=sys.stderr)
        print(e, file=sys.stderr)
        return 2

    g = fetch_game(game_id)
    # Lichess JSON `moves` are SAN tokens (e.g. "e4", "Be2"), not UCI ("e2e4").
    moves_san = str(g.get("moves") or "").split()
    if not moves_san:
        print("No moves in game export.", file=sys.stderr)
        return 2
    is_white = bot_is_white(g, BOT_USER)

    print(f"game_id={game_id} bot={BOT_USER} bot_color={'white' if is_white else 'black'} moves={len(moves_san)}")
    print(f"stockfish={STOCKFISH_PATH} pass1_depth={PASS1_DEPTH} pass2_depth={PASS2_DEPTH}")
    print(f"thresholds: mistake_drop_cp>={MISTAKE_DROP_CP} blunder_drop_cp>={BLUNDER_DROP_CP}")

    t0 = time.perf_counter()
    py_rss0 = _rss_bytes(os.getpid())

    # Start one Stockfish instance reused across all analyses (low RAM).
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf_pid = getattr(engine, "pid", None)
    sf_rss0 = _rss_bytes(int(sf_pid)) if isinstance(sf_pid, int) else None

    # Keep SF bounded if possible.
    try:
        engine.configure({"Threads": 1, "Hash": 128})
    except Exception:
        pass

    flagged: list[PlyEval] = []
    try:
        board = chess.Board()
        pass1: list[PlyEval] = []

        def board_before_ply(ply: int) -> chess.Board:
            """
            Rebuild a board with full move history up to (ply-1).
            This preserves threefold-repetition context, unlike set_fen().
            """
            b = chess.Board()
            for i in range(ply - 1):
                b.push(b.parse_san(moves_san[i]))
            return b

        for ply, played_san in enumerate(moves_san, start=1):
            side_is_white = (ply % 2) == 1
            if side_is_white != is_white:
                # opponent ply
                try:
                    mv_opp = board.parse_san(played_san)
                    board.push(mv_opp)
                except Exception:
                    print(f"Failed to parse opponent SAN ply={ply} san={played_san!r}", file=sys.stderr)
                    break
                continue

            fen_before = board.fen()

            # Evaluate best at pre-move position
            info_best = _first_info(
                engine.analyse(
                    board,
                    chess.engine.Limit(depth=PASS1_DEPTH),
                    multipv=1,
                    info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
                )
            )
            best_move = (info_best.get("pv") or [None])[0]
            if best_move is None:
                # Fallback: ask engine for a best move directly.
                try:
                    r = engine.play(board, chess.engine.Limit(depth=PASS1_DEPTH))
                    best_move = r.move
                except Exception:
                    best_move = None
            if best_move is None:
                # If SF gives no move, skip this ply.
                try:
                    mv_fallback = board.parse_san(played_san)
                    board.push(mv_fallback)
                except Exception:
                    print(f"Failed to parse bot SAN ply={ply} san={played_san!r}", file=sys.stderr)
                    break
                continue
            best_uci = best_move.uci()
            sc1 = info_best.get("score")
            if sc1 is None:
                # If score is absent, skip.
                continue
            best_cp = cp_pov(sc1, pov_white=is_white)

            # Evaluate after played move (position after your move)
            try:
                mv_bot = board.parse_san(played_san)
                played_uci = mv_bot.uci()
                board.push(mv_bot)
            except Exception:
                print(f"Failed to parse bot SAN ply={ply} san={played_san!r}", file=sys.stderr)
                break
            info_played = _first_info(
                engine.analyse(
                    board,
                    chess.engine.Limit(depth=PASS1_DEPTH),
                    multipv=1,
                    info=chess.engine.INFO_SCORE,
                )
            )
            sc2 = info_played.get("score")
            if sc2 is None:
                continue
            played_cp = cp_pov(sc2, pov_white=is_white)

            drop = best_cp - played_cp
            if drop >= BLUNDER_DROP_CP:
                label = "blunder"
            elif drop >= MISTAKE_DROP_CP:
                label = "mistake"
            else:
                label = "ok"

            pe = PlyEval(
                ply=ply,
                played_uci=played_uci,
                best_uci=best_uci,
                best_cp=best_cp,
                played_cp=played_cp,
                drop_cp=drop,
                label=label,
                fen_before=fen_before,
                ply_san=played_san,
            )
            pass1.append(pe)
            if label in {"mistake", "blunder"}:
                flagged.append(pe)

            if PRINT_ALL_PASS1 or label != "ok":
                print(
                    f"pass1 ply={ply:2d} played={played_uci:5s} best={best_uci:5s} "
                    f"best_cp={best_cp:+6d} played_cp={played_cp:+6d} drop={drop:+5d} {label}"
                )

        # Summary pass1
        n_ok = sum(1 for x in pass1 if x.label == "ok")
        n_m = sum(1 for x in pass1 if x.label == "mistake")
        n_b = sum(1 for x in pass1 if x.label == "blunder")
        print(f"pass1 summary: ok={n_ok} mistake={n_m} blunder={n_b} (bot plies analyzed={len(pass1)})")

        # Pass 2: re-evaluate flagged positions with deeper depth
        if not flagged:
            print("pass2: no flagged plies to re-analyze")
        else:
            print("pass2: re-analyzing flagged plies at deeper depth")
            exp_suggestions: list[tuple[str, str, int, int, int]] = []
            # Rebuild per flagged ply from SAN to preserve repetition history.
            for x in flagged:
                try:
                    board2 = board_before_ply(x.ply)
                except Exception as e:
                    print(f"pass2: failed to rebuild history for ply={x.ply}: {e}", file=sys.stderr)
                    continue
                info_best2 = _first_info(
                    engine.analyse(
                        board2,
                        chess.engine.Limit(depth=PASS2_DEPTH),
                        multipv=1,
                        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
                    )
                )
                best_move2 = (info_best2.get("pv") or [None])[0]
                if best_move2 is None:
                    try:
                        r2 = engine.play(board2, chess.engine.Limit(depth=PASS2_DEPTH))
                        best_move2 = r2.move
                    except Exception:
                        best_move2 = None
                best_uci2 = best_move2.uci() if best_move2 is not None else x.best_uci
                scb = info_best2.get("score")
                if scb is None:
                    continue
                best_cp2 = cp_pov(scb, pov_white=is_white)

                # played
                try:
                    # Prefer SAN for consistency with rebuilt board.
                    board2.push(board2.parse_san(x.ply_san))
                except Exception:
                    continue
                info_played2 = _first_info(
                    engine.analyse(
                        board2,
                        chess.engine.Limit(depth=PASS2_DEPTH),
                        multipv=1,
                        info=chess.engine.INFO_SCORE,
                    )
                )
                scp = info_played2.get("score")
                if scp is None:
                    continue
                played_cp2 = cp_pov(scp, pov_white=is_white)
                drop2 = best_cp2 - played_cp2
                label2 = "blunder" if drop2 >= BLUNDER_DROP_CP else ("mistake" if drop2 >= MISTAKE_DROP_CP else "ok")

                print(
                    f"pass2 ply={x.ply:2d} played={x.played_uci:5s} best={best_uci2:5s} "
                    f"best_cp={best_cp2:+6d} played_cp={played_cp2:+6d} drop={drop2:+5d} {label2}"
                )

                if label2 in {"mistake", "blunder"}:
                    # Suggest: add (fen_before, best_uci2) to experience book
                    exp_suggestions.append((x.fen_before, best_uci2, best_cp2, played_cp2, drop2))

            if exp_suggestions:
                print("\nexperience_book suggestions (fen_before -> best_uci):")
                for fen, best_uci, best_cp2, played_cp2, drop2 in exp_suggestions:
                    print(f"- best={best_uci} drop={drop2:+d} best_cp={best_cp2:+d} played_cp={played_cp2:+d}")
                    print(f"  fen: {fen}")
            else:
                print("No pass2 positions still exceeding thresholds; no experience suggestions.")

    finally:
        try:
            engine.quit()
        except Exception:
            pass

    t1 = time.perf_counter()
    py_rss1 = _rss_bytes(os.getpid())
    sf_rss1 = _rss_bytes(int(sf_pid)) if isinstance(sf_pid, int) else None

    print("\nresource summary:")
    print(f"- elapsed_s: {t1 - t0:.2f}")
    if py_rss0 is not None and py_rss1 is not None:
        print(f"- python_rss_mb: {py_rss0/1e6:.1f} -> {py_rss1/1e6:.1f}")
    else:
        print("- python_rss_mb: (install psutil for RSS reporting)")
    if sf_rss0 is not None and sf_rss1 is not None:
        print(f"- stockfish_rss_mb: {sf_rss0/1e6:.1f} -> {sf_rss1/1e6:.1f}")
    else:
        print("- stockfish_rss_mb: (install psutil for RSS reporting)")

    print("\nCompare with Lichess judgments:")
    print(f"- Lichess JSON has analysis? {('analysis' in g)} (this script does not use it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

