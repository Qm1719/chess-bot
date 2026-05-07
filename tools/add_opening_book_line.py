"""
add_opening_book_line.py

Given a Lichess game ID, use Stockfish (depth 18 by default) to check Black's early moves
(plies 2,4,6,8). If Black played something different from Stockfish's best move, construct
an opening book line (UCI tokens) and insert it into:

  submission v5/data/opening_lines.sqlite3

Construction:
- Start from the game's move list.
- At the first detected "Black != SF best" ply among the checked plies, replace Black's move
  with SF best move.
- By default, stop the line immediately after that substituted Black move. This avoids
  accidentally inserting White move choices from the game into your curated White book.
- Optional: if `--extend` is provided, try to extend the line with the game's subsequent moves
  as long as they remain legal under the substituted line, up to MAX_PLY (default 10).

This avoids inserting illegal continuations while still allowing extension when it fits.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.error
import urllib.request


def _add_paths() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    v5 = os.path.abspath(os.path.join(here, ".."))
    engine = os.path.join(v5, "engine")
    data = os.path.join(v5, "data")
    for p in (engine, data, v5):
        if p not in sys.path:
            sys.path.insert(0, p)
    return v5


V5_ROOT = _add_paths()


def _require_python_chess():
    try:
        import chess  # type: ignore
        import chess.pgn  # type: ignore
        import chess.engine  # type: ignore
    except Exception as e:
        raise SystemExit(
            "This script requires python-chess.\n"
            "Install:  py -m pip install python-chess\n"
            f"Error: {e}"
        )
    return chess


def _detect_stockfish_path() -> str:
    p = os.environ.get("STOCKFISH_PATH", "").strip()
    if p:
        return p
    try:
        from stockfish_path import detect_stockfish_path  # type: ignore

        return detect_stockfish_path()
    except Exception:
        return "stockfish"


def _fetch_pgn(game_id: str) -> str:
    url = f"https://lichess.org/game/export/{game_id}"
    req = urllib.request.Request(url, headers={"Accept": "application/x-chess-pgn"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise SystemExit(f"Failed to download PGN from Lichess ({url}). ({e})")


def _pgn_to_uci_moves(pgn_text: str) -> list[str]:
    chess = _require_python_chess()
    g = chess.pgn.read_game(io.StringIO(pgn_text))
    if g is None:
        raise SystemExit("Failed to parse PGN")
    return [m.uci() for m in g.mainline_moves()]


def _uci_list_to_line_with_substitution(
    uci_moves: list[str],
    *,
    substitute_ply: int,
    substitute_uci: str,
    max_ply: int,
    extend: bool,
) -> list[str]:
    """
    substitute_ply: 1-based ply index in the game.
    """
    chess = _require_python_chess()
    b = chess.Board()
    out: list[str] = []
    # First, replay up to ply-1 as-is.
    for u in uci_moves[: max(0, substitute_ply - 1)]:
        mv = chess.Move.from_uci(u)
        if mv not in b.legal_moves:
            break
        b.push(mv)
        out.append(u)
        if len(out) >= max_ply:
            return out

    # Apply substituted move.
    mv_sub = chess.Move.from_uci(substitute_uci)
    if mv_sub in b.legal_moves:
        b.push(mv_sub)
        out.append(substitute_uci)
    else:
        # If substitution itself is illegal, just stop at the prefix.
        return out

    # Default behavior: stop immediately after the substituted Black move.
    if not extend:
        return out

    if len(out) >= max_ply:
        return out

    # Try to extend with the remaining original game moves, as long as they stay legal.
    for u in uci_moves[substitute_ply:]:
        mv = chess.Move.from_uci(u)
        if mv not in b.legal_moves:
            break
        b.push(mv)
        out.append(u)
        if len(out) >= max_ply:
            break
    return out


def main(argv: list[str]) -> int:
    chess = _require_python_chess()
    import opening_lines_db  # type: ignore

    ap = argparse.ArgumentParser()
    ap.add_argument("game_id", help="Lichess 8-char game id")
    ap.add_argument("--max-ply", type=int, default=10)
    ap.add_argument("--sf-depth", type=int, default=18)
    ap.add_argument("--extend", action="store_true", help="Extend line past substituted Black move (default: off).")
    args = ap.parse_args(argv[1:])

    gid = args.game_id.strip()
    if not gid or len(gid) != 8:
        print("game_id must be an 8-character Lichess id", file=sys.stderr)
        return 2

    max_ply = max(1, int(args.max_ply))
    depth = max(1, int(args.sf_depth))

    pgn = _fetch_pgn(gid)
    uci_moves = _pgn_to_uci_moves(pgn)
    if not uci_moves:
        print("no moves in PGN", file=sys.stderr)
        return 2

    sf_path = _detect_stockfish_path()
    sf = chess.engine.SimpleEngine.popen_uci([sf_path])
    try:
        try:
            sf.configure({"Threads": 1, "Hash": 128})
        except Exception:
            pass

        # Check Black plies: 2,4,6,8 (i.e. black moves 1..4)
        check_plies = [2, 4, 6, 8]
        chosen_sub: tuple[int, str, str] | None = None  # (ply, played, best)

        b = chess.Board()
        for ply, u in enumerate(uci_moves, start=1):
            mv = chess.Move.from_uci(u)
            if mv not in b.legal_moves:
                break

            if ply in check_plies and b.turn == chess.BLACK:
                info = sf.analyse(b, chess.engine.Limit(depth=depth))
                pv = info.get("pv", []) or []
                best = pv[0].uci() if pv else "0000"
                if best != "0000" and best != u:
                    chosen_sub = (ply, u, best)
                    break

            b.push(mv)

        if chosen_sub is None:
            print(f"{gid}: no black mismatch found in plies {check_plies} at depth={depth}")
            return 0

        ply, played, best = chosen_sub
        line = _uci_list_to_line_with_substitution(
            uci_moves,
            substitute_ply=ply,
            substitute_uci=best,
            max_ply=max_ply,
            extend=bool(args.extend),
        )
        if not line:
            print(f"{gid}: could not construct a line (sub ply={ply} best={best})", file=sys.stderr)
            return 1

        note = f"lichess:{gid} black_ply={ply} played={played} sf_best={best} depth={depth}"
        r = opening_lines_db.add_line(line, note=note)
        print(r)
        print("uci", " ".join(line))
        if r == "inserted":
            print(f"db: {opening_lines_db.db_path()}")
        return 0 if r in {"inserted", "duplicate"} else 1
    finally:
        try:
            sf.quit()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

