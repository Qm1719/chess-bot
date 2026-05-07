"""
add_opening_line_from_pgn.py

Append an opening line from a PGN move string into:
  submission v5/data/opening_lines.sqlite3

The running engine picks up new rows when the DB file is updated (no restart).

It converts SAN -> UCI using python-chess (required for this script).
It also captures any `{ ... }` PGN comments and stores them in the `note` column.

Examples:
  py add_opening_line_from_pgn.py --pgn \"1. d4 Nf6 2. c4 d6 { A53 Old Indian } 3. Nc3 Bg4 4. h3 Bh5 5. Qb3 Nc6\"

Interactive:
  py add_opening_line_from_pgn.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Tuple

ROOT = os.path.abspath(os.path.dirname(__file__))
V5_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
_DATA = os.path.join(V5_ROOT, "data")
if _DATA not in sys.path:
    sys.path.insert(0, _DATA)


def _require_python_chess():
    try:
        import chess  # type: ignore
    except Exception as e:
        raise SystemExit(
            "This script requires python-chess.\n"
            "Install:  py -m pip install python-chess\n"
            f"Error: {e}"
        )
    return chess


def _extract_tokens_and_comments(pgn_moves: str) -> Tuple[List[str], List[str]]:
    """
    Returns (san_tokens, comments).
    Strips move numbers, results, and NAGs. Keeps SAN moves.
    """
    s = pgn_moves.strip()
    if not s:
        return [], []

    # Collect {...} comments (may include spaces), remove them from the move stream.
    comments: List[str] = []

    def _comment_repl(m: re.Match) -> str:
        txt = m.group(1).strip()
        if txt:
            comments.append(txt)
        return " "

    s = re.sub(r"\{([^}]*)\}", _comment_repl, s)

    # Remove line breaks and normalize spaces.
    s = re.sub(r"\s+", " ", s).strip()

    raw = s.split(" ")
    out: List[str] = []
    for tok in raw:
        t = tok.strip()
        if not t:
            continue
        # Remove move numbers like "1." or "23..." or "7.."
        if re.fullmatch(r"\d+\.+", t):
            continue
        if re.fullmatch(r"\d+\.\.\.", t):
            continue
        # Results
        if t in {"1-0", "0-1", "1/2-1/2", "*"}:
            continue
        # NAGs
        if t.startswith("$") and t[1:].isdigit():
            continue
        out.append(t)

    return out, comments


def _san_to_uci_line(san_tokens: List[str]) -> List[str]:
    chess = _require_python_chess()
    board = chess.Board()
    uci: List[str] = []
    for san in san_tokens:
        try:
            mv = board.parse_san(san)
        except Exception as e:
            raise SystemExit(f"Failed to parse SAN token {san!r}: {e}")
        uci.append(mv.uci())
        board.push(mv)
    return uci


def main() -> int:
    import opening_lines_db  # type: ignore

    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default=None, help="PGN move text (moves only)")
    ap.add_argument("--list", action="store_true", help="List recent lines in the DB and exit")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    if args.list:
        for i, uci_s, note in opening_lines_db.list_lines(limit=int(args.limit)):
            print(f"{i}\t{uci_s}\t{note or ''}")
        return 0

    if args.pgn is None:
        print("Paste PGN move line, then press Enter:")
        pgn = sys.stdin.readline().strip()
    else:
        pgn = args.pgn

    san, comments = _extract_tokens_and_comments(pgn)
    if not san:
        print("No SAN moves found.", file=sys.stderr)
        return 2

    uci = _san_to_uci_line(san)
    note = "; ".join(comments) if comments else None
    r = opening_lines_db.add_line(uci, note)
    print(r)
    print("uci", " ".join(uci))
    if r == "inserted":
        print(f"db: {opening_lines_db.db_path()}")
    return 0 if not str(r).startswith("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
