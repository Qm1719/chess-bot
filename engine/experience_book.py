"""
experience_book.py

Tiny persistent "experience book" (position key -> forced move) for your engine.

- Storage: SQLite (single small binary file, fast lookups, crash-safe)
- Keying: uses submission v3 `Position.key` computed from FEN

Usage (examples):
  py experience_book.py add --fen "<fen>" --move d1d4
  py experience_book.py get --fen "<fen>"
  py experience_book.py list --limit 20

If you run with no args, it prompts interactively.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time


ROOT = os.path.abspath(os.path.dirname(__file__))
V5_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
DB_PATH = os.path.join(V5_ROOT, "data", "experience_book.sqlite3")


def _add_engine_paths() -> None:
    engine = os.path.join(V5_ROOT, "engine")
    data = os.path.join(V5_ROOT, "data")
    for p in (engine, data, V5_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    # We store zobrist keys as 16-hex TEXT (unsigned 64-bit safe).
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS book (
          key TEXT PRIMARY KEY,
          uci TEXT NOT NULL,
          depth INTEGER,
          score INTEGER,
          ts INTEGER NOT NULL
        )
        """
    )
    # If an old schema exists (key INTEGER PRIMARY KEY), migrate it.
    try:
        cols = con.execute("PRAGMA table_info(book)").fetchall()
        # cols rows: (cid, name, type, notnull, dflt_value, pk)
        key_type = None
        for c in cols:
            if str(c[1]).lower() == "key":
                key_type = str(c[2]).upper()
                break
        if key_type is not None and key_type != "TEXT":
            con.execute("ALTER TABLE book RENAME TO book_old")
            con.execute(
                """
                CREATE TABLE book (
                  key TEXT PRIMARY KEY,
                  uci TEXT NOT NULL,
                  depth INTEGER,
                  score INTEGER,
                  ts INTEGER NOT NULL
                )
                """
            )
            # Best-effort copy: old keys may be signed; convert to 16-hex.
            rows = con.execute("SELECT key, uci, depth, score, ts FROM book_old").fetchall()
            for k, uci, depth, score, ts in rows:
                try:
                    kk = int(k)
                    kk &= (1 << 64) - 1
                    kh = f"{kk:016x}"
                    con.execute(
                        "INSERT OR REPLACE INTO book(key, uci, depth, score, ts) VALUES(?,?,?,?,?)",
                        (kh, uci, depth, score, ts),
                    )
                except Exception:
                    continue
            con.commit()
    except Exception:
        pass
    return con


def _keyhex_from_fen(fen: str) -> str:
    _add_engine_paths()
    from position import Position  # type: ignore

    pos = Position.from_fen(fen)
    k = int(pos.key) & ((1 << 64) - 1)
    return f"{k:016x}"


def add_entry(con: sqlite3.Connection, fen: str, uci: str, depth: int | None, score: int | None) -> str:
    key = _keyhex_from_fen(fen)
    ts = int(time.time())
    con.execute(
        "INSERT OR REPLACE INTO book(key, uci, depth, score, ts) VALUES(?,?,?,?,?)",
        (key, uci.strip(), depth, score, ts),
    )
    con.commit()
    return key


def get_entry(con: sqlite3.Connection, fen: str) -> tuple[str, str, int | None, int | None, int] | None:
    key = _keyhex_from_fen(fen)
    row = con.execute("SELECT key, uci, depth, score, ts FROM book WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    return str(row[0]), str(row[1]), row[2], row[3], int(row[4])


def list_entries(con: sqlite3.Connection, limit: int) -> list[tuple[str, str, int | None, int | None, int]]:
    rows = con.execute(
        "SELECT key, uci, depth, score, ts FROM book ORDER BY ts DESC LIMIT ?",
        (int(limit),),
    ).fetchall()
    out = []
    for r in rows:
        out.append((str(r[0]), str(r[1]), r[2], r[3], int(r[4])))
    return out


def _prompt(prompt: str) -> str:
    print(prompt, end="", flush=True)
    return sys.stdin.readline().strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    ap_add = sub.add_parser("add", help="Add/replace an entry")
    ap_add.add_argument("--fen", required=True)
    ap_add.add_argument("--move", required=True, help="UCI move, e.g. d1d4")
    ap_add.add_argument("--depth", type=int, default=None)
    ap_add.add_argument("--score", type=int, default=None, help="cp from side-to-move (optional)")

    ap_get = sub.add_parser("get", help="Lookup by FEN")
    ap_get.add_argument("--fen", required=True)

    ap_list = sub.add_parser("list", help="List newest entries")
    ap_list.add_argument("--limit", type=int, default=20)

    args = ap.parse_args()

    con = _connect(DB_PATH)
    try:
        if args.cmd is None:
            fen = _prompt("FEN: ")
            uci = _prompt("Move (UCI): ")
            key = add_entry(con, fen, uci, depth=None, score=None)
            print(f"added key={key} move={uci} db={DB_PATH}")
            return

        if args.cmd == "add":
            key = add_entry(con, args.fen, args.move, args.depth, args.score)
            print(f"added key={key} move={args.move} db={DB_PATH}")
            return

        if args.cmd == "get":
            r = get_entry(con, args.fen)
            if r is None:
                print("not_found")
                return
            key, uci, depth, score, ts = r
            print(f"key={key} move={uci} depth={depth} score={score} ts={ts}")
            return

        if args.cmd == "list":
            for key, uci, depth, score, ts in list_entries(con, args.limit):
                print(f"key={key} move={uci} depth={depth} score={score} ts={ts}")
            return
    finally:
        con.close()


if __name__ == "__main__":
    main()

