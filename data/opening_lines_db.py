"""
opening_lines_db.py — extra v4 opening lines in SQLite (no engine restart to pick up new rows).

Table stores one full line as space-separated UCI. Merged on top of the built-in _LINES in
opening_book_v4._get_merged_opening_book() when the DB file mtime changes.
"""

from __future__ import annotations

import os
import sqlite3
import time

_DATA_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(_DATA_DIR, "opening_lines.sqlite3")

# Migrated from former _LINES_FROM_OLD_GAMES in opening_book_v4.py (one-time seed if DB empty)
_SEED_LINES: list[tuple[str, str | None]] = [
    (
        "d2d4 g8f6 c2c4 d7d6 b1c3 c8g4 h2h3 g4h5 d1b3 b8c6",
        "A53 Old Indian Defense (seed)",
    ),
]


def db_path() -> str:
    return DB_PATH


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def init_db() -> None:
    con = _connect()
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS book_lines (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              uci_space TEXT NOT NULL,
              note TEXT,
              created INTEGER NOT NULL,
              UNIQUE(uci_space)
            )
            """
        )
        n = con.execute("SELECT COUNT(*) FROM book_lines").fetchone()
        if n and int(n[0]) == 0:
            now = int(time.time())
            for uci_s, note in _SEED_LINES:
                con.execute(
                    "INSERT OR IGNORE INTO book_lines(uci_space, note, created) VALUES(?,?,?)",
                    (uci_s, note, now),
                )
        con.commit()
    finally:
        con.close()


def ensure_db() -> None:
    init_db()


def load_all_lines() -> list[list[str]]:
    """All stored lines as UCI token lists. Creates DB and seed on first use."""
    ensure_db()
    con = _connect()
    try:
        rows = con.execute("SELECT uci_space FROM book_lines ORDER BY id").fetchall()
    finally:
        con.close()
    out: list[list[str]] = []
    for (s,) in rows:
        t = (s or "").strip()
        if not t:
            continue
        uci = t.split()
        if uci:
            out.append(uci)
    return out


def add_line(uci: list[str], note: str | None = None) -> str:
    """
    Insert one full line. Returns 'inserted', 'duplicate', or 'error: ...'.
    """
    if not uci:
        return "error: empty uci"
    ensure_db()
    s = " ".join(uci)
    now = int(time.time())
    con = _connect()
    try:
        cur = con.execute(
            "INSERT OR IGNORE INTO book_lines(uci_space, note, created) VALUES(?,?,?)",
            (s, (note or "").strip() or None, now),
        )
        con.commit()
        if cur.rowcount and cur.rowcount > 0:
            return "inserted"
        return "duplicate"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"
    finally:
        con.close()


def list_lines(limit: int = 50) -> list[tuple[int, str, str | None]]:
    ensure_db()
    con = _connect()
    try:
        rows = con.execute(
            "SELECT id, uci_space, note FROM book_lines ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    finally:
        con.close()
    return [(int(r[0]), str(r[1]), r[2]) for r in rows]
