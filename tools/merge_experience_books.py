from __future__ import annotations

import os
import sqlite3
import sys


def _db_path(name: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    v5 = os.path.abspath(os.path.join(here, ".."))
    return os.path.join(v5, "data", name)


def _count(con: sqlite3.Connection) -> int:
    row = con.execute("SELECT COUNT(*) FROM book").fetchone()
    return int(row[0] if row else 0)


def main(argv: list[str]) -> int:
    """
    Merge two experience books:
      - dst: data/experience_book.sqlite3 (updated in-place)
      - src: data/experience_book - Copy.sqlite3 (read-only)

    Policy: on key conflicts, keep whichever row has the larger `ts` (newer wins).
    """
    dst = _db_path("temp/experience_book.sqlite3")
    src = _db_path("temp/experience_book - Copy.sqlite3")

    if len(argv) >= 2 and argv[1].strip():
        src = os.path.abspath(argv[1].strip())
    if len(argv) >= 3 and argv[2].strip():
        dst = os.path.abspath(argv[2].strip())

    if not os.path.exists(src):
        print(f"missing src db: {src}")
        return 2
    if not os.path.exists(dst):
        print(f"missing dst db: {dst}")
        return 2

    con = sqlite3.connect(dst)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")

        before = _count(con)
        con.execute("ATTACH ? AS src", (src,))

        # Newer-wins merge on primary key `key`.
        #
        # Use a portable two-step merge so it works on older SQLite builds that
        # don't support "INSERT ... ON CONFLICT DO UPDATE".
        #
        # 1) Update existing keys where src.ts is newer.
        con.execute(
            """
            UPDATE book
               SET uci   = (SELECT s.uci   FROM src.book AS s WHERE s.key = book.key),
                   depth = (SELECT s.depth FROM src.book AS s WHERE s.key = book.key),
                   score = (SELECT s.score FROM src.book AS s WHERE s.key = book.key),
                   ts    = (SELECT s.ts    FROM src.book AS s WHERE s.key = book.key)
             WHERE EXISTS (SELECT 1 FROM src.book AS s WHERE s.key = book.key AND s.ts > book.ts)
            """
        )
        # 2) Insert missing keys.
        con.execute(
            """
            INSERT OR IGNORE INTO book(key, uci, depth, score, ts)
            SELECT key, uci, depth, score, ts FROM src.book
            """
        )
        con.commit()

        con.execute("DETACH src")
        after = _count(con)
    finally:
        con.close()

    print("merged experience books")
    print("src:", src)
    print("dst:", dst)
    print(f"rows: {before} -> {after}  (delta={after - before})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

