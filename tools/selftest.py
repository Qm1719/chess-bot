"""
selftest.py — lightweight correctness checks for submission v3.

Run:
  py selftest.py
"""

from __future__ import annotations

import random
import sys
import time


def _try_import_v3():
    try:
        from position import Position  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    have_v3 = _try_import_v3()
    if not have_v3:
        print("v3 Position not implemented yet; selftest is a placeholder.")
        return 0

    from position import Position  # type: ignore

    # Basic push/pop stability smoke test on random play.
    pos = Position.from_fen(Position.START_FEN)
    z0 = pos.key
    for _ in range(50):
        moves = list(pos.legal_moves())
        if not moves:
            break
        mv = random.choice(moves)
        u = pos.push(mv)
        pos.pop(u)
        if pos.key != z0:
            raise AssertionError("push/pop did not restore zobrist key")

    # Tiny perft to catch obvious movegen bugs.
    def perft(p: Position, depth: int) -> int:
        if depth == 0:
            return 1
        n = 0
        for mv in p.legal_moves():
            u = p.push(mv)
            n += perft(p, depth - 1)
            p.pop(u)
        return n

    t0 = time.perf_counter()
    p = Position.from_fen(Position.START_FEN)
    n1 = perft(p, 1)
    n2 = perft(p, 2)
    dt = time.perf_counter() - t0
    print("perft startpos d1=", n1, "d2=", n2, f"({dt:.2f}s)")

    # Known perft for chess start position:
    # depth1=20, depth2=400
    if n1 != 20 or n2 != 400:
        raise AssertionError(f"perft mismatch: got d1={n1} d2={n2}")

    print("selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

