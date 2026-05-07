from __future__ import annotations

import os
import sys

_V5_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENGINE = os.path.join(_V5_ROOT, "engine")
_DATA = os.path.join(_V5_ROOT, "data")
for _p in (_V5_ROOT, _ENGINE, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from position import Position
from agents_v4 import get_agent


def main() -> None:
    a = get_agent(10)
    p = Position.from_fen(Position.START_FEN)
    line: list[str] = []
    for _ in range(20):
        mv = a.choose_move(p, time_limit_s=0.05)
        if mv is None:
            break
        # sanity: chosen move should be legal and from_sq occupied
        if p.board[mv.from_sq] < 0:
            print("ERROR empty from_sq", mv.uci(), "fen", p.to_fen())
            return
        if all(m.uci() != mv.uci() for m in p.legal_moves()):
            print("ERROR illegal move", mv.uci(), "fen", p.to_fen())
            return
        line.append(mv.uci())
        p.push(mv)
    print("moves", len(line))
    print(" ".join(line))


if __name__ == "__main__":
    main()

