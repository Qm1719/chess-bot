"""
tt.py — Transposition table keyed by 64-bit zobrist (Position.key).
"""

from __future__ import annotations

from dataclasses import dataclass
import threading


def _dataclass_slots_kw_supported() -> bool:
    try:
        # Python < 3.10 doesn't accept `slots=`.
        @dataclass(slots=True)
        class _T:
            x: int

        return True
    except TypeError:
        return False


TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


@dataclass(**({"slots": True} if _dataclass_slots_kw_supported() else {}))
class TTEntry:
    depth: int
    score: int
    flag: int
    best: object | None


class TranspositionTable:
    def __init__(self, max_size: int = 1_000_000):
        self._max = max_size
        self._d: dict[int, TTEntry] = {}
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def probe(self, key: int, depth: int, alpha: int, beta: int):
        with self._lock:
            e = self._d.get(key)
            if e is None:
                return False, 0, None
            if e.depth >= depth:
                if e.flag == TT_EXACT:
                    return True, e.score, e.best
                if e.flag == TT_LOWER and e.score >= beta:
                    return True, e.score, e.best
                if e.flag == TT_UPPER and e.score <= alpha:
                    return True, e.score, e.best
            return False, 0, e.best

    def store(self, key: int, depth: int, score: int, flag: int, best):
        with self._lock:
            e = self._d.get(key)
            if e is None:
                if len(self._d) >= self._max:
                    return
                self._d[key] = TTEntry(depth=depth, score=score, flag=flag, best=best)
            else:
                if depth >= e.depth:
                    self._d[key] = TTEntry(depth=depth, score=score, flag=flag, best=best)

