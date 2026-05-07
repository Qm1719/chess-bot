from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import chess


@dataclass(frozen=True)
class AgentSpec:
    id: int
    name: str
    make: Callable[[], "Agent"]


class Agent:
    """
    Stateless interface: a fresh board is provided each move.
    Implementations may keep internal caches/TT.
    """

    name: str

    def choose_uci(self, board: chess.Board, *, think_s: float, game_id: str) -> str:
        raise NotImplementedError

    def close(self) -> None:  # optional
        return


class _Env:
    """Temporary env override (process-global)."""

    def __init__(self, kv: dict[str, str | None]):
        self._kv = kv
        self._old: dict[str, str | None] = {}

    def __enter__(self):
        for k, v in self._kv.items():
            self._old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)

    def __exit__(self, exc_type, exc, tb):
        for k, old in self._old.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        return False


def _import_v4():
    # Local import so testbot can run without engine deps in some contexts.
    here = os.path.abspath(os.path.dirname(__file__))
    v5_root = os.path.abspath(os.path.join(here, ".."))
    engine = os.path.join(v5_root, "engine")
    data = os.path.join(v5_root, "data")
    for p in (engine, data, v5_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    from position import Position  # type: ignore
    import agents_v4  # type: ignore

    return Position, agents_v4


def _v4_pick_move_by_root_scores(
    pos,
    agents_v4_mod,
    *,
    depth: int,
    pick: str,
    rng: random.Random,
    repetition_base: dict[int, int] | None,
    deadline: float | None,
) -> Optional[str]:
    """
    pick:
      - "best"
      - "second"
      - "random"
    """
    from tt import TranspositionTable  # type: ignore

    moves = list(pos.legal_moves())
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0].uci()

    # Score each root move at fixed depth.
    tt = TranspositionTable()
    scored: list[tuple[int, str]] = []
    timeout_exc = getattr(agents_v4_mod, "_SearchTimeout", Exception)
    for mv in moves:
        nodes_ref = [0]
        killers = [[None, None] for _ in range(agents_v4_mod._MAX_PLY)]
        history: dict = {}
        rep = dict(repetition_base) if repetition_base else {pos.key: 1}
        u = pos.push(mv)
        rep[pos.key] = rep.get(pos.key, 0) + 1
        try:
            try:
                sc = -agents_v4_mod._alpha_beta(
                    pos,
                    max(0, int(depth) - 1),
                    -10**9,
                    10**9,
                    nodes_ref,
                    tt,
                    killers,
                    history,
                    rep,
                    1,
                    deadline,
                    False,
                )
            except timeout_exc:
                # Time budget exceeded: keep what we have so far and fall back later.
                break
        finally:
            rep[pos.key] -= 1
            if rep[pos.key] <= 0:
                rep.pop(pos.key, None)
            pos.pop(u)
        scored.append((int(sc), mv.uci()))

    if not scored:
        return moves[0].uci()
    scored.sort(reverse=True, key=lambda x: x[0])
    if pick == "best":
        return scored[0][1]
    if pick == "second":
        return scored[1][1] if len(scored) > 1 else scored[0][1]
    return rng.choice([u for _, u in scored])


class AgentsV4Agent(Agent):
    def __init__(
        self,
        *,
        name: str,
        level: int = 10,
        opening_book: bool = True,
        syzygy: bool = True,
        experience_book: bool = True,
        sf_nnue: bool = True,
        nnue: bool = True,
        fixed_depth: int | None = None,
        epsilon_pick_depth1: float = 0.0,
        second_best_prob: float = 0.0,
        always_second_best: bool = False,
        seed: int = 0,
    ):
        self.name = name
        self._level = int(level)
        self._opening_book = bool(opening_book)
        self._syzygy = bool(syzygy)
        self._experience_book = bool(experience_book)
        self._sf_nnue = bool(sf_nnue)
        self._nnue = bool(nnue)
        self._fixed_depth = fixed_depth
        self._epsilon_pick_depth1 = float(epsilon_pick_depth1)
        self._second_best_prob = float(second_best_prob)
        self._always_second_best = bool(always_second_best)
        self._rng = random.Random(int(seed))

        Position, agents_v4_mod = _import_v4()
        self._Position = Position
        self._v4 = agents_v4_mod
        self._agent = agents_v4_mod.MinimaxAgent(self._level)

    def _env_overrides(self) -> dict[str, str | None]:
        kv: dict[str, str | None] = {}
        kv["V4_EXPERIENCE_BOOK"] = "1" if self._experience_book else "0"
        kv["SF_NNUE"] = "1" if self._sf_nnue else "0"
        kv["NNUE_OFF"] = None if self._nnue else "1"
        # Disable search spam unless explicitly requested by caller.
        if os.environ.get("V4_SEARCH_INFO") is None:
            kv["V4_SEARCH_INFO"] = "0"
        return kv

    def choose_uci(self, board: chess.Board, *, think_s: float, game_id: str) -> str:
        # Convert python-chess board to our engine Position via FEN.
        pos = self._Position.from_fen(board.fen())

        # Optional disable opening book by monkeypatching pick_book_move.
        old_pick = None
        if not self._opening_book:
            old_pick = getattr(self._v4, "pick_book_move", None)
            setattr(self._v4, "pick_book_move", lambda _pos: None)

        # Optional disable syzygy by monkeypatching probe funcs.
        old_probe_best = old_probe_bestmoves = None
        if not self._syzygy:
            old_probe_best = getattr(self._v4, "probe_bestmove", None)
            old_probe_bestmoves = getattr(self._v4, "probe_bestmoves", None)
            setattr(self._v4, "probe_bestmove", lambda _fen: None)
            setattr(self._v4, "probe_bestmoves", lambda _fen: None)

        # Repetition base: for this test harness, we use only the current node (safe default).
        repetition_base = {int(pos.key): 1}
        deadline = None
        if think_s is not None and think_s > 0:
            deadline = time.perf_counter() + float(think_s)

        try:
            with _Env(self._env_overrides()):
                # Random/second-best perturbations.
                roll = self._rng.random()
                if self._always_second_best:
                    uci = _v4_pick_move_by_root_scores(
                        pos,
                        self._v4,
                        depth=max(1, self._fixed_depth or 2),
                        pick="second",
                        rng=self._rng,
                        repetition_base=repetition_base,
                        deadline=deadline,
                    )
                    return uci or "0000"

                if self._second_best_prob > 0.0 and roll < self._second_best_prob:
                    uci = _v4_pick_move_by_root_scores(
                        pos,
                        self._v4,
                        depth=max(1, self._fixed_depth or 2),
                        pick="second",
                        rng=self._rng,
                        repetition_base=repetition_base,
                        deadline=deadline,
                    )
                    return uci or "0000"

                if self._epsilon_pick_depth1 > 0.0 and roll < self._epsilon_pick_depth1:
                    uci = _v4_pick_move_by_root_scores(
                        pos,
                        self._v4,
                        depth=1,
                        pick="best",
                        rng=self._rng,
                        repetition_base=repetition_base,
                        deadline=deadline,
                    )
                    return uci or "0000"

                # Fixed-depth mode (ignore full ID).
                if self._fixed_depth is not None:
                    uci = _v4_pick_move_by_root_scores(
                        pos,
                        self._v4,
                        depth=int(self._fixed_depth),
                        pick="best",
                        rng=self._rng,
                        repetition_base=repetition_base,
                        deadline=deadline,
                    )
                    return uci or "0000"

                mv = self._agent.choose_move(pos, time_limit_s=float(think_s), repetition_base=repetition_base)
                return mv.uci() if mv else "0000"
        finally:
            # Restore monkeypatches.
            if old_pick is not None:
                setattr(self._v4, "pick_book_move", old_pick)
            if old_probe_best is not None:
                setattr(self._v4, "probe_bestmove", old_probe_best)
            if old_probe_bestmoves is not None:
                setattr(self._v4, "probe_bestmoves", old_probe_bestmoves)


class RandomAgent(Agent):
    def __init__(self, *, seed: int = 0):
        self.name = "random"
        self._rng = random.Random(int(seed))

    def choose_uci(self, board: chess.Board, *, think_s: float, game_id: str) -> str:
        moves = list(board.legal_moves)
        if not moves:
            return "0000"
        return self._rng.choice(moves).uci()


class StockfishUciAgent(Agent):
    def __init__(self, *, name: str, depth: int, path: str | None = None):
        import chess.engine

        self.name = name
        self._depth = int(depth)
        self._path = path or os.environ.get("STOCKFISH_PATH", "").strip()
        if not self._path:
            raise RuntimeError("STOCKFISH_PATH is not set")
        # python-chess accepts either a list[str] command or a single string, but on some
        # versions/platforms passing a bare string can be treated as an iterable of chars.
        # Use a 1-item argv list for reliability (especially on Windows).
        self._engine = chess.engine.SimpleEngine.popen_uci([self._path])

    def choose_uci(self, board: chess.Board, *, think_s: float, game_id: str) -> str:
        import chess.engine

        # Depth-limited (ignores think_s).
        res = self._engine.play(board, chess.engine.Limit(depth=self._depth))
        return res.move.uci() if res.move else "0000"

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass


def build_agent_specs(seed: int = 0) -> list[AgentSpec]:
    """
    Agent list in the exact order the user requested (1..12).
    """
    s = int(seed)
    return [
        AgentSpec(1, "Stockfish depth 5", lambda: StockfishUciAgent(name="sf_d5", depth=5)),
        AgentSpec(2, "agents_v4 current", lambda: AgentsV4Agent(name="v4", level=10, seed=s + 3)),
        AgentSpec(
            3,
            "agents_v4 no books/syzygy/exp",
            lambda: AgentsV4Agent(
                name="v4_nobooks_nosyzygy_noexp",
                level=10,
                opening_book=False,
                syzygy=False,
                experience_book=False,
                seed=s + 4,
            ),
        ),
        AgentSpec(
            4,
            "agents_v4 50% pick depth1",
            lambda: AgentsV4Agent(name="v4_eps50_d1", epsilon_pick_depth1=0.50, seed=s + 8),
        ),
        AgentSpec(
            5,
            "agents_v4 10% pick second best",
            lambda: AgentsV4Agent(name="v4_second10", second_best_prob=0.10, seed=s + 9),
        ),
        AgentSpec(6, "Stockfish depth 3", lambda: StockfishUciAgent(name="sf_d3", depth=3)),
        AgentSpec(7, "agents_v4 no nnue (lite)", lambda: AgentsV4Agent(name="v4_nonnu", nnue=False, sf_nnue=False, seed=s + 6)),
        AgentSpec(
            8,
            "agents_v4 no nnue + quick depth 3",
            lambda: AgentsV4Agent(name="v4_nonnu_d3", nnue=False, sf_nnue=False, fixed_depth=3, seed=s + 7),
        ),
        AgentSpec(
            9,
            "agents_v4 always second best",
            lambda: AgentsV4Agent(name="v4_second100", always_second_best=True, seed=s + 10),
        ),
        AgentSpec(10, "agents_v4 quick depth 3", lambda: AgentsV4Agent(name="v4_d3", fixed_depth=3, seed=s + 5)),
        AgentSpec(
            11,
            "agents_v4 no nnue always second best",
            lambda: AgentsV4Agent(name="v4_nonnu_second100", nnue=False, sf_nnue=False, always_second_best=True, seed=s + 11),
        ),
        AgentSpec(12, "pure random", lambda: RandomAgent(seed=s + 12)),
    ]

