from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from agents import AgentSpec, build_agent_specs


@dataclass
class GameResult:
    white_id: int
    black_id: int
    result: str  # "1-0" | "0-1" | "1/2-1/2" | "*"
    termination: str
    ply: int
    pgn: str


def _think_time_s(rem_s: float, inc_s: float) -> float:
    # Simple time manager: spend ~1/25 remaining + most of increment, with caps.
    if rem_s <= 0.05:
        return 0.05
    t = max(0.05, rem_s / 25.0 + max(0.0, inc_s * 0.90))
    # Keep bounded so one move can't stall the entire tournament.
    return float(min(t, 12.0, max(0.05, rem_s - 0.02)))


def _adjudicate_max_plies_with_stockfish(board: chess.Board, *, depth: int = 18) -> tuple[str, str]:
    """
    Adjudicate a max-plies game using Stockfish at a fixed depth.

    Policy (white POV centipawns):
      - cp >= +100 -> white wins
      - cp <= -100 -> black wins
      - otherwise draw
    """
    import chess.engine

    path = os.environ.get("STOCKFISH_PATH", "").strip()
    if not path:
        return "1/2-1/2", "max_plies_no_stockfish"

    eng = chess.engine.SimpleEngine.popen_uci([path])
    try:
        info = eng.analyse(board, chess.engine.Limit(depth=int(depth)))
        sc = info.get("score", None)
        cp = None
        if sc is not None:
            try:
                cp = sc.pov(chess.WHITE).score(mate_score=100000)
            except Exception:
                cp = None
        if cp is None:
            return "1/2-1/2", "max_plies_sf_no_score"
        if cp >= 100:
            return "1-0", f"max_plies_sf_cp={cp}"
        if cp <= -100:
            return "0-1", f"max_plies_sf_cp={cp}"
        return "1/2-1/2", f"max_plies_sf_cp={cp}"
    finally:
        try:
            eng.quit()
        except Exception:
            pass


def _play_one_game(
    *,
    white: AgentSpec,
    black: AgentSpec,
    start_fen: str,
    base_s: float,
    inc_s: float,
    max_plies: int,
    seed_tag: str,
) -> GameResult:
    w = white.make()
    b = black.make()

    board = chess.Board(start_fen)
    rem = {chess.WHITE: float(base_s), chess.BLACK: float(base_s)}
    inc = {chess.WHITE: float(inc_s), chess.BLACK: float(inc_s)}

    game = chess.pgn.Game()
    game.headers["Event"] = "testbot round-robin"
    game.headers["Site"] = "local"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["Round"] = seed_tag
    game.headers["White"] = f"{white.id}:{white.name}"
    game.headers["Black"] = f"{black.id}:{black.name}"
    game.headers["TimeControl"] = f"{int(base_s)}+{int(inc_s)}"
    game.headers["FEN"] = start_fen

    node = game
    termination = "unknown"
    forced_result = False

    try:
        for ply in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break

            side = board.turn
            think_s = _think_time_s(rem[side], inc[side])

            t0 = time.perf_counter()
            if side == chess.WHITE:
                uci = w.choose_uci(board, think_s=think_s, game_id=seed_tag)
            else:
                uci = b.choose_uci(board, think_s=think_s, game_id=seed_tag)
            dt = time.perf_counter() - t0

            # Update clock (simple): subtract wall time; add increment after move.
            rem[side] = max(0.0, rem[side] - dt)

            mv = None
            try:
                mv = chess.Move.from_uci(uci)
            except Exception:
                mv = None

            if mv is None or mv not in board.legal_moves:
                # Illegal: lose.
                termination = f"illegal_move:{uci}"
                if side == chess.WHITE:
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1-0"
                forced_result = True
                break

            board.push(mv)
            rem[side] = rem[side] + inc[side]
            node = node.add_variation(mv)

            # Simple flag-fall: if you used up all time, lose immediately.
            if rem[side] <= 0.0:
                termination = "flag"
                if side == chess.WHITE:
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1-0"
                forced_result = True
                break

        if not forced_result:
            if board.is_game_over(claim_draw=True):
                game.headers["Result"] = board.result(claim_draw=True)
                termination = board.outcome(claim_draw=True).termination.name if board.outcome(claim_draw=True) else "game_over"
            else:
                res, term = _adjudicate_max_plies_with_stockfish(board, depth=18)
                game.headers["Result"] = res
                termination = term

    finally:
        try:
            w.close()
        except Exception:
            pass
        try:
            b.close()
        except Exception:
            pass

    game.headers["Termination"] = termination

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    pgn_str = game.accept(exporter)
    res = game.headers["Result"]
    return GameResult(
        white_id=white.id,
        black_id=black.id,
        result=res,
        termination=termination,
        ply=len(board.move_stack),
        pgn=pgn_str,
    )


def _score(res: str) -> tuple[float, float]:
    if res == "1-0":
        return 1.0, 0.0
    if res == "0-1":
        return 0.0, 1.0
    if res == "1/2-1/2":
        return 0.5, 0.5
    return 0.0, 0.0


def _parse_agent_id(s: str) -> int | None:
    try:
        head = (s or "").split(":", 1)[0].strip()
        return int(head)
    except Exception:
        return None


def _load_existing_results(pgn_path: Path) -> dict[str, chess.pgn.Game]:
    """
    Load existing games from results.pgn keyed by Round tag.
    If duplicate Round tags exist, the last one wins.
    """
    if not pgn_path.exists():
        return {}
    out: dict[str, chess.pgn.Game] = {}
    with pgn_path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            rt = str(g.headers.get("Round", "") or "").strip()
            if not rt:
                continue
            out[rt] = g
    return out


def _gr_from_pgn_game(g: chess.pgn.Game) -> GameResult:
    wid = _parse_agent_id(str(g.headers.get("White", ""))) or 0
    bid = _parse_agent_id(str(g.headers.get("Black", ""))) or 0
    return GameResult(
        white_id=wid,
        black_id=bid,
        result=str(g.headers.get("Result", "*")).strip(),
        termination=str(g.headers.get("Termination", "")).strip(),
        ply=0,
        pgn="",
    )


def main() -> int:
    here = Path(__file__).resolve().parent
    out_dir = here / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    pgn_out = out_dir / "results.pgn"
    summary_out = out_dir / "summary.txt"

    # Config
    start_fen = chess.STARTING_FEN
    max_plies = int(os.environ.get("TESTBOT_MAX_PLIES", "200"))  # 100 moves
    seed = int(os.environ.get("TESTBOT_SEED", "0"))
    resume = os.environ.get("TESTBOT_RESUME", "1").strip().lower() in {"1", "true", "yes", "on"}

    # Main time control: 300+3
    base_main = 300.0
    inc_main = 3.0
    # Tiebreak: 180+0
    base_tb = 180.0
    inc_tb = 0.0

    # Each pairing:
    # - main: 4 games (each side twice)
    # - if tied on points: 2 games (each side once) at tb time control
    agents = build_agent_specs(seed=seed)
    n = len(agents)

    # Score matrix
    pts = {a.id: 0.0 for a in agents}
    games_played = {a.id: 0 for a in agents}
    pair_summary: list[str] = []
    game_no = 0

    def spec(i: int) -> AgentSpec:
        return agents[i - 1]

    name_by_id = {a.id: a.name for a in agents}

    existing_by_round: dict[str, chess.pgn.Game] = _load_existing_results(pgn_out) if resume else {}
    done_rounds: set[str] = set(existing_by_round.keys())
    if done_rounds:
        # Recompute points from existing games.
        for g in existing_by_round.values():
            wid = _parse_agent_id(str(g.headers.get("White", "")))
            bid = _parse_agent_id(str(g.headers.get("Black", "")))
            res = str(g.headers.get("Result", "*")).strip()
            if wid is None or bid is None:
                continue
            sw, sb = _score(res)
            pts[wid] += sw
            pts[bid] += sb
            games_played[wid] += 1
            games_played[bid] += 1
        game_no = len(done_rounds)

        # Rebuild per-pair summary for pairs that have at least the 4 main games completed.
        def _pair_points(tag: str, a_id: int, b_id: int) -> tuple[float, float]:
            pa = pb = 0.0
            for rt in (
                f"{tag}-main-Aw1",
                f"{tag}-main-Bw1",
                f"{tag}-main-Aw2",
                f"{tag}-main-Bw2",
                f"{tag}-tb-Aw",
                f"{tag}-tb-Bw",
            ):
                g = existing_by_round.get(rt)
                if g is None:
                    continue
                wid = _parse_agent_id(str(g.headers.get("White", "")))
                bid = _parse_agent_id(str(g.headers.get("Black", "")))
                res = str(g.headers.get("Result", "*")).strip()
                if wid is None or bid is None:
                    continue
                sw, sb = _score(res)
                if wid == a_id and bid == b_id:
                    pa += sw
                    pb += sb
                elif wid == b_id and bid == a_id:
                    pb += sw
                    pa += sb
            return pa, pb

        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                tag = f"{i:02d}v{j:02d}"
                main_done = all(
                    [
                        f"{tag}-main-Aw1" in done_rounds,
                        f"{tag}-main-Bw1" in done_rounds,
                        f"{tag}-main-Aw2" in done_rounds,
                        f"{tag}-main-Bw2" in done_rounds,
                    ]
                )
                if not main_done:
                    continue
                a = spec(i)
                b = spec(j)
                pa, pb = _pair_points(tag, a.id, b.id)
                pair_summary.append(
                    f"{tag}  {a.id}:{a.name} vs {b.id}:{b.name}  points {pa:.1f}-{pb:.1f}  OK"
                )

    def write_progress() -> None:
        lines = []
        lines.append(f"IN_PROGRESS  updated={time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        # In-progress ranking uses the same rules as the final table:
        # Total Points (per-opponent match result) -> Sonneborn-Berger -> Direct encounter -> Wins.
        pair_match: dict[tuple[int, int], tuple[float, float]] = {}
        for ln in pair_summary:
            try:
                _, rest = ln.split("  ", 1)
                sides, tail = rest.split("  points ", 1)
                a_s, b_s = sides.split(" vs ", 1)
                a_id = int(a_s.split(":", 1)[0])
                b_id = int(b_s.split(":", 1)[0])
                pts_s = tail.split("  ", 1)[0]
                pa_s, pb_s = pts_s.split("-", 1)
                pa = float(pa_s)
                pb = float(pb_s)
            except Exception:
                continue
            if pa > pb:
                pair_match[(a_id, b_id)] = (1.0, 0.0)
            elif pb > pa:
                pair_match[(a_id, b_id)] = (0.0, 1.0)
            else:
                pair_match[(a_id, b_id)] = (0.5, 0.5)

        match_points = {a.id: 0.0 for a in agents}
        wins = {a.id: 0 for a in agents}
        for (a_id, b_id), (sa, sb) in pair_match.items():
            match_points[a_id] += sa
            match_points[b_id] += sb
            if sa > sb:
                wins[a_id] += 1
            elif sb > sa:
                wins[b_id] += 1

        sb_score = {a.id: 0.0 for a in agents}
        for (a_id, b_id), (sa, sb) in pair_match.items():
            sb_score[a_id] += sa * match_points[b_id]
            sb_score[b_id] += sb * match_points[a_id]

        def _direct_encounter_cmp(a_id: int, b_id: int) -> float:
            if a_id < b_id:
                sa, sb = pair_match.get((a_id, b_id), (0.5, 0.5))
                return sa - sb
            sa, sb = pair_match.get((b_id, a_id), (0.5, 0.5))
            return sb - sa

        rows = [(a.id, match_points[a.id], sb_score[a.id], wins[a.id]) for a in agents]
        rows.sort(key=lambda r: (-r[1], -r[2], -r[3], r[0]))
        changed = True
        while changed:
            changed = False
            for i in range(len(rows) - 1):
                a = rows[i]
                b = rows[i + 1]
                if abs(a[1] - b[1]) < 1e-9 and abs(a[2] - b[2]) < 1e-9:
                    if _direct_encounter_cmp(a[0], b[0]) < 0:
                        rows[i], rows[i + 1] = rows[i + 1], rows[i]
                        changed = True

        lines.append("Current ranking (in-progress)")
        lines.append(f"{'Rank':>4}  {'Bot Name':<38}  {'Pts':>5}  {'S-B':>7}  {'W':>3}")
        lines.append(f"{'-'*4:>4}  {'-'*38:<38}  {'-'*5:>5}  {'-'*7:>7}  {'-'*3:>3}")
        for idx, (aid, mp, sb, w) in enumerate(rows, start=1):
            nm = name_by_id.get(aid, "?")
            lines.append(f"{idx:>4}  {nm:<38.38}  {mp:>5.1f}  {sb:>7.2f}  {w:>3d}")
        lines.append("")
        lines.append("Per-pair summary:")
        lines.extend(pair_summary)
        summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Fresh outputs only when not resuming (or file absent).
    if (not resume) or (not pgn_out.exists()):
        pgn_out.write_text("", encoding="utf-8")
    write_progress()

    # Round-robin pairings
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            a = spec(i)
            b = spec(j)

            tag = f"{i:02d}v{j:02d}"
            results: list[GameResult] = []

            # 4 main games: A as White twice, B as White twice.
            for k in range(2):
                rt = f"{tag}-main-Aw{k+1}"
                if rt in done_rounds and rt in existing_by_round:
                    results.append(_gr_from_pgn_game(existing_by_round[rt]))
                else:
                    gr = _play_one_game(
                        white=a,
                        black=b,
                        start_fen=start_fen,
                        base_s=base_main,
                        inc_s=inc_main,
                        max_plies=max_plies,
                        seed_tag=rt,
                    )
                    results.append(gr)
                    game_no += 1
                    print(
                        f"\n=== finished game {game_no}: {gr.white_id} vs {gr.black_id}  {gr.result}  ({gr.termination}) ===\n"
                    )
                    print(gr.pgn)
                    print("\n")
                    with pgn_out.open("a", encoding="utf-8") as f:
                        f.write(gr.pgn)
                        f.write("\n\n")
                    done_rounds.add(rt)

                rt = f"{tag}-main-Bw{k+1}"
                if rt in done_rounds and rt in existing_by_round:
                    results.append(_gr_from_pgn_game(existing_by_round[rt]))
                else:
                    gr = _play_one_game(
                        white=b,
                        black=a,
                        start_fen=start_fen,
                        base_s=base_main,
                        inc_s=inc_main,
                        max_plies=max_plies,
                        seed_tag=rt,
                    )
                    results.append(gr)
                    game_no += 1
                    print(
                        f"\n=== finished game {game_no}: {gr.white_id} vs {gr.black_id}  {gr.result}  ({gr.termination}) ===\n"
                    )
                    print(gr.pgn)
                    print("\n")
                    with pgn_out.open("a", encoding="utf-8") as f:
                        f.write(gr.pgn)
                        f.write("\n\n")
                    done_rounds.add(rt)

            pa = 0.0
            pb = 0.0
            for gr in results:
                sw, sb = _score(gr.result)
                if gr.white_id == a.id:
                    pa += sw
                    pb += sb
                else:
                    pb += sw
                    pa += sb
                # On resume, existing games already contributed to pts/games_played.
                if gr.pgn:
                    pts[gr.white_id] += sw
                    pts[gr.black_id] += sb
                    games_played[gr.white_id] += 1
                    games_played[gr.black_id] += 1

            tb_results: list[GameResult] = []
            if abs(pa - pb) < 1e-9:
                # Tiebreak: 2 games, each side once.
                rt = f"{tag}-tb-Aw"
                if rt in done_rounds and rt in existing_by_round:
                    tb_results.append(_gr_from_pgn_game(existing_by_round[rt]))
                else:
                    gr = _play_one_game(
                        white=a,
                        black=b,
                        start_fen=start_fen,
                        base_s=base_tb,
                        inc_s=inc_tb,
                        max_plies=max_plies,
                        seed_tag=rt,
                    )
                    tb_results.append(gr)
                    game_no += 1
                    print(
                        f"\n=== finished game {game_no}: {gr.white_id} vs {gr.black_id}  {gr.result}  ({gr.termination}) ===\n"
                    )
                    print(gr.pgn)
                    print("\n")
                    with pgn_out.open("a", encoding="utf-8") as f:
                        f.write(gr.pgn)
                        f.write("\n\n")
                    done_rounds.add(rt)

                rt = f"{tag}-tb-Bw"
                if rt in done_rounds and rt in existing_by_round:
                    tb_results.append(_gr_from_pgn_game(existing_by_round[rt]))
                else:
                    gr = _play_one_game(
                        white=b,
                        black=a,
                        start_fen=start_fen,
                        base_s=base_tb,
                        inc_s=inc_tb,
                        max_plies=max_plies,
                        seed_tag=rt,
                    )
                    tb_results.append(gr)
                    game_no += 1
                    print(
                        f"\n=== finished game {game_no}: {gr.white_id} vs {gr.black_id}  {gr.result}  ({gr.termination}) ===\n"
                    )
                    print(gr.pgn)
                    print("\n")
                    with pgn_out.open("a", encoding="utf-8") as f:
                        f.write(gr.pgn)
                        f.write("\n\n")
                    done_rounds.add(rt)
                for gr in tb_results:
                    sw, sb = _score(gr.result)
                    if gr.white_id == a.id:
                        pa += sw
                        pb += sb
                    else:
                        pb += sw
                        pa += sb
                    if gr.pgn:
                        pts[gr.white_id] += sw
                        pts[gr.black_id] += sb
                        games_played[gr.white_id] += 1
                        games_played[gr.black_id] += 1

                if abs(pa - pb) < 1e-9:
                    pass

            # Replace any previous summary line for this pairing (resume-safe).
            pair_summary = [ln for ln in pair_summary if not ln.startswith(f"{tag}  ")]
            pair_summary.append(
                f"{tag}  {a.id}:{a.name} vs {b.id}:{b.name}  points {pa:.1f}-{pb:.1f}  OK"
            )

            # Persist progress after each pairing (safe for long runs / interruptions)
            write_progress()

    # Final ranking by per-opponent match result:
    #   win vs opponent = 1.0, draw = 0.5, loss = 0.0
    # This matches a 12-bot round robin "max score = 11.0" table.
    pair_match: dict[tuple[int, int], tuple[float, float]] = {}
    for ln in pair_summary:
        # Format:
        # 01v02  1:Name vs 2:Name  points 3.0-1.0  OK
        try:
            left, rest = ln.split("  ", 1)
            sides, tail = rest.split("  points ", 1)
            a_s, b_s = sides.split(" vs ", 1)
            a_id = int(a_s.split(":", 1)[0])
            b_id = int(b_s.split(":", 1)[0])
            pts_s = tail.split("  ", 1)[0]
            pa_s, pb_s = pts_s.split("-", 1)
            pa = float(pa_s)
            pb = float(pb_s)
        except Exception:
            continue
        if pa > pb:
            pair_match[(a_id, b_id)] = (1.0, 0.0)
        elif pb > pa:
            pair_match[(a_id, b_id)] = (0.0, 1.0)
        else:
            pair_match[(a_id, b_id)] = (0.5, 0.5)

    match_points = {a.id: 0.0 for a in agents}
    wins = {a.id: 0 for a in agents}
    for (a_id, b_id), (sa, sb) in pair_match.items():
        match_points[a_id] += sa
        match_points[b_id] += sb
        if sa > sb:
            wins[a_id] += 1
        elif sb > sa:
            wins[b_id] += 1

    sb_score = {a.id: 0.0 for a in agents}
    for (a_id, b_id), (sa, sb) in pair_match.items():
        # Sonneborn-Berger:
        # add opponent full score for wins, half opponent score for draws.
        sb_score[a_id] += sa * match_points[b_id]
        sb_score[b_id] += sb * match_points[a_id]

    def _direct_encounter_cmp(a_id: int, b_id: int) -> float:
        if a_id < b_id:
            sa, sb = pair_match.get((a_id, b_id), (0.5, 0.5))
            return sa - sb
        sa, sb = pair_match.get((b_id, a_id), (0.5, 0.5))
        return sb - sa

    rows = [(a.id, match_points[a.id], sb_score[a.id], wins[a.id]) for a in agents]

    # Custom sort with direct-encounter secondary to S-B when needed.
    def _sort_key(row: tuple[int, float, float, int]) -> tuple[float, float, int, int]:
        aid, mp, sb, w = row
        # direct encounter is handled in a post-pass for exact point+SB ties
        return (-mp, -sb, -w, aid)

    rows.sort(key=_sort_key)

    # Post-pass: for adjacent exact ties on points and S-B, use direct encounter.
    changed = True
    while changed:
        changed = False
        for i in range(len(rows) - 1):
            a = rows[i]
            b = rows[i + 1]
            if abs(a[1] - b[1]) < 1e-9 and abs(a[2] - b[2]) < 1e-9:
                de = _direct_encounter_cmp(a[0], b[0])
                if de < 0:
                    rows[i], rows[i + 1] = rows[i + 1], rows[i]
                    changed = True

    lines = []
    lines.append("Final table")
    lines.append("")
    lines.append(f"{'Rank':>4}  {'Bot Name':<38}  {'Pts':>5}  {'S-B':>7}  {'W':>3}")
    lines.append(f"{'-'*4:>4}  {'-'*38:<38}  {'-'*5:>5}  {'-'*7:>7}  {'-'*3:>3}")
    for idx, (aid, mp, sb, w) in enumerate(rows, start=1):
        nm = name_by_id.get(aid, "?")
        lines.append(f"{idx:>4}  {nm:<38.38}  {mp:>5.1f}  {sb:>7.2f}  {w:>3d}")
    lines.append("")
    lines.append("Per-pair summary:")
    lines.extend(pair_summary)

    summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("")
    print(f"wrote: {pgn_out}")
    print(f"wrote: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

