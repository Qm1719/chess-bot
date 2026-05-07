from __future__ import annotations

import time
from pathlib import Path

import chess.pgn

from agents import build_agent_specs


def _parse_agent_id(s: str) -> int | None:
    try:
        head = (s or "").split(":", 1)[0].strip()
        return int(head)
    except Exception:
        return None


def _score(res: str) -> tuple[float, float]:
    if res == "1-0":
        return 1.0, 0.0
    if res == "0-1":
        return 0.0, 1.0
    if res == "1/2-1/2":
        return 0.5, 0.5
    return 0.0, 0.0


def _load_existing_by_round(pgn_path: Path) -> dict[str, chess.pgn.Game]:
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


def main() -> int:
    here = Path(__file__).resolve().parent
    out_dir = here / "out"
    pgn_out = out_dir / "results.pgn"
    summary_out = out_dir / "summary.txt"

    agents = build_agent_specs(seed=0)
    n = len(agents)
    name_by_id = {a.id: a.name for a in agents}

    existing_by_round = _load_existing_by_round(pgn_out)
    done_rounds = set(existing_by_round.keys())

    # Build per-pair points based on the recorded games.
    pair_summary: list[str] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            tag = f"{i:02d}v{j:02d}"
            a_id = i
            b_id = j

            def _acc(rt: str, pa: float, pb: float) -> tuple[float, float]:
                g = existing_by_round.get(rt)
                if g is None:
                    return pa, pb
                wid = _parse_agent_id(str(g.headers.get("White", "")))
                bid = _parse_agent_id(str(g.headers.get("Black", "")))
                res = str(g.headers.get("Result", "*")).strip()
                if wid is None or bid is None:
                    return pa, pb
                sw, sb = _score(res)
                if wid == a_id and bid == b_id:
                    return pa + sw, pb + sb
                if wid == b_id and bid == a_id:
                    return pa + sb, pb + sw
                return pa, pb

            pa = pb = 0.0
            main_tags = [f"{tag}-main-Aw1", f"{tag}-main-Bw1", f"{tag}-main-Aw2", f"{tag}-main-Bw2"]
            main_done = all(rt in done_rounds for rt in main_tags)
            if not main_done:
                continue
            for rt in main_tags:
                pa, pb = _acc(rt, pa, pb)

            pair_summary.append(
                f"{tag}  {a_id}:{name_by_id.get(a_id,'?')} vs {b_id}:{name_by_id.get(b_id,'?')}  "
                f"points {pa:.1f}-{pb:.1f}  OK"
            )

    # Ranking from pair_summary (same as tournament.py write_progress/final table).
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

    lines: list[str] = []
    lines.append(f"IN_PROGRESS  updated={time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Current ranking (from existing PGNs)")
    lines.append(f"{'Rank':>4}  {'Bot Name':<38}  {'Pts':>5}  {'S-B':>7}  {'W':>3}")
    lines.append(f"{'-'*4:>4}  {'-'*38:<38}  {'-'*5:>5}  {'-'*7:>7}  {'-'*3:>3}")
    for idx, (aid, mp, sb, w) in enumerate(rows, start=1):
        nm = name_by_id.get(aid, "?")
        lines.append(f"{idx:>4}  {nm:<38.38}  {mp:>5.1f}  {sb:>7.2f}  {w:>3d}")
    lines.append("")
    lines.append("Per-pair summary:")
    lines.extend(pair_summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

