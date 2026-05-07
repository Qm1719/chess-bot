from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

# ops/selftrain.py -> submission v5
V5_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = V5_ROOT / "engine"
DATA_DIR = V5_ROOT / "data"
# Default env file: `selftrain.env` in submission v5 (override with SELFTRAIN_ENV_PATH)
_SELFTRAIN_DEFAULT_ENV = V5_ROOT / "selftrain.env"
_SELFTRAIN_FALLBACK_ENV = V5_ROOT / ".env"
INGEST_LOG_PATH = DATA_DIR / "selftrain_ingest.log"


def _env_str(key: str, default: str) -> str:
    """Like os.environ.get but treat empty/whitespace as unset (env files often use KEY=)."""
    v = os.environ.get(key, "")
    if isinstance(v, str):
        v = v.strip()
    return v if v else default


def _add_engine_import_path() -> None:
    p = str(ENGINE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    p2 = str(V5_ROOT)
    if p2 not in sys.path:
        sys.path.insert(0, p2)


@dataclass
class SelfTrainState:
    version: int = 1
    day: str | None = None  # YYYYMMDD
    runs_today: int = 0  # count of selftrain invocations today (after daily reset)
    training_games_completed: int = 0
    # Game IDs that were fully ingested (analysis -> experience book), idempotent.
    ingested_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.ingested_ids is None:
            self.ingested_ids = []


def _state_path() -> Path:
    return Path(_env_str("SELFTRAIN_STATE_PATH", str(DATA_DIR / "selftrain_state.json")))


def _load_state() -> SelfTrainState:
    p = _state_path()
    if not p.exists():
        return SelfTrainState()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        st = SelfTrainState(
            version=int(d.get("version", 1)),
            day=str(d.get("day")) if d.get("day") else None,
            runs_today=int(d.get("runs_today", 0)),
            training_games_completed=int(d.get("training_games_completed", 0)),
            ingested_ids=list(d.get("ingested_ids", []) or []),
        )
        return st
    except Exception:
        return SelfTrainState()


def _save_state(st: SelfTrainState) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(st), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)


def _today_yyyymmdd() -> str:
    return time.strftime("%Y%m%d")


def _daily_reset_enabled() -> bool:
    v = os.environ.get("SELFTRAIN_DAILY_RESET", "1").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _append_ingest_log(msg: str) -> None:
    """Append one line to `data/selftrain_ingest.log` (training / ingest audit)."""
    try:
        INGEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        with INGEST_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{ts} {msg}\n")
    except Exception:
        pass


def _resolve_lichess_token() -> str | None:
    """
    Canonical name: LICHESS_TOKEN (same OAuth token everywhere).
    lichess-bot still reads LICHESS_BOT_TOKEN; we sync that before subprocess startup.
    """
    t = (
        os.environ.get("LICHESS_TOKEN", "").strip()
        or os.environ.get("LICHESS_BOT_TOKEN", "").strip()
        or os.environ.get("LICHESS_API_TOKEN", "").strip()
    )
    return t or None


def _lichess_bot_name() -> str:
    """Lichess account name (not secret). LICHESS_BOT_NAME preferred; SELFTRAIN_BOT_USER is legacy."""
    return (
        os.environ.get("LICHESS_BOT_NAME", "").strip()
        or os.environ.get("SELFTRAIN_BOT_USER", "").strip()
    )


def _sync_lichess_env_for_bot() -> None:
    """Expose token as LICHESS_BOT_TOKEN for lichess-bot if only LICHESS_TOKEN is set."""
    tok = _resolve_lichess_token()
    if not tok:
        return
    if not os.environ.get("LICHESS_BOT_TOKEN", "").strip():
        os.environ["LICHESS_BOT_TOKEN"] = tok
    if not os.environ.get("LICHESS_TOKEN", "").strip():
        os.environ["LICHESS_TOKEN"] = tok


@dataclass
class RunSummary:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    elo_delta: int = 0
    exp_rows_added: int = 0
    # (game_id, keyhex, uci)
    exp_used: list[tuple[str, str, str]] | None = None

    def __post_init__(self) -> None:
        if self.exp_used is None:
            self.exp_used = []


def _truncate_log_by_lines(path: Path, keep_last: int = 10_000) -> None:
    try:
        if not path.exists():
            return
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(data) <= keep_last:
            return
        tail = data[-keep_last:]
        path.write_text("\n".join(tail) + "\n", encoding="utf-8")
    except Exception:
        return


def _read_exp_usage(log_path: Path) -> list[tuple[str, str, str]]:
    if not log_path.exists():
        return []
    out: list[tuple[str, str, str]] = []
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            gid, keyhex, uci = parts
            out.append((gid.strip(), keyhex.strip(), uci.strip()))
    except Exception:
        return out
    return out


def _http_json(
    method: str,
    url: str,
    *,
    token: str | None,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: int = 60,
) -> tuple[int, Any]:
    h = dict(headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    # Lichess supports ndjson; for single-game export this is still fine.
    if "Accept" not in h and "accept" not in {k.lower() for k in h.keys()}:
        h["Accept"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method.upper(), headers=h)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (user-controlled base URL is intentional)
        body = resp.read()
        code = int(getattr(resp, "status", 200) or 200)
    # Some endpoints can return application/x-ndjson with a single line.
    if not body:
        return code, None
    s = body.decode("utf-8", errors="replace").strip()
    if "\n" in s:
        # If multiple JSON objects, take the first (shouldn't happen for single game export)
        s = s.splitlines()[0]
    return code, json.loads(s)


def _lichess_base() -> str:
    return os.environ.get("LICHESS_BASE", "https://lichess.org").rstrip("/")


def export_game_json(game_id: str, token: str | None) -> dict[str, Any]:
    # Documented: GET /game/export/{gameId} with `Accept: application/json`
    base = _lichess_base()
    q = urllib.parse.urlencode(
        {
            "moves": "true",
            "pgnInJson": "true",
            "tags": "true",
            "clocks": "true",
            "evals": "true",
            "accuracy": "true",
            "opening": "true",
            "division": "true",
            "literate": "true",
        }
    )
    url = f"{base}/game/export/{urllib.parse.quote(game_id)}?{q}"
    code, obj = _http_json("GET", url, token=token, headers={"Accept": "application/json"})
    if code // 100 != 2 or not isinstance(obj, dict):
        raise RuntimeError(f"export_game_json failed: http={code} id={game_id!r} obj_type={type(obj).__name__}")
    return obj


def analysis_ready(g: dict[str, Any]) -> bool:
    a = g.get("analysis", None)
    if not isinstance(a, list) or len(a) < 2:
        return False
    # Heuristic: "judgment" only appears after Lichess server analysis has processed the game
    for x in a:
        if isinstance(x, dict) and x.get("judgment"):
            return True
    return False


def _bot_outcome(g: dict[str, Any], bot_user: str) -> str:
    """
    Return one of: 'win', 'loss', 'draw', 'unknown'.
    Uses Lichess game export JSON fields.
    """
    is_white = _bot_color_from_json(g, bot_user)
    if is_white is None:
        return "unknown"
    status = str(g.get("status", "") or "").strip().lower()
    winner = str(g.get("winner", "") or "").strip().lower()
    if winner in {"white", "black"}:
        bot_won = (winner == "white") if is_white else (winner == "black")
        return "win" if bot_won else "loss"
    if status in {"draw", "stalemate", "repetition", "timevsinsufficient", "insufficient", "agreement"}:
        return "draw"
    # Many other terminal statuses exist; if no winner is provided, treat as unknown.
    return "unknown"


def _is_abort(g: dict[str, Any]) -> bool:
    status = str(g.get("status", "") or "").strip().lower()
    return status in {"aborted", "nostart"}


def wait_for_server_analysis(
    game_id: str, token: str | None, *, max_wait_s: int, sleep_s: float
) -> dict[str, Any]:
    t0 = time.time()
    last: dict[str, Any] | None = None
    while time.time() - t0 <= max_wait_s:
        g = export_game_json(game_id, token=token)
        last = g
        if analysis_ready(g):
            return g
        time.sleep(sleep_s)
    if last is None:
        raise RuntimeError("wait_for_server_analysis: no data")
    return last


def _telegram_send(text: str) -> None:
    tok = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not tok or not chat:
        return
    # Keep messages short (RAM-friendly) and avoid huge stack traces
    if len(text) > 3500:
        text = text[:3500] + "\n... (truncated)"
    q = urllib.parse.urlencode(
        {
            "chat_id": chat,
            "text": text,
            "disable_web_page_preview": "true",
        }
    )
    url = f"https://api.telegram.org/bot{tok}/sendMessage?{q}"
    try:
        urllib.request.urlopen(url, timeout=30).read()  # noqa: S310
    except Exception:
        return


def _discord_send(text: str) -> None:
    wh = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not wh:
        return
    debug = os.environ.get("SELFTRAIN_NOTIFY_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    if len(text) > 2000:
        text = text[:2000] + "\n... (truncated)"
    payload = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310
        wh,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            # Some frontends (Cloudflare/WAF) may reject requests without a UA.
            "User-Agent": "Chess2Selftrain/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            _ = resp.read()
            code = int(getattr(resp, "status", 204) or 204)
        if debug:
            print(f"[selftrain] discord webhook ok http={code}", flush=True)
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if debug:
            print(f"[selftrain] discord webhook failed http={getattr(e, 'code', None)} body={body!r}", flush=True)
        return
    except Exception as e:
        if debug:
            print(f"[selftrain] discord webhook failed: {e}", flush=True)
        return


def notify(msg: str) -> None:
    _telegram_send(msg)
    _discord_send(msg)

def _notify_only_start_end() -> bool:
    """
    If enabled, only send notifications at start and when the run finishes the daily cap.
    All other progress messages stay local (stdout/logs).
    """
    v = os.environ.get("SELFTRAIN_NOTIFY_ONLY_START_END", "1").strip().lower()
    return v in {"1", "true", "yes", "on"}


def notify_event(kind: str, msg: str) -> None:
    """
    kind: 'start' | 'end' | 'error' | 'progress'
    """
    if _notify_only_start_end() and kind not in {"start", "end"}:
        return
    notify(msg)


def _say(msg: str) -> None:
    """Always print; also notify if channels configured."""
    print(f"[selftrain] {msg}", flush=True)
    notify_event("progress", msg)


_JUDGMENT_SET = {"Blunder", "Mistake", "Inaccuracy"}


def _bot_color_from_json(g: dict[str, Any], bot_user: str) -> bool | None:
    wname = ((g.get("players") or {}).get("white") or {}).get("user") or {}
    bname = ((g.get("players") or {}).get("black") or {}).get("user") or {}
    w = str(wname.get("name") or "").lower()
    b = str(bname.get("name") or "").lower()
    u = bot_user.lower()
    if u and u == w:
        return True
    if u and u == b:
        return False
    return None


def ingest_experience_from_lichess_json(
    g: dict[str, Any], *, bot_user: str, only_judgments: bool = True
) -> int:
    """
    Reads Lichess's server analysis array (judgment + best) and stores best moves
    in experience_book.sqlite3 for positions where the bot played a labeled move.

    This intentionally relies on the official `GET /game/export/{id}` JSON fields, not
    the website "request analysis" click path.
    """
    if not analysis_ready(g):
        return 0
    is_white = _bot_color_from_json(g, bot_user)
    if is_white is None:
        raise RuntimeError("Cannot determine bot color; set LICHESS_BOT_NAME correctly")

    moves = str(g.get("moves", "") or "").split()
    analysis = g.get("analysis", None)
    if not isinstance(analysis, list):
        return 0
    # Basic sanity: Lichess provides analysis[0] for the startpos, then one entry per half-move in most cases.
    if len(analysis) < len(moves) + 1:
        return 0

    _add_engine_import_path()
    from experience_book import _connect, add_entry, DB_PATH  # type: ignore

    con = _connect(str(DB_PATH))
    added = 0
    try:
        # Lichess JSON aligns `judgment` on the *destination* index after a move.
        # `i` is 0-based move index. After playing moves[i], we look at analysis[i+1].
        for i in range(0, len(moves)):
            side_is_white = (i % 2) == 0
            if side_is_white != is_white:
                continue
            pos = analysis[i + 1] if (i + 1) < len(analysis) else None
            if not isinstance(pos, dict):
                continue
            j = pos.get("judgment", None)
            if not isinstance(j, dict):
                continue
            name = str(j.get("name", "") or "")
            if only_judgments and name not in _JUDGMENT_SET:
                continue

            best = str(pos.get("best", "") or "").strip()
            if len(best) < 4 or len(best) > 5:
                continue

            # FEN *before* bot's move at index i: play i prior moves
            from chess import Board  # type: ignore

            b = Board()
            for k in range(0, i):
                b.push_uci(moves[k])
            fen = b.fen()

            # Store the recommended correction move (not the played move)
            add_entry(con, fen, best, depth=None, score=None)
            added += 1
    finally:
        con.close()
    return added


def _sf_enabled() -> bool:
    v = os.environ.get("SELFTRAIN_LOCAL_SF", "1").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _sf_path() -> str:
    try:
        # Centralized detection (supports submission v5/engine/stockfish/ layout).
        from engine.stockfish_path import detect_stockfish_path  # type: ignore

        return str(detect_stockfish_path())
    except Exception:
        pass
    return "stockfish"


def _sf_depth1() -> int:
    try:
        return int(os.environ.get("SELFTRAIN_SF_PASS1_DEPTH", "12"))
    except Exception:
        return 12


def _sf_depth2() -> int:
    try:
        return int(os.environ.get("SELFTRAIN_SF_PASS2_DEPTH", "18"))
    except Exception:
        return 18


def _sf_mistake_cp() -> int:
    try:
        return int(os.environ.get("SELFTRAIN_SF_MISTAKE_CP", "80"))
    except Exception:
        return 100


def _sf_blunder_cp() -> int:
    try:
        return int(os.environ.get("SELFTRAIN_SF_BLUNDER_CP", "200"))
    except Exception:
        return 200


def _delete_win_pgns() -> bool:
    v = os.environ.get("SELFTRAIN_DELETE_WIN_PGNS", "1").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _sf_analyse_game_two_pass(pgn_path: Path, *, bot_user: str) -> list[tuple[str, str, int]]:
    """
    Return list of (fen_before, best_uci, drop_cp) for bot-side mistakes/blunders.
    Pass 1 at depth1 scans all bot plies and flags drops >= mistake_cp.
    Pass 2 at depth2 rechecks only flagged plies and returns those still >= mistake_cp.
    """
    try:
        import chess  # type: ignore
        import chess.engine  # type: ignore
        import chess.pgn  # type: ignore
    except Exception as e:
        _say(f"local SF: missing python-chess; skipping local analysis ({e})")
        return []

    depth1 = _sf_depth1()
    depth2 = _sf_depth2()
    mistake_cp = _sf_mistake_cp()
    blunder_cp = _sf_blunder_cp()
    sf_path = _sf_path()
    _say(f"local SF: using engine={sf_path!r} depths={depth1}/{depth2} thresholds m={mistake_cp} b={blunder_cp}")

    # Parse PGN into move list and detect bot color.
    try:
        with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
            game = chess.pgn.read_game(f)
    except Exception as e:
        _say(f"local SF: failed to read PGN {pgn_path.name}: {e}")
        return []
    if game is None:
        _say(f"local SF: PGN parse returned no game for {pgn_path.name}")
        return []

    headers = game.headers
    w = str(headers.get("White", "") or "").strip().lower().removeprefix("bot ").strip()
    b = str(headers.get("Black", "") or "").strip().lower().removeprefix("bot ").strip()
    u = bot_user.strip().lower()
    if u == w:
        bot_is_white = True
    elif u == b:
        bot_is_white = False
    else:
        _say(f"local SF: cannot detect bot color from PGN headers (White={headers.get('White')} Black={headers.get('Black')})")
        return []

    moves_uci: list[str] = []
    node = game
    for mv in game.mainline_moves():
        moves_uci.append(mv.uci())

    if not moves_uci:
        return []

    def _cp_pov(score) -> int:
        # Normalize to bot POV. Use a large mate score cap for deltas.
        try:
            s = score.pov(chess.WHITE if bot_is_white else chess.BLACK)
        except Exception:
            s = score
        v = s.score(mate_score=100000)
        return int(v or 0)

    def _first_info(x):
        if isinstance(x, list):
            return x[0] if x else {}
        return x if isinstance(x, dict) else {}

    def board_before_ply(ply: int) -> chess.Board:
        b0 = chess.Board()
        for i in range(ply - 1):
            b0.push_uci(moves_uci[i])
        return b0

    # Start SF once for this game.
    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    except Exception as e:
        _say(f"local SF: failed to start stockfish at {sf_path!r}: {e}")
        return []

    try:
        try:
            engine.configure({"Threads": 1, "Hash": 128})
        except Exception:
            pass

        flagged: list[tuple[int, str, str]] = []  # (ply, fen_before, played_uci)
        # Pass 1
        for ply, played_uci in enumerate(moves_uci, start=1):
            side_is_white = (ply % 2) == 1
            if side_is_white != bot_is_white:
                continue
            bpos = board_before_ply(ply)
            fen_before = bpos.fen()
            info_best = _first_info(
                engine.analyse(
                    bpos,
                    chess.engine.Limit(depth=depth1),
                    multipv=1,
                    info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
                )
            )
            sc1 = info_best.get("score")
            pv = info_best.get("pv") or []
            best_mv = pv[0] if pv else None
            if best_mv is None:
                try:
                    best_mv = engine.play(bpos, chess.engine.Limit(depth=depth1)).move
                except Exception:
                    best_mv = None
            if best_mv is None or sc1 is None:
                continue
            best_cp = _cp_pov(sc1)

            try:
                b_after = bpos.copy(stack=True)
                b_after.push_uci(played_uci)
            except Exception:
                continue
            info_played = _first_info(
                engine.analyse(
                    b_after,
                    chess.engine.Limit(depth=depth1),
                    multipv=1,
                    info=chess.engine.INFO_SCORE,
                )
            )
            sc2 = info_played.get("score")
            if sc2 is None:
                continue
            played_cp = _cp_pov(sc2)
            drop = best_cp - played_cp
            if drop >= mistake_cp:
                flagged.append((ply, fen_before, played_uci))

        # Pass 2 (recheck flagged)
        out: list[tuple[str, str, int]] = []
        for ply, fen_before, played_uci in flagged:
            bpos = board_before_ply(ply)
            info_best2 = _first_info(
                engine.analyse(
                    bpos,
                    chess.engine.Limit(depth=depth2),
                    multipv=1,
                    info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
                )
            )
            scb = info_best2.get("score")
            pv2 = info_best2.get("pv") or []
            best_mv2 = pv2[0] if pv2 else None
            if best_mv2 is None:
                try:
                    best_mv2 = engine.play(bpos, chess.engine.Limit(depth=depth2)).move
                except Exception:
                    best_mv2 = None
            if best_mv2 is None or scb is None:
                continue
            best_cp2 = _cp_pov(scb)
            best_uci2 = best_mv2.uci()

            try:
                b_after = bpos.copy(stack=True)
                b_after.push_uci(played_uci)
            except Exception:
                continue
            info_played2 = _first_info(
                engine.analyse(
                    b_after,
                    chess.engine.Limit(depth=depth2),
                    multipv=1,
                    info=chess.engine.INFO_SCORE,
                )
            )
            scp = info_played2.get("score")
            if scp is None:
                continue
            played_cp2 = _cp_pov(scp)
            drop2 = best_cp2 - played_cp2
            if drop2 >= mistake_cp:
                out.append((fen_before, best_uci2, drop2))

        if out:
            _say(
                f"local SF: suggestions={len(out)} (depths {depth1}/{depth2}, thresholds m={mistake_cp} b={blunder_cp}) for {pgn_path.name}"
            )
        return out
    finally:
        try:
            engine.quit()
        except Exception:
            pass


def _extract_id_from_pgn_path(p: Path) -> str | None:
    m = re.search(r" - ([A-Za-z0-9]{8})\.pgn$", p.name)
    if m:
        return m.group(1)
    m2 = re.search(r"([A-Za-z0-9]{8})\.pgn$", p.name)
    if m2:
        return m2.group(1)
    return None


def discover_pgn_files(pgn_dir: Path) -> list[Path]:
    """
    Return all PGN files recursively under pgn_dir.
    Used only for diagnostics (not for the batch counter), because archives may live underneath.
    """
    if not pgn_dir.exists():
        return []
    out: list[Path] = []
    for p in sorted(pgn_dir.glob("**/*.pgn")):
        if p.is_file():
            out.append(p)
    return out


def discover_unarchived_pgn_files(pgn_dir: Path, archive_dir: Path) -> list[Path]:
    """
    PGN records ready to ingest/archive: files directly under pgn_dir (not under archive_dir).
    This matches lichess-bot's default behavior of writing per-game PGNs in the configured folder.
    """
    if not pgn_dir.exists():
        return []
    out: list[Path] = []
    for p in sorted(pgn_dir.glob("*.pgn")):
        if not p.is_file():
            continue
        # Safety: ignore anything that is already inside archive_dir (in case user points pgn_dir too wide)
        try:
            if archive_dir.exists() and archive_dir.resolve() in p.resolve().parents:
                continue
        except Exception:
            pass
        out.append(p)
    return out


def _tail_text(p: Path, max_bytes: int = 32_000) -> str:
    try:
        b = p.read_bytes()
    except Exception:
        return ""
    if len(b) > max_bytes:
        b = b[-max_bytes:]
    return b.decode("utf-8", errors="replace")


def _bot_log_path() -> Path:
    return Path(os.environ.get("SELFTRAIN_BOT_LOG", str(DATA_DIR / "selftrain_lichess_bot.log")))


def _is_token_error(text: str) -> bool:
    t = (text or "").lower()
    return (
        "error in retrieving information about the bot's token" in t
        or "check that it was copied correctly into your configuration file" in t
        or "invalid token" in t
        or "invalid oauth" in t
    )


def _is_transient_network_error(text: str) -> bool:
    t = (text or "").lower()
    # Keep this narrow: only retry on likely transient connectivity issues.
    needles = [
        "connection reset",
        "timed out",
        "temporary failure",
        "name or service not known",
        "dns",
        "remote disconnected",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "too many requests",
        "rate limit",
    ]
    return any(n in t for n in needles)


def _run_bot(
    lichess_bot_dir: Path,
    *,
    max_seconds: int,
    max_new_pgns: int,
    pgn_dir: Path,
    archive_dir: Path,
    env: dict[str, str],
) -> int:
    """
    Run lichess-bot for up to max_seconds or until max_new_pgns new PGNs appear.
    If lichess-bot crashes due to token/connectivity issues, retry with backoff.
    """
    cmd = [sys.executable, "-u", "lichess-bot.py"]
    t0 = time.time()
    # We count *unarchived* PGNs (batch-ready records), not deltas since launch.
    # This makes the system robust to restarts and lets us ingest in fixed-size batches.
    before = {p.resolve() for p in discover_unarchived_pgn_files(pgn_dir, archive_dir)}

    max_restarts = int(os.environ.get("SELFTRAIN_BOT_MAX_RESTARTS", "8") or 8)
    base_backoff = float(os.environ.get("SELFTRAIN_BOT_RESTART_BACKOFF", "20") or 20)
    backoff = base_backoff
    restarts = 0

    log_path = _bot_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[selftrain] lichess-bot log: {log_path}", flush=True)

    while True:
        remaining = max_seconds - int(time.time() - t0)
        if remaining <= 0:
            return 0

        # Append logs across restarts so we can tail for error signatures.
        with open(log_path, "a", encoding="utf-8", errors="replace") as logf:
            logf.write(f"\n--- selftrain: starting lichess-bot (restart={restarts}) ---\n")
            logf.flush()
            proc = subprocess.Popen(  # noqa: S603,S607
                cmd,
                cwd=str(lichess_bot_dir),
                env=env,
                stdout=logf,
                stderr=logf,
            )
            try:
                print(
                    f"[selftrain] lichess-bot started pid={proc.pid} cwd={lichess_bot_dir} "
                    f"(remaining_s~={remaining})",
                    flush=True,
                )
                last_heartbeat = time.time()
                while time.time() - t0 < max_seconds:
                    code = proc.poll()
                    if code is not None:
                        break
                    if max_new_pgns > 0:
                        now = {p.resolve() for p in discover_unarchived_pgn_files(pgn_dir, archive_dir)}
                        if len(now) >= max_new_pgns:
                            proc.send_signal(getattr(signal, "SIGTERM", 15))
                            try:
                                proc.wait(timeout=30)
                            except Exception:
                                proc.kill()
                            return int(proc.poll() or 0)
                    # Console heartbeat so it's obvious the bot is alive.
                    if time.time() - last_heartbeat >= 30:
                        last_heartbeat = time.time()
                        try:
                            now = {p.resolve() for p in discover_unarchived_pgn_files(pgn_dir, archive_dir)}
                            new_ct = len(now)
                            total_ct = len(discover_pgn_files(pgn_dir))
                        except Exception:
                            new_ct = -1
                            total_ct = -1
                        print(
                            f"[selftrain] running... pid={proc.pid} new_pgns={new_ct}/{max_new_pgns} total_pgns={total_ct} "
                            f"elapsed_s~={int(time.time() - t0)}",
                            flush=True,
                        )
                    time.sleep(1.0)

                # time budget hit: stop
                if proc.poll() is None:
                    proc.send_signal(getattr(signal, "SIGTERM", 15))
                    try:
                        proc.wait(timeout=30)
                    except Exception:
                        proc.kill()
                code = int(proc.poll() or 0)
            finally:
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        if code == 0:
            return 0

        # Decide whether to retry based on log tail.
        tail = _tail_text(log_path)
        if _is_token_error(tail):
            notify(
                "selftrain: lichess-bot token/auth error; will retry.\n"
                f"restart={restarts}/{max_restarts} backoff_s={int(backoff)}\n"
                "If this repeats, rotate the token and update selftrain.env/config.yml."
            )
        elif _is_transient_network_error(tail):
            notify(
                "selftrain: lichess-bot network/rate-limit issue; will retry.\n"
                f"restart={restarts}/{max_restarts} backoff_s={int(backoff)}"
            )
        else:
            # Unknown crash: don't loop forever.
            notify(
                "selftrain: lichess-bot exited unexpectedly; not retrying.\n"
                f"exit_code={code}\n"
                f"log_tail={tail[-800:].replace(chr(0), '')}"
            )
            return code

        restarts += 1
        if restarts > max_restarts:
            notify("selftrain: exceeded SELFTRAIN_BOT_MAX_RESTARTS; stopping.")
            return code

        time.sleep(backoff)
        backoff = min(backoff * 1.5, 5 * 60)


def _resolve_env_path() -> Path:
    p = os.environ.get("SELFTRAIN_ENV_PATH", "").strip()
    if p:
        return Path(p)
    if _SELFTRAIN_DEFAULT_ENV.is_file():
        return _SELFTRAIN_DEFAULT_ENV
    return _SELFTRAIN_FALLBACK_ENV


def _next_batch_dir(base: Path) -> Path:
    """
    Create a unique batch directory under `base`:
      base/YYYYMMDD-batch-001, -002, ...
    """
    base.mkdir(parents=True, exist_ok=True)
    date = time.strftime("%Y%m%d")
    n = 1
    while True:
        p = base / f"{date}-batch-{n:03d}"
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            return p
        n += 1


def main() -> int:
    # Load selftrain.env (preferred) or .env; override path with SELFTRAIN_ENV_PATH
    env_path = _resolve_env_path()
    if env_path.exists():
        print(f"[selftrain] loading env: {env_path}", flush=True)
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(env_path, override=False)
        except Exception:
            # Minimal manual parser: KEY=VAL lines
            for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and k not in os.environ:
                    os.environ[k] = v

    lichess_bot_dir = Path(
        _env_str("SELFTRAIN_LICHESS_BOT_DIR", str(V5_ROOT / "lichess-bot"))
    )
    pgn_dir = Path(_env_str("SELFTRAIN_PGN_DIR", str(lichess_bot_dir / "game_records")))
    archive_dir = Path(_env_str("SELFTRAIN_ARCHIVE_DIR", str(pgn_dir / "selftrain_archive")))
    _sync_lichess_env_for_bot()
    token = _resolve_lichess_token()
    if not token:
        print(
            "Set LICHESS_TOKEN to your Lichess OAuth token (same as lichess-bot). "
            "Legacy: LICHESS_BOT_TOKEN or LICHESS_API_TOKEN.",
            file=sys.stderr,
        )
        return 2
    bot_user = _lichess_bot_name()
    if not bot_user:
        print(
            "Set LICHESS_BOT_NAME to your Lichess bot username (not the token). "
            "Legacy: SELFTRAIN_BOT_USER.",
            file=sys.stderr,
        )
        return 2

    max_seconds = int(os.environ.get("SELFTRAIN_MAX_SECONDS", "21600"))  # 6h
    max_games = int(os.environ.get("SELFTRAIN_MAX_GAMES", "25"))
    # Total automated training games to ingest (does not include your "reserved" games, which
    # you handle by not running this script for those 25). Default matches the 75-total/25-reserved plan.
    # Cumulative cap across many runs, stored in `selftrain_state.json` (ingested game IDs).
    max_total_training = int(os.environ.get("SELFTRAIN_TRAINING_GAMES", "75"))
    # Lichess server analysis can take a bit; we re-fetch `/game/export/{id}` until `analysis[].judgment`
    # shows up, or we hit the total time budget. `POLL` is the sleep between attempts.
    analysis_wait = int(os.environ.get("SELFTRAIN_ANALYSIS_MAX_WAIT", "1200"))  # 20m
    analysis_poll = float(os.environ.get("SELFTRAIN_ANALYSIS_POLL", "30"))

    st = _load_state()
    # Daily reset so each noon run targets daily quota.
    today = _today_yyyymmdd()
    if _daily_reset_enabled() and st.day and st.day != today:
        _say(f"daily reset: {st.day} -> {today}")
        st.training_games_completed = 0
        st.ingested_ids = []
        st.runs_today = 0
    st.day = today
    st.runs_today = int(st.runs_today or 0) + 1
    _save_state(st)

    # Clear per-run experience-book usage log.
    exp_usage_log = DATA_DIR / "experience_usage.tsv"
    try:
        exp_usage_log.write_text("", encoding="utf-8")
    except Exception:
        pass
    os.environ["V4_EXPERIENCE_USAGE_LOG"] = str(exp_usage_log)

    # Truncate lichess-bot log to last 10k lines.
    try:
        auto_log = lichess_bot_dir / "lichess_bot_auto_logs" / "lichess-bot.log"
        _truncate_log_by_lines(auto_log, keep_last=10_000)
    except Exception:
        pass

    start_msg = (
        "selftrain: starting batch\n"
        f"bot={bot_user}\n"
        f"lichess_bot_dir={lichess_bot_dir}\n"
        f"max_seconds={max_seconds} max_games={max_games}\n"
        f"runs_today={st.runs_today} ingest=all_unarchived_each_pass\n"
        f"training_games={st.training_games_completed}/{max_total_training}"
    )
    print("[selftrain] " + start_msg.replace("\n", " | "), flush=True)
    notify_event("start", start_msg)

    # Child env: lichess-bot reads LICHESS_BOT_TOKEN (lib/config.py); keep LICHESS_TOKEN in sync.
    child_env = dict(os.environ)
    child_env["LICHESS_BOT_TOKEN"] = token
    child_env["LICHESS_TOKEN"] = token

    def _pp_add(pp: str, item: str) -> str:
        if not item:
            return pp
        parts = [p for p in pp.split(os.pathsep) if p]
        if item in parts:
            return pp
        return item + os.pathsep + pp if pp else item

    child_env["PYTHONPATH"] = _pp_add(str(child_env.get("PYTHONPATH", "")), str(V5_ROOT))

    # Repeat: drain *all* unarchived PGNs on each pass (startup or right after lichess-bot hits max_games),
    # counting only toward SELFTRAIN_TRAINING_GAMES when a draw/loss is fully processed. See data/selftrain_ingest.log.
    t_start = time.time()
    summary = RunSummary()
    post_bot = False
    while st.training_games_completed < max_total_training:
        unarchived = discover_unarchived_pgn_files(pgn_dir, archive_dir)

        if unarchived:
            phase = "post_bot" if post_bot else "startup"
            batch = sorted(unarchived, key=lambda x: x.stat().st_mtime)
            batch_dir = _next_batch_dir(archive_dir)
            _append_ingest_log(
                f"batch_begin phase={phase} n_pgns={len(batch)} "
                f"training={st.training_games_completed}/{max_total_training} "
                f"runs_today={st.runs_today} -> {batch_dir.name}"
            )
            _say(f"ingesting batch size={len(batch)} (all unarchived, phase={phase}) -> {batch_dir}")

            ingested = 0
            for p in batch:
                if st.training_games_completed >= max_total_training:
                    _append_ingest_log(
                        f"batch_stopped reason=training_cap "
                        f"training={st.training_games_completed}/{max_total_training}"
                    )
                    break
                gid = _extract_id_from_pgn_path(p)
                if not gid:
                    # Still archive it so we don't get stuck forever.
                    try:
                        shutil.move(str(p), str(batch_dir / p.name))
                    except Exception:
                        pass
                    continue
                if gid in set(st.ingested_ids or ()):
                    # Already ingested; archive it.
                    try:
                        shutil.move(str(p), str(batch_dir / p.name))
                    except Exception:
                        pass
                    continue

                # Only learn from draws/losses (your spec). Wins are archived without analysis wait.
                try:
                    g0 = export_game_json(gid, token=token)
                except Exception as e:
                    _say(f"export failed for {gid}: {e} (keeping record for later)")
                    continue
                outcome = _bot_outcome(g0, bot_user=bot_user)
                # Update summary stats
                if outcome == "win":
                    summary.wins += 1
                elif outcome == "draw":
                    summary.draws += 1
                elif outcome == "loss":
                    summary.losses += 1
                try:
                    is_w = _bot_color_from_json(g0, bot_user) is True
                    pl = (g0.get("players") or {}).get("white" if is_w else "black") or {}
                    summary.elo_delta += int(pl.get("ratingDiff") or 0)
                except Exception:
                    pass
                if _is_abort(g0):
                    _say(f"skipping abort {gid} (deleting PGN record)")
                    _append_ingest_log(f"skip gid={gid} reason=abort")
                    try:
                        p.unlink()
                    except Exception:
                        # fallback: try archive
                        try:
                            shutil.move(str(p), str(batch_dir / p.name))
                        except Exception:
                            pass
                    continue
                if outcome == "win":
                    if _delete_win_pgns():
                        _say(f"skipping win {gid} (deleting PGN record)")
                        _append_ingest_log(f"skip gid={gid} reason=win_delete_pgn")
                        try:
                            p.unlink()
                        except Exception:
                            # fallback: try move into batch for safety
                            try:
                                shutil.move(str(p), str(batch_dir / p.name))
                            except Exception:
                                pass
                    else:
                        _say(f"skipping win {gid} (archiving record only)")
                        _append_ingest_log(f"skip gid={gid} reason=win_archive")
                        try:
                            shutil.move(str(p), str(batch_dir / p.name))
                        except Exception:
                            pass
                    continue
                if outcome == "unknown":
                    _say(f"unknown outcome for {gid}; keeping record for later ({p.name})")
                    _append_ingest_log(f"skip gid={gid} reason=unknown_outcome")
                    continue

                n = 0
                # Primary: local Stockfish two-pass at depths 12/18 (configurable), so we don't stall waiting
                # for Lichess server analysis to appear.
                if _sf_enabled():
                    sug = _sf_analyse_game_two_pass(p, bot_user=bot_user)
                    if sug:
                        _add_engine_import_path()
                        from experience_book import _connect, add_entry, DB_PATH  # type: ignore

                        con = _connect(str(DB_PATH))
                        try:
                            for fen_before, best_uci, drop_cp in sug:
                                add_entry(con, fen_before, best_uci, depth=_sf_depth2(), score=None)
                                n += 1
                        finally:
                            con.close()
                        summary.exp_rows_added += n
                    else:
                        _say(f"local SF produced no suggestions for {gid}")

                # Fallback: use Lichess server judgments when present (or after waiting).
                # if n == 0:
                #     _say(f"waiting analysis for {gid} ({outcome}) (max_wait_s={analysis_wait} poll_s={analysis_poll})")
                #     g = wait_for_server_analysis(gid, token, max_wait_s=analysis_wait, sleep_s=analysis_poll)
                #     if not analysis_ready(g):
                #         _say(f"no server judgments yet for {gid}; keeping record for later ({p.name})")
                #         # Do not archive; keep it to retry later.
                #         continue
                #     n = ingest_experience_from_lichess_json(g, bot_user=bot_user, only_judgments=True)

                st.ingested_ids = list(st.ingested_ids or [])
                st.ingested_ids.append(gid)
                st.training_games_completed += 1
                ingested += 1
                _save_state(st)
                _append_ingest_log(
                    f"training_gid gid={gid} outcome={outcome} exp_rows={n} "
                    f"total={st.training_games_completed}/{max_total_training}"
                )
                _say("-----")
                _say(
                    f"ingested {gid} (+{n} experience entries); "
                    f"training_games={st.training_games_completed}/{max_total_training}; "
                    f"elapsed_s~={int(time.time() - t_start)}"
                )

                # Archive processed record.
                try:
                    shutil.move(str(p), str(batch_dir / p.name))
                except Exception:
                    pass

                if st.training_games_completed >= max_total_training:
                    notify("selftrain: reached SELFTRAIN_TRAINING_GAMES; stop.")
                    return 0

            _append_ingest_log(
                f"batch_end phase={phase} training_events_added={ingested} "
                f"training={st.training_games_completed}/{max_total_training}"
            )

            if ingested == 0:
                _say(
                    "batch finished; no games ingested (analysis may be missing). "
                    "Leaving un-analyzed PGNs in place to retry later."
                )

            time.sleep(5)
            continue

        # No pending PGNs — run lichess-bot until MAX_GAMES new files land (then loop ingests all).
        try:
            total_ct = len(discover_pgn_files(pgn_dir))
        except Exception:
            total_ct = -1
        _say(f"no unarchived PGNs yet (pgn_dir={pgn_dir} total_recursive_pgns={total_ct}); starting bot")
        try:
            tail = _tail_text(_bot_log_path(), max_bytes=32_000)
            if tail.strip():
                _say("lichess-bot log tail (last ~2000 chars):")
                _say(tail[-2000:].replace(chr(0), ""))
        except Exception:
            pass
        try:
            pgn_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        _append_ingest_log(
            f"bot_run_begin max_seconds={max_seconds} max_new_pgns={max_games} "
            f"training={st.training_games_completed}/{max_total_training}"
        )
        _run_bot(
            lichess_bot_dir,
            max_seconds=max_seconds,
            max_new_pgns=max_games,
            pgn_dir=pgn_dir,
            archive_dir=archive_dir,
            env=child_env,
        )
        _append_ingest_log(
            f"bot_run_end training={st.training_games_completed}/{max_total_training}"
        )
        post_bot = True
        time.sleep(30)

    # End-of-run summary to Discord/Telegram
    summary.exp_used = _read_exp_usage(exp_usage_log)
    used_hits = len(summary.exp_used)
    used_by_game: dict[str, list[str]] = {}
    for gid, keyhex, uci in summary.exp_used:
        gk = gid or "unknown"
        used_by_game.setdefault(gk, [])
        if len(used_by_game[gk]) < 8:
            used_by_game[gk].append(uci)
    used_lines = []
    for gid, moves in list(used_by_game.items())[:10]:
        used_lines.append(f"- {gid}: {', '.join(moves)}")
    msg = (
        "selftrain summary\n"
        f"W/D/L={summary.wins}/{summary.draws}/{summary.losses}  elo_delta={summary.elo_delta}\n"
        f"exp_book added_rows={summary.exp_rows_added}  exp_used_hits={used_hits}\n"
        + ("exp used (sample):\n" + "\n".join(used_lines) + "\n" if used_lines else "")
    )
    _say(msg)
    notify_event("end", msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
