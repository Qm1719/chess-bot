"""
main.py — Entry point for the Chess AI Assignment project.

Usage:
  python main.py
      Interactive menu: Play (GUI) | Lichess bot (no matchmaking) | Self-train
  python main.py gui | play
      Launch the GUI
  python main.py lichess | lichess-nomatch
      Run lichess-bot with matchmaking disabled (challenges in only)
  python main.py selftrain
      Run the autonomous self-train pipeline (see ops/selftrain.py)
  python main.py bench [level] [games]
  python main.py match [Lw] [Lb] [games]
"""

from __future__ import annotations

import os
import sys

_V5_ROOT = os.path.abspath(os.path.dirname(__file__))
_ENGINE = os.path.join(_V5_ROOT, "engine")
_DATA = os.path.join(_V5_ROOT, "data")
for _p in (_V5_ROOT, _ENGINE, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_v5_env() -> None:
    """Load `selftrain.env` then `.env` if present (later files only fill missing keys)."""
    for name in ("selftrain.env", ".env"):
        p = os.path.join(_V5_ROOT, name)
        if not os.path.isfile(p):
            continue
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(p, override=False)
        except Exception:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'").strip('"')
                    if k and k not in os.environ:
                        os.environ[k] = v


def run_gui() -> None:
    from engine.gui import main

    main()


def run_benchmark(level: int = 5, games: int = 10) -> None:
    from engine.agents_v4 import benchmark

    print(f"Benchmarking Level {level} vs Random over {games} games...\n")
    wins, draws, losses = benchmark(level=level, games=games, verbose=True)
    total = wins + draws + losses
    print(f"\nFinal: {wins}/{total} wins, {draws} draws, {losses} losses")
    win_pct = 100 * wins / total if total else 0
    print(f"Win rate: {win_pct:.1f}%")


def run_match(level_w: int, level_b: int, games: int = 10) -> None:
    from engine.agents_v4 import get_agent, run_match

    agent_w = get_agent(level_w)
    agent_b = get_agent(level_b)
    wins_w = wins_b = draws = 0
    print(f"Match: {agent_w.name} (White) vs {agent_b.name} (Black), {games} games\n")
    for i in range(games):
        result = run_match(agent_w, agent_b, verbose=False)
        if result == "white":
            wins_w += 1
        elif result == "black":
            wins_b += 1
        else:
            draws += 1
        print(f"  Game {i+1:2d}: {result}")
    print(f"\nWhite wins: {wins_w}  |  Black wins: {wins_b}  |  Draws: {draws}")


def _env_pp_prepend_v5(env: dict[str, str]) -> dict[str, str]:
    e = dict(env)
    pp = e.get("PYTHONPATH", "")
    e["PYTHONPATH"] = _V5_ROOT + (os.pathsep + pp if pp else "")
    tok = (
        (e.get("LICHESS_TOKEN") or "").strip()
        or (e.get("LICHESS_BOT_TOKEN") or "").strip()
        or (e.get("LICHESS_API_TOKEN") or "").strip()
    )
    if tok:
        e["LICHESS_TOKEN"] = tok
        e["LICHESS_BOT_TOKEN"] = tok
    return e


def run_lichess_bot_no_matchmaking() -> int:
    """Run lichess-bot with `matchmaking.allow_matchmaking: false` (no outgoing seeks)."""
    try:
        import yaml  # type: ignore
    except Exception:
        print("PyYAML is required. Install: pip install pyyaml", file=sys.stderr)
        return 2

    load_v5_env()
    lichess_dir = os.path.join(_V5_ROOT, "lichess-bot")
    cfg_path = os.path.join(lichess_dir, "config.yml")
    if not os.path.isfile(cfg_path):
        print(f"Missing {cfg_path}", file=sys.stderr)
        return 2

    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        print("config.yml: expected mapping", file=sys.stderr)
        return 2
    data.setdefault("matchmaking", {})["allow_matchmaking"] = False

    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yml",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        yaml.safe_dump(data, tmp, sort_keys=False, allow_unicode=True)
        tmp_path = tmp.name

    import subprocess

    env = _env_pp_prepend_v5(os.environ)
    try:
        p = subprocess.run(
            [sys.executable, "-u", "lichess-bot.py", "--config", tmp_path],
            cwd=lichess_dir,
            env=env,
        )
        return int(p.returncode)
    except OSError as e:
        print(f"Failed to start lichess-bot: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def run_selftrain() -> int:
    load_v5_env()
    from ops.selftrain import main as selftrain_main

    return int(selftrain_main())


def _interactive_menu() -> None:
    load_v5_env()
    print(
        "1) Play (local GUI)\n"
        "2) Lichess bot — listen only (no matchmaking; incoming challenges / games OK)\n"
        "3) Self-train — run bot + pull Lichess analysis → experience book (see ops/selftrain.py)\n"
    )
    c = input("Choice [1-3] (q=quit): ").strip().lower()
    if c in ("1", "p", "play", "gui"):
        run_gui()
    elif c in ("2", "l", "lichess", "bot"):
        raise SystemExit(run_lichess_bot_no_matchmaking())
    elif c in ("3", "s", "selftrain", "train"):
        raise SystemExit(run_selftrain())
    elif c in ("q", "quit", ""):
        return
    else:
        print("Unknown choice.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]]
    if not args or args[0] in ("menu", "i", "interactive"):
        _interactive_menu()
    elif args[0] in ("gui", "play", "1"):
        run_gui()
    elif args[0] in ("lichess", "lichess-nomatch", "bot", "2"):
        raise SystemExit(run_lichess_bot_no_matchmaking())
    elif args[0] in ("selftrain", "train", "3"):
        raise SystemExit(run_selftrain())
    elif args[0] == "bench":
        level = int(args[1]) if len(args) > 1 else 5
        games = int(args[2]) if len(args) > 2 else 10
        run_benchmark(level, games)
    elif args[0] == "match":
        lw = int(args[1]) if len(args) > 1 else 3
        lb = int(args[2]) if len(args) > 2 else 7
        g = int(args[3]) if len(args) > 3 else 10
        run_match(lw, lb, g)
    else:
        print(__doc__)
        raise SystemExit(2)
