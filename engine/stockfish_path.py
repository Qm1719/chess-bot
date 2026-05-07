from __future__ import annotations

import os
import platform
from pathlib import Path


def _v5_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_exe(p: Path) -> bool:
    try:
        return p.is_file()
    except Exception:
        return False


def _pick_newest(paths: list[Path]) -> str | None:
    ok = [p for p in paths if _is_exe(p)]
    if not ok:
        return None
    try:
        ok.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        ok.sort(key=lambda p: str(p))
    return str(ok[0])


def detect_stockfish_path() -> str:
    """
    Detect a Stockfish UCI executable path.

    Priority:
      1) STOCKFISH_PATH
      2) SELFTRAIN_STOCKFISH_PATH (back-compat for selftrain)
      3) submission v5/engine/stockfish/ (your new layout; OS-specific preference)
      4) repo_root/stockfish/ (older layout)
      5) "stockfish" (PATH)
    """
    v = (os.environ.get("STOCKFISH_PATH", "") or "").strip()
    if v:
        return v
    v = (os.environ.get("SELFTRAIN_STOCKFISH_PATH", "") or "").strip()
    if v:
        return v

    v5 = _v5_root()
    engine_stockfish = v5 / "engine" / "stockfish"

    sysname = platform.system().lower()
    is_windows = sysname.startswith("win")
    is_linux = sysname.startswith("linux")

    # Your intended layout: put one Windows exe and one Ubuntu/Linux binary here.
    if engine_stockfish.is_dir():
        if is_windows:
            cand = _pick_newest(list(engine_stockfish.glob("*.exe")))
            if cand:
                return cand
        if is_linux:
            # Common names: stockfish, stockfish-ubuntu, stockfish-linux-x86-64-...
            cand = _pick_newest([p for p in engine_stockfish.glob("*") if p.is_file() and p.suffix == ""])
            if cand:
                return cand
        # Fallback: any file in the folder
        cand = _pick_newest([p for p in engine_stockfish.glob("*") if p.is_file()])
        if cand:
            return cand

    # Older layout used in this repo: repo_root/stockfish/*.exe
    try:
        repo_root = v5.parent
        cand_dir = repo_root / "stockfish"
        if cand_dir.is_dir():
            if is_windows:
                cand = _pick_newest(list(cand_dir.glob("*.exe")))
                if cand:
                    return cand
            cand = _pick_newest([p for p in cand_dir.glob("*") if p.is_file()])
            if cand:
                return cand
    except Exception:
        pass

    return "stockfish"

