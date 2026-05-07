"""nnue_infer.py — numpy inference for trained NNUE weights (optional, lives in v4)."""

from __future__ import annotations

import os

import numpy as np

from nnue_features import Position, encode_nnue

_WEIGHTS_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "nnue", "nnue_weights.npz"))
_state: dict | bool | None = None


def _load() -> None:
    global _state
    if _state is not None:
        return
    if not os.path.isfile(_WEIGHTS_PATH):
        _state = False
        return
    _state = dict(np.load(_WEIGHTS_PATH))


def enabled() -> bool:
    if os.environ.get("NNUE_OFF", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    _load()
    return isinstance(_state, dict)


def blend() -> float:
    """Mix factor: 0 = static eval only, 1 = pure NNUE term (still blended formula in agents)."""
    _load()
    if not isinstance(_state, dict):
        return 0.0
    b = _state.get("blend")
    if b is None:
        return 0.5
    return float(np.asarray(b).reshape(-1)[0])


def kind() -> str:
    """'cp' for absolute centipawns, 'delta' for correction to static eval."""
    _load()
    if not isinstance(_state, dict):
        return "cp"
    k = _state.get("kind")
    if k is None:
        return "cp"
    try:
        s = str(np.asarray(k).reshape(-1)[0])
    except Exception:
        return "cp"
    return "delta" if "delta" in s.lower() else "cp"


def score_white(pos: Position) -> int:
    """
    White-centric score in centipawns from the NNUE head (matches v3 evaluate() scale).
    """
    _load()
    if not isinstance(_state, dict):
        return 0
    W1 = _state["W1"]
    b1 = _state["b1"]
    W2 = _state["W2"]
    b2 = _state["b2"]
    W3 = _state["W3"]
    b3 = _state["b3"]
    scale = float(np.asarray(_state.get("out_scale", np.array([100.0]))).reshape(-1)[0])

    x = encode_nnue(pos)
    h = np.maximum(0.0, x @ W1 + b1)
    h2 = np.maximum(0.0, h @ W2 + b2)
    y = float(np.dot(h2, W3.ravel()) + float(np.asarray(b3).reshape(-1)[0]))
    return int(y * scale)
