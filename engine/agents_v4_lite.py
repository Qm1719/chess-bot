"""
agents_v4_lite.py

Lite wrapper around `agents_v4.py` that forces NNUE off.
Used by some lichess-bot bridge variants.
"""

from __future__ import annotations

import os

# Force-disable NNUE in this process.
os.environ["NNUE_OFF"] = "1"
os.environ["SF_NNUE_BLEND"] = "0"

from agents_v4 import get_agent  # noqa: E402,F401

