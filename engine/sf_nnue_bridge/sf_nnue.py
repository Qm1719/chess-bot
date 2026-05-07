from __future__ import annotations

import ctypes
import os
import sys
from ctypes import c_char_p, c_int, c_uint64, POINTER


_HERE = os.path.abspath(os.path.dirname(__file__))
_LIB_PATH = os.path.join(_HERE, "sf_nnue_bridge.dll")
if os.name != "nt":
    # Linux/macOS build uses a shared object.
    _LIB_PATH = os.path.join(_HERE, "sf_nnue_bridge.so")

# When the environment variable is unset, this default is used. Set SF_NNUE=0 in the
# environment to force-disable without editing this file.
_DEFAULT_SF_NNUE = "1"

# Make sure v5 engine modules are importable when running standalone.
_V5_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ENGINE_DIR = os.path.join(_V5_ROOT, "engine")
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)


class StockfishNNUEBridge:
    def __init__(self, dll_path: str | None = None):
        self._dll_path = dll_path or _LIB_PATH
        self._dll = None

    def load(self) -> None:
        if self._dll is not None:
            return
        if not os.path.isfile(self._dll_path):
            raise FileNotFoundError(self._dll_path)
        # MinGW exports use cdecl by default; use CDLL (cdecl), not WinDLL (stdcall).
        self._dll = ctypes.CDLL(self._dll_path)

        self._dll.sf_nnue_init.argtypes = [c_char_p]
        self._dll.sf_nnue_init.restype = c_int

        self._dll.sf_nnue_eval_cp.argtypes = [POINTER(c_uint64), c_int]
        self._dll.sf_nnue_eval_cp.restype = c_int

        self._dll.sf_nnue_eval_cp_fen.argtypes = [c_char_p]
        self._dll.sf_nnue_eval_cp_fen.restype = c_int

        self._dll.sf_nnue_close.argtypes = []
        self._dll.sf_nnue_close.restype = None

    def init(self, evalfile_path: str) -> None:
        self.load()
        rc = int(self._dll.sf_nnue_init(evalfile_path.encode("utf-8")))
        if rc != 0:
            raise RuntimeError(f"sf_nnue_init failed rc={rc}")

    def eval_white_cp_from_fen(self, fen: str) -> int:
        """Full FEN (castling, ep, clocks) — matches Stockfish's Position; preferred over bitboards."""
        self.load()
        return int(self._dll.sf_nnue_eval_cp_fen(fen.encode("utf-8")))

    def eval_white_cp_from_bitboards(self, bb12: list[int] | tuple[int, ...], side_to_move: int) -> int:
        """
        bb12: 12 uint64 bitboards in python-engine piece order:
          0..5 white P N B R Q K, 6..11 black p n b r q k
        side_to_move: 0=white, 1=black
        returns: centipawns from white's perspective (positive = white better)
        """
        self.load()
        if len(bb12) != 12:
            raise ValueError("bb12 must have length 12")
        arr_t = c_uint64 * 12
        arr = arr_t(*[int(x) & 0xFFFFFFFFFFFFFFFF for x in bb12])
        return int(self._dll.sf_nnue_eval_cp(arr, int(side_to_move)))

    def close(self) -> None:
        if self._dll is None:
            return
        try:
            self._dll.sf_nnue_close()
        finally:
            self._dll = None


_bridge: StockfishNNUEBridge | None = None


def bridge() -> StockfishNNUEBridge:
    global _bridge
    if _bridge is None:
        _bridge = StockfishNNUEBridge()
    return _bridge


def enabled() -> bool:
    v = os.environ.get("SF_NNUE", _DEFAULT_SF_NNUE).strip()
    return v not in ("", "0", "false", "False")


def eval_white_cp_from_pos(pos, evalfile_path: str) -> int:
    b = bridge()
    b.init(evalfile_path)
    # Use engine FEN so castling rights and en passant match Stockfish (bitboard-only FEN is wrong).
    fen = pos.to_fen()
    return b.eval_white_cp_from_fen(fen)


def smoke_test(evalfile_path: str) -> None:
    # Late import to avoid path issues if called standalone
    from position import Position  # type: ignore

    p = Position.from_fen(Position.START_FEN)
    cp = eval_white_cp_from_pos(p, evalfile_path)
    print("startpos cp (white):", cp)

