"""Syzygy probing for the bundled 3–4–5 piece set (see repo 3-4-5_pieces_Syzygy/)."""

from __future__ import annotations

import os
from typing import Optional

try:
    import chess  # type: ignore
    import chess.syzygy  # type: ignore
except Exception:  # pragma: no cover
    chess = None


def _candidate_tb_roots() -> list[str]:
    here = os.path.abspath(os.path.dirname(__file__))
    v5_root = os.path.abspath(os.path.join(here, ".."))
    return [
        # Preferred: put TB packs under submission v5/data/syzygy/...
        os.path.join(v5_root, "data", "syzygy"),
        os.path.join(v5_root, "data", "3-4-5_pieces_Syzygy", "3-4-5"),
        # Back-compat fallbacks (if someone places TBs next to engine folder)
        os.path.join(here, "..", "3-4-5_pieces_Syzygy", "3-4-5"),
        os.path.join(here, "..", "..", "3-4-5_pieces_Syzygy", "3-4-5"),
    ]


def _dir_has_syzygy_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        for name in os.listdir(path):
            low = name.lower()
            if low.endswith(".rtbw") or low.endswith(".rtbz"):
                return True
    except OSError:
        return False
    return False


def _find_syzygy_leaf_dir(root: str) -> Optional[str]:
    """
    Syzygy packs are sometimes organized with subdirectories (e.g. WDL/DTZ splits).
    Accept either:
      - a directory that directly contains *.rtbw/*.rtbz, OR
      - a directory whose immediate children contain those files.
    """
    if _dir_has_syzygy_files(root):
        return root
    try:
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p) and _dir_has_syzygy_files(p):
                return p
    except OSError:
        return None
    return None


def find_syzygy_path() -> Optional[str]:
    for key in ("SYZYGY_PATH", "SYZYGY"):
        path = os.environ.get(key, "").strip()
        if path:
            leaf = _find_syzygy_leaf_dir(path)
            if leaf:
                return leaf
    for path in _candidate_tb_roots():
        resolved = os.path.normpath(os.path.abspath(path))
        leaf = _find_syzygy_leaf_dir(resolved)
        if leaf:
            return leaf
    return None


def max_syzygy_pieces() -> int:
    try:
        return max(3, min(7, int(os.environ.get("SYZYGY_MAX_PIECES", "5"))))
    except ValueError:
        return 5


def _tiebreak_tuple(board: "chess.Board", mv: "chess.Move") -> tuple[int, int, str]:
    """
    When WDL and DTZ keys tie, Syzygy won't tell us which move converts faster (no DTM).
    Prefer practical conversion:
      1) promotions (especially to queen),
      2) pawn pushes toward promotion,
      3) checks,
      4) other pieces (stable UCI last).
    """
    uci = mv.uci()
    p = board.piece_at(mv.from_square)
    if p is None:
        return (99, 0, uci)

    # Promotions first (python-chess: mv.promotion is piece type int).
    if mv.promotion is not None:
        # Prefer queen promotion, then rook, then bishop/knight.
        promo_rank = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 2}.get(mv.promotion, 3)
        return (0, promo_rank, uci)

    # Pawn pushes toward promotion.
    color = board.turn
    if p.piece_type == chess.PAWN:
        fr = chess.square_rank(mv.from_square)
        tr = chess.square_rank(mv.to_square)
        adv = (tr - fr) if color == chess.WHITE else (fr - tr)
        # print(f"adv: {adv} uci: {uci}")
        if adv > 0:
            return (1, -adv, uci)
        return (2, 0, uci)
    
    # Checks next.
    gives_check = False
    try:
        board.push(mv)
        gives_check = board.is_check()
    finally:
        try:
            board.pop()
        except Exception:
            pass
    if gives_check:
        return (3, 0, uci)

    # Otherwise: stable ordering by piece class.
    if p.piece_type == chess.ROOK:
        return (6, 0, uci)
    if p.piece_type == chess.KING:
        return (7, 0, uci)
    if p.piece_type == chess.QUEEN:
        return (4, 0, uci)
    if p.piece_type == chess.BISHOP:
        return (5, 0, uci)
    if p.piece_type == chess.KNIGHT:
        return (5, 0, uci)
    return (8, 0, uci)


def probe_bestmove(fen: str) -> Optional[str]:
    """
    Return best move UCI from Syzygy if available, else None.
    Uses WDL probing. Only positions with at most max_syzygy_pieces() (default 5)
    are probed, matching the 3–4–5 tablebase scope.
    """
    if chess is None:
        return None
    path = find_syzygy_path()
    if not path:
        return None
    board = chess.Board(fen)
    if board.castling_rights:
        return None
    pawnless = not any(p.piece_type == chess.PAWN for p in board.piece_map().values())
    n = len(board.piece_map())
    if n > max_syzygy_pieces():
        return None
    with chess.syzygy.open_tablebase(path) as tb:
        best: chess.Move | None = None
        best_wdl: int | None = None
        best_dtz_key: int | None = None

        for mv in board.legal_moves:
            board.push(mv)
            try:
                # Syzygy probes are from the perspective of the side to move.
                # After pushing a move, side-to-move flips, so invert to score
                # the move from the current side's perspective.
                wdl_cur = -tb.probe_wdl(board)
                try:
                    dtz_cur = -tb.probe_dtz(board)
                except Exception:
                    dtz_cur = 0
            except Exception:
                board.pop()
                return None
            finally:
                board.pop()

            # Prioritize WDL. In pawnless endings, also use DTZ to guide mating/conversion
            # (there are no "zeroing moves" except mate, so DTZ provides the needed gradient).
            w = int(wdl_cur)
            if pawnless and w != 0:
                d = int(dtz_cur)
                dtz_key = (-d) if w > 0 else d
            else:
                dtz_key = 0
            if best_wdl is None or w > best_wdl:
                best_wdl = w
                best_dtz_key = dtz_key
                best = mv
            elif best is not None and w == best_wdl:
                if pawnless and best_dtz_key is not None and dtz_key != best_dtz_key:
                    if dtz_key > best_dtz_key:
                        best_dtz_key = dtz_key
                        best = mv
                else:
                    # Deterministic "practical" tie-break.
                    if _tiebreak_tuple(board, mv) < _tiebreak_tuple(board, best):
                        best = mv
                        # print(f"best: {best.uci()}")

        return best.uci() if best else None


def probe_bestmoves(fen: str) -> Optional[list[str]]:
    """
    Return the list of best moves (UCI).

    Policy:
      - Always prioritize WDL.
      - If the position is pawnless (no pawns on the board) and WDL is decisive,
        also use DTZ as a tiebreak. This avoids "winning but aimless" checking loops
        in KQK / KRK etc where DTZ provides the needed conversion gradient.

    This is intentionally *less opinionated* than probe_bestmove(): if multiple moves
    are equally best by tablebase criteria, we return them all so the engine can pick
    using its own eval/search/preferences.
    """
    if chess is None:
        return None
    path = find_syzygy_path()
    if not path:
        return None
    board = chess.Board(fen)
    if board.castling_rights:
        return None
    pawnless = not any(p.piece_type == chess.PAWN for p in board.piece_map().values())
    n = len(board.piece_map())
    if n > max_syzygy_pieces():
        return None

    best_wdl: int | None = None
    best_moves: list[chess.Move] = []
    best_dtz_key: int | None = None

    with chess.syzygy.open_tablebase(path) as tb:
        for mv in board.legal_moves:
            board.push(mv)
            try:
                wdl_cur = -tb.probe_wdl(board)
                try:
                    dtz_cur = -tb.probe_dtz(board)
                except Exception:
                    dtz_cur = 0
            except Exception:
                board.pop()
                return None
            finally:
                board.pop()

            w = int(wdl_cur)
            if pawnless and w != 0:
                d = int(dtz_cur)
                dtz_key = (-d) if w > 0 else d
            else:
                dtz_key = 0
            if best_wdl is None or w > best_wdl:
                best_wdl = w
                best_dtz_key = dtz_key
                best_moves = [mv]
            elif w == best_wdl:
                if pawnless and best_dtz_key is not None and dtz_key != best_dtz_key:
                    if dtz_key > best_dtz_key:
                        best_dtz_key = dtz_key
                        best_moves = [mv]
                    # else: worse DTZ, skip
                else:
                    best_moves.append(mv)

    # Stable deterministic order (so the engine gets a consistent list).
    out = sorted({m.uci() for m in best_moves})
    return out


def probe_info(fen: str) -> Optional[dict[str, object]]:
    """
    Return a small info dict from Syzygy if available, else None.

    Keys:
      - path: tablebase path used
      - pieces: number of pieces in position
      - wdl: WDL for side-to-move (-2..2, where >0 is winning)
      - bestmove: best move UCI (or None)
      - best_wdl: WDL of best move (same scale)
      - best_dtz: DTZ of best move (signed, 0 if unavailable)
    """
    if chess is None:
        return None
    path = find_syzygy_path()
    if not path:
        return None
    board = chess.Board(fen)
    if board.castling_rights:
        return None
    pawnless = not any(p.piece_type == chess.PAWN for p in board.piece_map().values())
    n = len(board.piece_map())
    if n > max_syzygy_pieces():
        return None

    with chess.syzygy.open_tablebase(path) as tb:
        try:
            wdl = int(tb.probe_wdl(board))
        except Exception:
            return None

        best: chess.Move | None = None
        best_wdl: int | None = None
        best_dtz = 0
        best_dtz_key: int | None = None

        for mv in board.legal_moves:
            board.push(mv)
            try:
                wdl_cur = -tb.probe_wdl(board)
                try:
                    dtz_cur = -tb.probe_dtz(board)
                except Exception:
                    dtz_cur = 0
            except Exception:
                board.pop()
                return None
            finally:
                board.pop()

            w = int(wdl_cur)
            if pawnless and w != 0:
                d = int(dtz_cur)
                dtz_key = (-d) if w > 0 else d
            else:
                dtz_key = 0
            if best_wdl is None or w > best_wdl:
                best = mv
                best_wdl = w
                best_dtz = int(dtz_cur)
                best_dtz_key = dtz_key
            elif best is not None and w == best_wdl:
                if pawnless and best_dtz_key is not None and dtz_key != best_dtz_key:
                    if dtz_key > best_dtz_key:
                        best = mv
                        best_wdl = w
                        best_dtz = int(dtz_cur)
                        best_dtz_key = dtz_key
                else:
                    if _tiebreak_tuple(board, mv) < _tiebreak_tuple(board, best):
                        best = mv
                        best_wdl = w
                        best_dtz = int(dtz_cur)

        return {
            "path": path,
            "pieces": n,
            "wdl": wdl,
            "bestmove": best.uci() if best else None,
            "best_wdl": int(best_wdl or 0),
            "best_dtz": best_dtz,
        }


def probe_move_list(fen: str) -> Optional[list[dict[str, object]]]:
    """
    Return per-legal-move Syzygy outcomes for side-to-move, or None if unavailable.

    Each entry contains:
      - uci: move in UCI
      - wdl: WDL for side-to-move after making this move (-2..2)
      - dtz: DTZ for side-to-move after making this move (signed; 0 if unavailable)

    Sorted best-to-worst prioritizing WDL. In pawnless decisive endings, DTZ is used as
    a secondary key; otherwise DTZ is ignored. Finally a practical tiebreak.
    """
    if chess is None:
        return None
    path = find_syzygy_path()
    if not path:
        return None
    board = chess.Board(fen)
    if board.castling_rights:
        return None
    pawnless = not any(p.piece_type == chess.PAWN for p in board.piece_map().values())
    n = len(board.piece_map())
    if n > max_syzygy_pieces():
        return None

    out: list[dict[str, object]] = []
    with chess.syzygy.open_tablebase(path) as tb:
        for mv in board.legal_moves:
            board.push(mv)
            try:
                wdl_cur = -tb.probe_wdl(board)
                try:
                    dtz_cur = -tb.probe_dtz(board)
                except Exception:
                    dtz_cur = 0
            except Exception:
                board.pop()
                return None
            finally:
                board.pop()

            out.append({"uci": mv.uci(), "wdl": int(wdl_cur), "dtz": int(dtz_cur)})

    def _sort_key(d: dict[str, object]) -> tuple[int, int, tuple[int, int, str]]:
        uci = str(d.get("uci") or "")
        wdl_cur = int(d.get("wdl") or 0)
        dtz_cur = int(d.get("dtz") or 0)
        if pawnless and wdl_cur != 0:
            dtz_key = (-dtz_cur) if wdl_cur > 0 else dtz_cur
        else:
            dtz_key = 0
        # Higher WDL is better; then DTZ (pawnless decisive only); then tiebreak uses the move itself.
        # We compute the tiebreak tuple using the root board.
        mv = chess.Move.from_uci(uci) if uci else None
        tbk = (99, 0, uci) if mv is None else _tiebreak_tuple(board, mv)
        return (wdl_cur, dtz_key, tbk)

    out.sort(key=_sort_key, reverse=True)
    return out
