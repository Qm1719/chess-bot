"""
position.py — bitboard-based position, legal move generation, and push/pop.

This is the core for submission v3.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from bitboard import (
    bb,
    iter_bits,
    file_of,
    rank_of,
    KNIGHT_ATTACKS,
    KING_ATTACKS,
    PAWN_ATTACKS,
    sliding_attacks,
    DIAG_DIRS,
    ORTHO_DIRS,
)


# Piece indices: 0..5 white, 6..11 black
WP, WN, WB, WR, WQ, WK = range(6)
BP, BN, BB, BR, BQ, BK = range(6, 12)

WHITE, BLACK = 0, 1

PIECE_FROM_CHAR = {
    "P": WP, "N": WN, "B": WB, "R": WR, "Q": WQ, "K": WK,
    "p": BP, "n": BN, "b": BB, "r": BR, "q": BQ, "k": BK,
}
CHAR_FROM_PIECE = {v: k for k, v in PIECE_FROM_CHAR.items()}


def color_of(piece: int) -> int:
    return WHITE if piece < 6 else BLACK


def type_of(piece: int) -> int:
    return piece % 6


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


_Z = random.Random(0xBADC0FFE)
_Z_PIECE = [[_Z.getrandbits(64) for _ in range(64)] for __ in range(12)]
_Z_TURN = _Z.getrandbits(64)
_Z_CASTLE = [_Z.getrandbits(64) for _ in range(16)]  # castle rights as 4-bit mask
_Z_EPFILE = [_Z.getrandbits(64) for _ in range(9)]   # 0..7 file, 8 = none


@dataclass(frozen=True)
class Move:
    from_sq: int
    to_sq: int
    promo: int | None = None  # piece type 0..5 (P,N,B,R,Q,K) but only N/B/R/Q used
    is_ep: bool = False
    is_castle: bool = False

    def uci(self) -> str:
        def name(sq: int) -> str:
            return "abcdefgh"[file_of(sq)] + str(rank_of(sq) + 1)

        s = name(self.from_sq) + name(self.to_sq)
        if self.promo is not None:
            s += "nbrq"[{1: 0, 2: 1, 3: 2, 4: 3}.get(self.promo, 3)]
        return s


@dataclass
class Undo:
    mv: Move
    moved_piece: int
    captured: int | None
    captured_sq: int | None
    ep_sq: int | None
    castle: int
    halfmove: int
    fullmove: int
    key: int
    rook_move: tuple[int, int, int] | None  # (rook_piece, from, to)


class Position:
    START_FEN = START_FEN

    __slots__ = (
        "bb", "occ_w", "occ_b", "occ",
        "side", "castle", "ep_sq", "halfmove", "fullmove",
        "board", "key"
    )

    def __init__(self):
        self.bb = [0] * 12
        self.occ_w = 0
        self.occ_b = 0
        self.occ = 0
        self.side = WHITE
        self.castle = 0  # bits: 1=K,2=Q,4=k,8=q
        self.ep_sq: int | None = None
        self.halfmove = 0
        self.fullmove = 1
        self.board = [-1] * 64  # mailbox piece idx or -1
        self.key = 0

    @classmethod
    def from_fen(cls, fen: str) -> "Position":
        p = cls()
        parts = fen.split()
        rows = parts[0].split("/")
        sq = 56
        for row in rows:
            f = 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                else:
                    piece = PIECE_FROM_CHAR[ch]
                    s = sq + f
                    p._put(piece, s)
                    f += 1
            sq -= 8

        p.side = WHITE if parts[1] == "w" else BLACK
        p.castle = 0
        if "K" in parts[2]: p.castle |= 1
        if "Q" in parts[2]: p.castle |= 2
        if "k" in parts[2]: p.castle |= 4
        if "q" in parts[2]: p.castle |= 8
        p.ep_sq = None if parts[3] == "-" else (("abcdefgh".index(parts[3][0])) + (int(parts[3][1]) - 1) * 8)
        p.halfmove = int(parts[4]) if len(parts) > 4 else 0
        p.fullmove = int(parts[5]) if len(parts) > 5 else 1
        p._recalc_occ()
        p.key = p._compute_key()
        return p

    def to_fen(self) -> str:
        rows = []
        for r in range(7, -1, -1):
            empty = 0
            s = ""
            for c in range(8):
                sq = r * 8 + c
                p = self.board[sq]
                if p < 0:
                    empty += 1
                else:
                    if empty:
                        s += str(empty)
                        empty = 0
                    s += CHAR_FROM_PIECE[p]
            if empty:
                s += str(empty)
            rows.append(s)
        placement = "/".join(rows)
        stm = "w" if self.side == WHITE else "b"
        castle = ""
        if self.castle & 1: castle += "K"
        if self.castle & 2: castle += "Q"
        if self.castle & 4: castle += "k"
        if self.castle & 8: castle += "q"
        if not castle:
            castle = "-"
        ep = "-"
        if self.ep_sq is not None:
            ep = "abcdefgh"[file_of(self.ep_sq)] + str(rank_of(self.ep_sq) + 1)
        return f"{placement} {stm} {castle} {ep} {self.halfmove} {self.fullmove}"

    def _recalc_occ(self) -> None:
        self.occ_w = 0
        self.occ_b = 0
        for i in range(6):
            self.occ_w |= self.bb[i]
            self.occ_b |= self.bb[i + 6]
        self.occ = self.occ_w | self.occ_b

    def _compute_key(self) -> int:
        k = 0
        for piece in range(12):
            x = self.bb[piece]
            for s in iter_bits(x):
                k ^= _Z_PIECE[piece][s]
        if self.side == BLACK:
            k ^= _Z_TURN
        k ^= _Z_CASTLE[self.castle]
        k ^= _Z_EPFILE[8 if self.ep_sq is None else file_of(self.ep_sq)]
        return k

    def _put(self, piece: int, sq: int) -> None:
        self.bb[piece] |= bb(sq)
        self.board[sq] = piece

    def _remove(self, piece: int, sq: int) -> None:
        self.bb[piece] ^= bb(sq)
        self.board[sq] = -1

    def _move_piece(self, piece: int, fr: int, to: int) -> None:
        self.bb[piece] ^= bb(fr) | bb(to)
        self.board[fr] = -1
        self.board[to] = piece

    def king_sq(self, side: int) -> int:
        k = self.bb[WK if side == WHITE else BK]
        return (k & -k).bit_length() - 1

    def attackers_to(self, sq: int, by_side: int) -> int:
        occ = self.occ
        att = 0
        if by_side == WHITE:
            # pawns that attack sq are black-style from sq (reverse)
            for s in iter_bits(PAWN_ATTACKS[BLACK][sq] & self.bb[WP]):
                att |= bb(s)
        else:
            for s in iter_bits(PAWN_ATTACKS[WHITE][sq] & self.bb[BP]):
                att |= bb(s)

        pieces = self.bb
        att |= KNIGHT_ATTACKS[sq] & (pieces[WN] if by_side == WHITE else pieces[BN])
        att |= KING_ATTACKS[sq] & (pieces[WK] if by_side == WHITE else pieces[BK])
        bishops = (pieces[WB] | pieces[WQ]) if by_side == WHITE else (pieces[BB] | pieces[BQ])
        rooks = (pieces[WR] | pieces[WQ]) if by_side == WHITE else (pieces[BR] | pieces[BQ])
        att |= sliding_attacks(sq, occ, DIAG_DIRS) & bishops
        att |= sliding_attacks(sq, occ, ORTHO_DIRS) & rooks
        return att

    def in_check(self, side: int | None = None) -> bool:
        if side is None:
            side = self.side
        ksq = self.king_sq(side)
        return self.attackers_to(ksq, by_side=(BLACK if side == WHITE else WHITE)) != 0

    def _pseudo_moves(self):
        side = self.side
        us_occ = self.occ_w if side == WHITE else self.occ_b
        them_occ = self.occ_b if side == WHITE else self.occ_w
        occ = self.occ

        # Pawns
        if side == WHITE:
            pawns = self.bb[WP]
            for fr in iter_bits(pawns):
                r = rank_of(fr)
                one = fr + 8
                if one < 64 and self.board[one] == -1:
                    if r == 6:
                        for promo in (1, 2, 3, 4):
                            yield Move(fr, one, promo=promo)
                    else:
                        yield Move(fr, one)
                        if r == 1:
                            two = fr + 16
                            if self.board[two] == -1:
                                yield Move(fr, two)
                caps = PAWN_ATTACKS[WHITE][fr] & them_occ
                for to in iter_bits(caps):
                    if r == 6:
                        for promo in (1, 2, 3, 4):
                            yield Move(fr, to, promo=promo)
                    else:
                        yield Move(fr, to)
                if self.ep_sq is not None and (PAWN_ATTACKS[WHITE][fr] & bb(self.ep_sq)):
                    yield Move(fr, self.ep_sq, is_ep=True)
        else:
            pawns = self.bb[BP]
            for fr in iter_bits(pawns):
                r = rank_of(fr)
                one = fr - 8
                if one >= 0 and self.board[one] == -1:
                    if r == 1:
                        for promo in (1, 2, 3, 4):
                            yield Move(fr, one, promo=promo)
                    else:
                        yield Move(fr, one)
                        if r == 6:
                            two = fr - 16
                            if self.board[two] == -1:
                                yield Move(fr, two)
                caps = PAWN_ATTACKS[BLACK][fr] & them_occ
                for to in iter_bits(caps):
                    if r == 1:
                        for promo in (1, 2, 3, 4):
                            yield Move(fr, to, promo=promo)
                    else:
                        yield Move(fr, to)
                if self.ep_sq is not None and (PAWN_ATTACKS[BLACK][fr] & bb(self.ep_sq)):
                    yield Move(fr, self.ep_sq, is_ep=True)

        # Knights
        knights = self.bb[WN if side == WHITE else BN]
        for fr in iter_bits(knights):
            targets = KNIGHT_ATTACKS[fr] & ~us_occ
            for to in iter_bits(targets):
                yield Move(fr, to)

        # Bishops/Rooks/Queens
        bishops = self.bb[WB if side == WHITE else BB]
        for fr in iter_bits(bishops):
            targets = sliding_attacks(fr, occ, DIAG_DIRS) & ~us_occ
            for to in iter_bits(targets):
                yield Move(fr, to)
        rooks = self.bb[WR if side == WHITE else BR]
        for fr in iter_bits(rooks):
            targets = sliding_attacks(fr, occ, ORTHO_DIRS) & ~us_occ
            for to in iter_bits(targets):
                yield Move(fr, to)
        queens = self.bb[WQ if side == WHITE else BQ]
        for fr in iter_bits(queens):
            targets = (sliding_attacks(fr, occ, DIAG_DIRS) | sliding_attacks(fr, occ, ORTHO_DIRS)) & ~us_occ
            for to in iter_bits(targets):
                yield Move(fr, to)

        # King
        ksq = self.king_sq(side)
        targets = KING_ATTACKS[ksq] & ~us_occ
        for to in iter_bits(targets):
            yield Move(ksq, to)

        # Castling (very basic legality checks)
        if side == WHITE and ksq == 4 and not self.in_check(WHITE):
            if (self.castle & 1) and self.board[5] == -1 and self.board[6] == -1:
                if self.attackers_to(5, BLACK) == 0 and self.attackers_to(6, BLACK) == 0:
                    yield Move(4, 6, is_castle=True)
            if (self.castle & 2) and self.board[3] == -1 and self.board[2] == -1 and self.board[1] == -1:
                if self.attackers_to(3, BLACK) == 0 and self.attackers_to(2, BLACK) == 0:
                    yield Move(4, 2, is_castle=True)
        if side == BLACK and ksq == 60 and not self.in_check(BLACK):
            if (self.castle & 4) and self.board[61] == -1 and self.board[62] == -1:
                if self.attackers_to(61, WHITE) == 0 and self.attackers_to(62, WHITE) == 0:
                    yield Move(60, 62, is_castle=True)
            if (self.castle & 8) and self.board[59] == -1 and self.board[58] == -1 and self.board[57] == -1:
                if self.attackers_to(59, WHITE) == 0 and self.attackers_to(58, WHITE) == 0:
                    yield Move(60, 58, is_castle=True)

    def legal_moves(self):
        for mv in self._pseudo_moves():
            u = self.push(mv)
            ok = not self.in_check(BLACK if self.side == WHITE else WHITE)  # side already flipped in push
            self.pop(u)
            if ok:
                yield mv

    def _pseudo_captures(self):
        """Yield pseudo-legal captures only (incl. ep and capture-promotions)."""
        side = self.side
        us_occ = self.occ_w if side == WHITE else self.occ_b
        them_occ = self.occ_b if side == WHITE else self.occ_w
        occ = self.occ

        # Pawns: captures + ep + capture promotions
        if side == WHITE:
            pawns = self.bb[WP]
            for fr in iter_bits(pawns):
                r = rank_of(fr)
                caps = PAWN_ATTACKS[WHITE][fr] & them_occ
                for to in iter_bits(caps):
                    if r == 6:
                        for promo in (1, 2, 3, 4):
                            yield Move(fr, to, promo=promo)
                    else:
                        yield Move(fr, to)
                if self.ep_sq is not None and (PAWN_ATTACKS[WHITE][fr] & bb(self.ep_sq)):
                    yield Move(fr, self.ep_sq, is_ep=True)
        else:
            pawns = self.bb[BP]
            for fr in iter_bits(pawns):
                r = rank_of(fr)
                caps = PAWN_ATTACKS[BLACK][fr] & them_occ
                for to in iter_bits(caps):
                    if r == 1:
                        for promo in (1, 2, 3, 4):
                            yield Move(fr, to, promo=promo)
                    else:
                        yield Move(fr, to)
                if self.ep_sq is not None and (PAWN_ATTACKS[BLACK][fr] & bb(self.ep_sq)):
                    yield Move(fr, self.ep_sq, is_ep=True)

        # Knights
        knights = self.bb[WN if side == WHITE else BN]
        for fr in iter_bits(knights):
            targets = KNIGHT_ATTACKS[fr] & them_occ
            for to in iter_bits(targets):
                yield Move(fr, to)

        # Bishops/Rooks/Queens
        bishops = self.bb[WB if side == WHITE else BB]
        for fr in iter_bits(bishops):
            targets = sliding_attacks(fr, occ, DIAG_DIRS) & them_occ
            for to in iter_bits(targets):
                yield Move(fr, to)
        rooks = self.bb[WR if side == WHITE else BR]
        for fr in iter_bits(rooks):
            targets = sliding_attacks(fr, occ, ORTHO_DIRS) & them_occ
            for to in iter_bits(targets):
                yield Move(fr, to)
        queens = self.bb[WQ if side == WHITE else BQ]
        for fr in iter_bits(queens):
            targets = (sliding_attacks(fr, occ, DIAG_DIRS) | sliding_attacks(fr, occ, ORTHO_DIRS)) & them_occ
            for to in iter_bits(targets):
                yield Move(fr, to)

        # King captures (no castling in captures)
        ksq = self.king_sq(side)
        targets = KING_ATTACKS[ksq] & them_occ
        for to in iter_bits(targets):
            yield Move(ksq, to)

    def legal_captures(self):
        """Yield legal captures only (fast path for quiescence)."""
        for mv in self._pseudo_captures():
            u = self.push(mv)
            ok = not self.in_check(BLACK if self.side == WHITE else WHITE)  # side already flipped
            self.pop(u)
            if ok:
                yield mv

    def push(self, mv: Move) -> Undo:
        captured = self.board[mv.to_sq]
        piece = self.board[mv.from_sq]
        assert piece != -1
        undo = Undo(
            mv=mv,
            moved_piece=piece,
            captured=None,
            captured_sq=None,
            ep_sq=self.ep_sq,
            castle=self.castle,
            halfmove=self.halfmove,
            fullmove=self.fullmove,
            key=self.key,
            rook_move=None,
        )

        # update ep hash out
        self.key ^= _Z_EPFILE[8 if self.ep_sq is None else file_of(self.ep_sq)]
        self.ep_sq = None

        # capture
        if mv.is_ep:
            cap_sq = mv.to_sq - 8 if self.side == WHITE else mv.to_sq + 8
            cap_piece = self.board[cap_sq]
            undo.captured = cap_piece
            undo.captured_sq = cap_sq
            self._remove(cap_piece, cap_sq)
            self.key ^= _Z_PIECE[cap_piece][cap_sq]
        elif captured != -1:
            undo.captured = captured
            undo.captured_sq = mv.to_sq
            self._remove(captured, mv.to_sq)
            self.key ^= _Z_PIECE[captured][mv.to_sq]

        # move / promotion
        self._move_piece(piece, mv.from_sq, mv.to_sq)
        self.key ^= _Z_PIECE[piece][mv.from_sq] ^ _Z_PIECE[piece][mv.to_sq]

        if mv.promo is not None:
            # replace pawn with promoted piece
            new_piece = (mv.promo if self.side == WHITE else mv.promo + 6)
            self._remove(piece, mv.to_sq)
            self.key ^= _Z_PIECE[piece][mv.to_sq]
            self._put(new_piece, mv.to_sq)
            self.key ^= _Z_PIECE[new_piece][mv.to_sq]
            piece = new_piece

        # castling rook move
        if mv.is_castle:
            if self.side == WHITE:
                if mv.to_sq == 6:
                    rook_from, rook_to = 7, 5
                else:
                    rook_from, rook_to = 0, 3
                rook = self.board[rook_from]
                self._move_piece(rook, rook_from, rook_to)
                self.key ^= _Z_PIECE[rook][rook_from] ^ _Z_PIECE[rook][rook_to]
                undo.rook_move = (rook, rook_from, rook_to)
            else:
                if mv.to_sq == 62:
                    rook_from, rook_to = 63, 61
                else:
                    rook_from, rook_to = 56, 59
                rook = self.board[rook_from]
                self._move_piece(rook, rook_from, rook_to)
                self.key ^= _Z_PIECE[rook][rook_from] ^ _Z_PIECE[rook][rook_to]
                undo.rook_move = (rook, rook_from, rook_to)

        # update castling rights if king/rook moved or rook captured
        old_castle = self.castle
        if piece in (WK, BK):
            self.castle &= 0 if piece == WK else 0  # clear later by mask
        if mv.from_sq == 4 or mv.to_sq == 4:
            self.castle &= ~3
        if mv.from_sq == 60 or mv.to_sq == 60:
            self.castle &= ~12
        # rooks
        if mv.from_sq == 0 or mv.to_sq == 0:
            self.castle &= ~2
        if mv.from_sq == 7 or mv.to_sq == 7:
            self.castle &= ~1
        if mv.from_sq == 56 or mv.to_sq == 56:
            self.castle &= ~8
        if mv.from_sq == 63 or mv.to_sq == 63:
            self.castle &= ~4

        if self.castle != old_castle:
            self.key ^= _Z_CASTLE[old_castle] ^ _Z_CASTLE[self.castle]

        # double pawn push -> ep
        if piece in (WP, BP) and abs(mv.to_sq - mv.from_sq) == 16:
            self.ep_sq = (mv.from_sq + mv.to_sq) // 2
        # hash ep in
        self.key ^= _Z_EPFILE[8 if self.ep_sq is None else file_of(self.ep_sq)]

        # clocks
        if piece in (WP, BP) or captured != -1 or mv.is_ep:
            self.halfmove = 0
        else:
            self.halfmove += 1
        if self.side == BLACK:
            self.fullmove += 1

        # flip side
        self.side ^= 1
        self.key ^= _Z_TURN

        self._recalc_occ()
        return undo

    def pop(self, u: Undo) -> None:
        # restore side first (push flipped it)
        self.side ^= 1

        # restore meta
        self.ep_sq = u.ep_sq
        self.castle = u.castle
        self.halfmove = u.halfmove
        self.fullmove = u.fullmove
        self.key = u.key

        mv = u.mv

        # undo castling rook move
        if u.rook_move is not None:
            rook, rf, rt = u.rook_move
            self._move_piece(rook, rt, rf)

        # undo promotion: if move was a promotion, the piece currently on to_sq is promoted piece.
        moved_piece = u.moved_piece
        cur_piece = self.board[mv.to_sq]
        if mv.promo is not None:
            # remove promoted piece, restore pawn of correct color
            self._remove(cur_piece, mv.to_sq)
            self._put(moved_piece, mv.to_sq)

        # move piece back
        self._move_piece(moved_piece, mv.to_sq, mv.from_sq)

        # restore captured piece
        if u.captured is not None and u.captured_sq is not None:
            self._put(u.captured, u.captured_sq)

        self._recalc_occ()


def repetition_counts_along_uci_line(start_fen: str, uci_moves: list[str]) -> dict[int, int]:
    """
    Count how many times each zobrist key (Position.key) appears when replaying
    from start_fen through uci_moves. Used for threefold repetition in search.
    """
    p = Position.from_fen(start_fen)
    counts: dict[int, int] = {}
    counts[p.key] = counts.get(p.key, 0) + 1
    for uci in uci_moves:
        mv = None
        for m in p.legal_moves():
            if m.uci() == uci:
                mv = m
                break
        if mv is None:
            break
        p.push(mv)
        counts[p.key] = counts.get(p.key, 0) + 1
    return counts

