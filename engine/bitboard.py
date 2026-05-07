"""
bitboard.py — bitboard utilities and precomputed attack tables.

Squares use 0..63 with a1=0, b1=1, ..., h8=63 (LSB = a1).
This matches common bitboard conventions and keeps shifts simple.
"""

from __future__ import annotations

from dataclasses import dataclass


def bb(sq: int) -> int:
    return 1 << sq


FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080
RANK_1 = 0x00000000000000FF
RANK_8 = 0xFF00000000000000


def file_of(sq: int) -> int:
    return sq & 7


def rank_of(sq: int) -> int:
    return sq >> 3


def pop_lsb(x: int) -> tuple[int, int]:
    """Return (lsb_index, new_x) for nonzero x."""
    lsb = x & -x
    idx = (lsb.bit_length() - 1)
    return idx, x ^ lsb


def iter_bits(x: int):
    while x:
        s, x = pop_lsb(x)
        yield s


_KNIGHT_DELTAS = ((2, 1), (2, -1), (-2, 1), (-2, -1),
                  (1, 2), (1, -2), (-1, 2), (-1, -2))
_KING_DELTAS = ((-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1))


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


KNIGHT_ATTACKS = [0] * 64
KING_ATTACKS = [0] * 64
PAWN_ATTACKS = [[0] * 64 for _ in range(2)]  # 0=white,1=black

for sq in range(64):
    r, c = rank_of(sq), file_of(sq)

    k = 0
    for dr, dc in _KNIGHT_DELTAS:
        rr, cc = r + dr, c + dc
        if _in_bounds(rr, cc):
            k |= bb(rr * 8 + cc)
    KNIGHT_ATTACKS[sq] = k

    k = 0
    for dr, dc in _KING_DELTAS:
        rr, cc = r + dr, c + dc
        if _in_bounds(rr, cc):
            k |= bb(rr * 8 + cc)
    KING_ATTACKS[sq] = k

    # pawn attacks (from sq)
    w = 0
    if r < 7:
        if c > 0:
            w |= bb((r + 1) * 8 + (c - 1))
        if c < 7:
            w |= bb((r + 1) * 8 + (c + 1))
    b_ = 0
    if r > 0:
        if c > 0:
            b_ |= bb((r - 1) * 8 + (c - 1))
        if c < 7:
            b_ |= bb((r - 1) * 8 + (c + 1))
    PAWN_ATTACKS[0][sq] = w
    PAWN_ATTACKS[1][sq] = b_


@dataclass(frozen=True)
class Rays:
    diag: tuple[int, ...]
    ortho: tuple[int, ...]


DIAG_DIRS = (9, 7, -7, -9)
ORTHO_DIRS = (8, -8, 1, -1)


def sliding_attacks(sq: int, occ: int, deltas: tuple[int, ...]) -> int:
    """Generate rook/bishop/queen rays from sq given occupancy."""
    attacks = 0
    r0, c0 = rank_of(sq), file_of(sq)

    delta_to_step = {
        9: (1, 1),
        7: (1, -1),
        -7: (-1, 1),
        -9: (-1, -1),
        8: (1, 0),
        -8: (-1, 0),
        1: (0, 1),
        -1: (0, -1),
    }

    for d in deltas:
        dr, dc = delta_to_step[d]
        r, c = r0 + dr, c0 + dc
        while 0 <= r < 8 and 0 <= c < 8:
            s2 = r * 8 + c
            attacks |= bb(s2)
            if occ & bb(s2):
                break
            r += dr
            c += dc
    return attacks

