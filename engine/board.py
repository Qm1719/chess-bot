"""
board.py — Chess board, rules, and state management.
Handles: piece constants, board representation, legal move generation
(castling, en passant, promotion), check / checkmate / stalemate / draw
detection, and FEN parsing / serialisation.
"""

# ── Piece constants ────────────────────────────────────────────────────────────
EMPTY  = 0
PAWN   = 1
KNIGHT = 2
BISHOP = 3
ROOK   = 4
QUEEN  = 5
KING   = 6

WHITE =  1
BLACK = -1

PIECE_VALUE = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 20000}

_FEN_TO_PIECE = {
    'P':  PAWN,   'N':  KNIGHT,  'B':  BISHOP,
    'R':  ROOK,   'Q':  QUEEN,   'K':  KING,
    'p': -PAWN,   'n': -KNIGHT,  'b': -BISHOP,
    'r': -ROOK,   'q': -QUEEN,   'k': -KING,
}
_PIECE_TO_FEN = {v: k for k, v in _FEN_TO_PIECE.items()}

PROMOTION_PIECES = (QUEEN, ROOK, BISHOP, KNIGHT)
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ── Helpers ────────────────────────────────────────────────────────────────────
def sq(r, c):
    return r * 8 + c

def rc(s):
    return s >> 3, s & 7

def piece_type(p):
    return abs(p)

def piece_color(p):
    return WHITE if p > 0 else BLACK

def piece_fen(p):
    return _PIECE_TO_FEN.get(p, '.')

def sq_name(s):
    r, c = rc(s)
    return "abcdefgh"[c] + str(8 - r)


# ── Fast board-level attack detection (works on plain list[64]) ───────────────
_KNIGHT_DELTAS = ((2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2))
_DIAG_DIRS     = ((1,1),(1,-1),(-1,1),(-1,-1))
_STRAIGHT_DIRS = ((1,0),(-1,0),(0,1),(0,-1))
_KING_DELTAS   = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))

def _board_attacked(board, s, by_color):
    """True if square s is attacked by any piece of by_color on the given board."""
    r, c = s >> 3, s & 7

    # Pawn attacks
    pawn_p = PAWN * by_color
    pr = r + 1 if by_color == WHITE else r - 1
    if 0 <= pr < 8:
        if c > 0 and board[pr * 8 + c - 1] == pawn_p:
            return True
        if c < 7 and board[pr * 8 + c + 1] == pawn_p:
            return True

    # Knight
    knight_p = KNIGHT * by_color
    for dr, dc in _KNIGHT_DELTAS:
        ar, ac = r + dr, c + dc
        if 0 <= ar < 8 and 0 <= ac < 8 and board[ar * 8 + ac] == knight_p:
            return True

    # King
    king_p = KING * by_color
    for dr, dc in _KING_DELTAS:
        ar, ac = r + dr, c + dc
        if 0 <= ar < 8 and 0 <= ac < 8 and board[ar * 8 + ac] == king_p:
            return True

    # Diagonal sliders (Bishop, Queen)
    bq1, bq2 = BISHOP * by_color, QUEEN * by_color
    for dr, dc in _DIAG_DIRS:
        ar, ac = r + dr, c + dc
        while 0 <= ar < 8 and 0 <= ac < 8:
            p = board[ar * 8 + ac]
            if p:
                if p == bq1 or p == bq2:
                    return True
                break
            ar += dr; ac += dc

    # Straight sliders (Rook, Queen)
    rq1, rq2 = ROOK * by_color, QUEEN * by_color
    for dr, dc in _STRAIGHT_DIRS:
        ar, ac = r + dr, c + dc
        while 0 <= ar < 8 and 0 <= ac < 8:
            p = board[ar * 8 + ac]
            if p:
                if p == rq1 or p == rq2:
                    return True
                break
            ar += dr; ac += dc

    return False


# ── Move ───────────────────────────────────────────────────────────────────────
class Move:
    __slots__ = ('from_sq', 'to_sq', 'promotion', 'is_ep', 'is_castle',
                 'captured', 'score')

    def __init__(self, from_sq, to_sq, promotion=EMPTY,
                 is_ep=False, is_castle=0, captured=EMPTY):
        self.from_sq   = from_sq
        self.to_sq     = to_sq
        self.promotion = promotion   # QUEEN/ROOK/BISHOP/KNIGHT or EMPTY
        self.is_ep     = is_ep       # en-passant capture
        self.is_castle = is_castle   # 1=kingside, -1=queenside, 0=none
        self.captured  = captured    # piece on destination square (for ordering)
        self.score     = 0

    def uci(self):
        s = sq_name(self.from_sq) + sq_name(self.to_sq)
        if self.promotion:
            s += "qrbn"[PROMOTION_PIECES.index(self.promotion)]
        return s

    def __eq__(self, other):
        return isinstance(other, Move) and (
            self.from_sq == other.from_sq and
            self.to_sq   == other.to_sq   and
            self.promotion == other.promotion
        )

    def __hash__(self):
        return hash((self.from_sq, self.to_sq, self.promotion))

    def __repr__(self):
        return f"Move({self.uci()})"


# ── GameState ──────────────────────────────────────────────────────────────────
class GameState:
    """
    board[sq] ∈ {-6,..,0,..,6}  (negative = black, positive = white)
    castling: dict with keys 'K','Q','k','q'
    ep_sq: square index that can be captured en-passant, or -1
    """

    __slots__ = ('board', 'turn', 'castling', 'ep_sq',
                 'halfmove', 'fullmove', '_move_history')

    def __init__(self):
        self.board        = [EMPTY] * 64
        self.turn         = WHITE
        self.castling     = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.ep_sq        = -1
        self.halfmove     = 0
        self.fullmove     = 1
        self._move_history = []

    # ── Constructors ──────────────────────────────────────────────────────────
    @classmethod
    def from_fen(cls, fen=STARTING_FEN):
        gs = cls()
        parts = fen.strip().split()
        placement, turn_ch, castle_str, ep_str = parts[0], parts[1], parts[2], parts[3]
        halfmove  = int(parts[4]) if len(parts) > 4 else 0
        fullmove  = int(parts[5]) if len(parts) > 5 else 1

        r = 0
        c = 0
        for ch in placement:
            if ch == '/':
                r += 1; c = 0
            elif ch.isdigit():
                c += int(ch)
            else:
                gs.board[sq(r, c)] = _FEN_TO_PIECE[ch]
                c += 1

        gs.turn     = WHITE if turn_ch == 'w' else BLACK
        gs.castling = {
            'K': 'K' in castle_str,
            'Q': 'Q' in castle_str,
            'k': 'k' in castle_str,
            'q': 'q' in castle_str,
        }
        gs.ep_sq    = -1 if ep_str == '-' else sq(8 - int(ep_str[1]), ord(ep_str[0]) - ord('a'))
        gs.halfmove = halfmove
        gs.fullmove = fullmove
        return gs

    def to_fen(self):
        rows = []
        for r in range(8):
            empty = 0
            row_s = ''
            for c in range(8):
                p = self.board[sq(r, c)]
                if p == EMPTY:
                    empty += 1
                else:
                    if empty:
                        row_s += str(empty); empty = 0
                    row_s += _PIECE_TO_FEN[p]
            if empty:
                row_s += str(empty)
            rows.append(row_s)
        placement  = '/'.join(rows)
        turn_ch    = 'w' if self.turn == WHITE else 'b'
        castle_str = ''.join(k for k in ('K', 'Q', 'k', 'q') if self.castling[k]) or '-'
        ep_str     = sq_name(self.ep_sq) if self.ep_sq >= 0 else '-'
        return f"{placement} {turn_ch} {castle_str} {ep_str} {self.halfmove} {self.fullmove}"

    def copy(self):
        gs = GameState.__new__(GameState)
        gs.board          = self.board[:]
        gs.turn           = self.turn
        gs.castling       = dict(self.castling)
        gs.ep_sq          = self.ep_sq
        gs.halfmove       = self.halfmove
        gs.fullmove       = self.fullmove
        gs._move_history  = self._move_history[:]
        return gs

    # ── Square queries ────────────────────────────────────────────────────────
    def piece_at(self, s):
        return self.board[s]

    def find_king(self, color):
        target = KING * color
        for s in range(64):
            if self.board[s] == target:
                return s
        return -1

    # ── Attack detection ──────────────────────────────────────────────────────
    def is_attacked(self, s, by_color):
        return _board_attacked(self.board, s, by_color)

    def is_in_check(self, color=None):
        if color is None:
            color = self.turn
        k = self.find_king(color)
        if k < 0:
            return False
        return _board_attacked(self.board, k, -color)

    # ── Pseudo-legal move generation ──────────────────────────────────────────
    def _gen_pseudo_moves(self):
        moves  = []
        board  = self.board
        color  = self.turn
        ep_sq  = self.ep_sq
        castle = self.castling

        for s in range(64):
            p = board[s]
            if p == EMPTY or piece_color(p) != color:
                continue
            pt = piece_type(p)
            r, c = rc(s)

            if pt == PAWN:
                moves.extend(self._pawn_moves(s, r, c, color, board, ep_sq))
            elif pt == KNIGHT:
                for dr, dc in ((2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)):
                    ar, ac = r+dr, c+dc
                    if 0 <= ar < 8 and 0 <= ac < 8:
                        ts = sq(ar, ac)
                        target = board[ts]
                        if target == EMPTY or piece_color(target) != color:
                            moves.append(Move(s, ts, captured=target))
            elif pt == BISHOP:
                moves.extend(self._slide(s, r, c, color, board,
                                         ((1,1),(1,-1),(-1,1),(-1,-1))))
            elif pt == ROOK:
                moves.extend(self._slide(s, r, c, color, board,
                                         ((1,0),(-1,0),(0,1),(0,-1))))
            elif pt == QUEEN:
                moves.extend(self._slide(s, r, c, color, board,
                    ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))))
            elif pt == KING:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == dc == 0:
                            continue
                        ar, ac = r+dr, c+dc
                        if 0 <= ar < 8 and 0 <= ac < 8:
                            ts = sq(ar, ac)
                            target = board[ts]
                            if target == EMPTY or piece_color(target) != color:
                                moves.append(Move(s, ts, captured=target))
                moves.extend(self._castling_moves(s, r, c, color, board, castle))

        return moves

    def _pawn_moves(self, s, r, c, color, board, ep_sq):
        moves     = []
        direction = -1 if color == WHITE else 1
        start_row = 6 if color == WHITE else 1
        promo_row = 0 if color == WHITE else 7

        nr = r + direction
        if 0 <= nr < 8 and board[sq(nr, c)] == EMPTY:
            if nr == promo_row:
                for pp in PROMOTION_PIECES:
                    moves.append(Move(s, sq(nr, c), promotion=pp))
            else:
                moves.append(Move(s, sq(nr, c)))
                if r == start_row and board[sq(nr + direction, c)] == EMPTY:
                    moves.append(Move(s, sq(nr + direction, c)))

        for dc in (-1, 1):
            ac = c + dc
            if 0 <= ac < 8 and 0 <= nr < 8:
                ts = sq(nr, ac)
                target = board[ts]
                if target != EMPTY and piece_color(target) != color:
                    if nr == promo_row:
                        for pp in PROMOTION_PIECES:
                            moves.append(Move(s, ts, promotion=pp, captured=target))
                    else:
                        moves.append(Move(s, ts, captured=target))
                elif ts == ep_sq and ep_sq >= 0:
                    moves.append(Move(s, ts, is_ep=True))

        return moves

    def _slide(self, s, r, c, color, board, directions):
        moves = []
        for dr, dc in directions:
            ar, ac = r + dr, c + dc
            while 0 <= ar < 8 and 0 <= ac < 8:
                ts     = sq(ar, ac)
                target = board[ts]
                if target == EMPTY:
                    moves.append(Move(s, ts))
                elif piece_color(target) != color:
                    moves.append(Move(s, ts, captured=target))
                    break
                else:
                    break
                ar += dr; ac += dc
        return moves

    def _castling_moves(self, king_sq, r, c, color, board, castle):
        moves = []
        if self.is_in_check(color):
            return moves
        if color == WHITE:
            if castle['K'] and board[sq(7,5)] == EMPTY and board[sq(7,6)] == EMPTY:
                if not self.is_attacked(sq(7,5), BLACK) and not self.is_attacked(sq(7,6), BLACK):
                    moves.append(Move(sq(7,4), sq(7,6), is_castle=1))
            if castle['Q'] and (board[sq(7,3)] == EMPTY and board[sq(7,2)] == EMPTY
                                  and board[sq(7,1)] == EMPTY):
                if not self.is_attacked(sq(7,3), BLACK) and not self.is_attacked(sq(7,2), BLACK):
                    moves.append(Move(sq(7,4), sq(7,2), is_castle=-1))
        else:
            if castle['k'] and board[sq(0,5)] == EMPTY and board[sq(0,6)] == EMPTY:
                if not self.is_attacked(sq(0,5), WHITE) and not self.is_attacked(sq(0,6), WHITE):
                    moves.append(Move(sq(0,4), sq(0,6), is_castle=1))
            if castle['q'] and (board[sq(0,3)] == EMPTY and board[sq(0,2)] == EMPTY
                                  and board[sq(0,1)] == EMPTY):
                if not self.is_attacked(sq(0,3), WHITE) and not self.is_attacked(sq(0,2), WHITE):
                    moves.append(Move(sq(0,4), sq(0,2), is_castle=-1))
        return moves

    # ── Legal move generation ──────────────────────────────────────────────────
    def _is_legal_fast(self, move):
        """Check legality using only a board copy (no full GameState alloc)."""
        board = self.board[:]
        color = self.turn
        fr, fc = move.from_sq >> 3, move.from_sq & 7

        board[move.to_sq]   = board[move.from_sq]
        board[move.from_sq] = EMPTY

        if move.is_ep:
            _, tc = rc(move.to_sq)
            board[fr * 8 + tc] = EMPTY

        if move.promotion:
            board[move.to_sq] = move.promotion * color

        if move.is_castle:
            if color == WHITE:
                if move.is_castle == 1:
                    board[61] = board[63]; board[63] = EMPTY
                else:
                    board[59] = board[56]; board[56] = EMPTY
            else:
                if move.is_castle == 1:
                    board[5] = board[7]; board[7] = EMPTY
                else:
                    board[3] = board[0]; board[0] = EMPTY

        king_p  = KING * color
        king_sq = next((s for s in range(64) if board[s] == king_p), -1)
        if king_sq < 0:
            return False
        return not _board_attacked(board, king_sq, -color)

    def get_legal_moves(self):
        return [m for m in self._gen_pseudo_moves() if self._is_legal_fast(m)]

    def get_legal_moves_for(self, s):
        return [m for m in self.get_legal_moves() if m.from_sq == s]

    # ── Apply move (returns new state) ────────────────────────────────────────
    def _apply(self, move):
        """Fast apply for search — does NOT preserve _move_history."""
        gs               = GameState.__new__(GameState)
        gs.board         = self.board[:]
        gs.turn          = self.turn
        gs.castling      = dict(self.castling)
        gs.ep_sq         = self.ep_sq
        gs.halfmove      = self.halfmove
        gs.fullmove      = self.fullmove
        gs._move_history = []
        board    = gs.board
        color    = gs.turn
        fr, fc   = rc(move.from_sq)
        tr, tc   = rc(move.to_sq)
        moving   = board[move.from_sq]
        pt       = piece_type(moving)
        captured = board[move.to_sq]

        board[move.to_sq]   = moving
        board[move.from_sq] = EMPTY

        if move.is_ep:
            board[sq(fr, tc)] = EMPTY
            captured = PAWN * (-color)

        if move.promotion:
            board[move.to_sq] = move.promotion * color

        if move.is_castle:
            if color == WHITE:
                if move.is_castle == 1:
                    board[sq(7,5)] = board[sq(7,7)]; board[sq(7,7)] = EMPTY
                else:
                    board[sq(7,3)] = board[sq(7,0)]; board[sq(7,0)] = EMPTY
            else:
                if move.is_castle == 1:
                    board[sq(0,5)] = board[sq(0,7)]; board[sq(0,7)] = EMPTY
                else:
                    board[sq(0,3)] = board[sq(0,0)]; board[sq(0,0)] = EMPTY

        if pt == KING:
            if color == WHITE:
                gs.castling['K'] = False; gs.castling['Q'] = False
            else:
                gs.castling['k'] = False; gs.castling['q'] = False
        if pt == ROOK:
            if move.from_sq == sq(7,7): gs.castling['K'] = False
            if move.from_sq == sq(7,0): gs.castling['Q'] = False
            if move.from_sq == sq(0,7): gs.castling['k'] = False
            if move.from_sq == sq(0,0): gs.castling['q'] = False
        if captured != EMPTY:
            if move.to_sq == sq(7,7): gs.castling['K'] = False
            if move.to_sq == sq(7,0): gs.castling['Q'] = False
            if move.to_sq == sq(0,7): gs.castling['k'] = False
            if move.to_sq == sq(0,0): gs.castling['q'] = False

        gs.ep_sq = sq((fr + tr) // 2, fc) if pt == PAWN and abs(tr - fr) == 2 else -1

        gs.halfmove = 0 if (pt == PAWN or captured != EMPTY) else self.halfmove + 1

        if color == BLACK:
            gs.fullmove += 1

        gs.turn = -color
        return gs

    def make_move(self, move):
        """Public API — preserves move history for GUI and game replay."""
        gs = self._apply(move)
        gs._move_history = self._move_history + [move]
        return gs

    # ── Game-over detection ───────────────────────────────────────────────────
    def has_legal_moves(self):
        return any(self._is_legal_fast(m) for m in self._gen_pseudo_moves())

    def is_checkmate(self):
        return self.is_in_check() and not self.has_legal_moves()

    def is_stalemate(self):
        return not self.is_in_check() and not self.has_legal_moves()

    def is_draw_fifty(self):
        return self.halfmove >= 100

    def is_draw_insufficient(self):
        pieces = [p for p in self.board if p != EMPTY]
        if len(pieces) == 2:
            return True
        if len(pieces) == 3:
            types = [piece_type(p) for p in pieces]
            return BISHOP in types or KNIGHT in types
        if len(pieces) == 4:
            whites = [p for p in pieces if p > 0]
            blacks = [p for p in pieces if p < 0]
            if (len(whites) == 2 and len(blacks) == 2
                    and piece_type(whites[0]) in (KING, BISHOP)
                    and piece_type(whites[1]) in (KING, BISHOP)
                    and piece_type(blacks[0]) in (KING, BISHOP)
                    and piece_type(blacks[1]) in (KING, BISHOP)):
                bsqs = [s for s in range(64) if piece_type(self.board[s]) == BISHOP]
                if len(bsqs) == 2:
                    r0, c0 = rc(bsqs[0])
                    r1, c1 = rc(bsqs[1])
                    if (r0 + c0) % 2 == (r1 + c1) % 2:
                        return True
        return False

    def game_result(self):
        """Returns 'white', 'black', 'draw', or None (game ongoing)."""
        if self.is_checkmate():
            return 'black' if self.turn == WHITE else 'white'
        if self.is_stalemate() or self.is_draw_fifty() or self.is_draw_insufficient():
            return 'draw'
        return None

    # ── Position key for repetition tracking ─────────────────────────────────
    def position_key(self):
        return ' '.join(self.to_fen().split()[:4])

    # ── Utility ───────────────────────────────────────────────────────────────
    def material_balance(self):
        total = 0
        for p in self.board:
            if p == EMPTY:
                continue
            val = PIECE_VALUE[piece_type(p)]
            total += val if p > 0 else -val
        return total

    def __repr__(self):
        lines = []
        for r in range(8):
            row_s = ''
            for c in range(8):
                p = self.board[sq(r, c)]
                row_s += (_PIECE_TO_FEN.get(p, '.') + ' ')
            lines.append(f"{8-r} | {row_s}")
        lines.append("   a b c d e f g h")
        return '\n'.join(lines)
