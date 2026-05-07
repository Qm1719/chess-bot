"""
gui.py — Chess Game GUI (Pygame)
Supports:
  - Human vs Human
  - Human vs AI (choose side)
  - AI vs AI (watch two agents play)
  - AI preset levels 1–10 (L1 Random; L2–L10 handicap variants of v4 — see agents_v4.SkillConfig)
  - Independent clock presets for White / Black (1, 3, 5, 10, 15 min each): counts down on
    the side to move. AI moves use an online-style slice of remaining time (not the whole clock
    each turn) plus cooperative stop_event; Human-vs-AI spawns short ponder bursts on the human's clock.
  - Pawn promotion dialog
  - Move history sidebar
  - Threefold-repetition tracking
  - New Game / Resign / Flip Board
"""
from __future__ import annotations
import sys
import os
import threading

import pygame

from board import (
    GameState, Move,
    EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    WHITE, BLACK,
    piece_type, piece_color, sq, rc,
    STARTING_FEN, PIECE_VALUE,
)
from agents_v4 import get_agent

# ── Layout constants ──────────────────────────────────────────────────────────
SQ_SIZE    = 100          # pixels per square
BOARD_PX   = SQ_SIZE * 8  # 576
SIDEBAR_W  = 280
MARGIN     = 12
WIN_W      = BOARD_PX + SIDEBAR_W + MARGIN * 3
WIN_H      = BOARD_PX + MARGIN * 2
FPS        = 30

# ── Colours ───────────────────────────────────────────────────────────────────
C_LIGHT        = (240, 217, 181)
C_DARK         = (181, 136,  99)
C_SELECTED     = ( 20, 200, 100, 180)
C_LEGAL        = (100, 200, 100, 130)
C_LAST_MOVE    = (255, 255,  80, 120)
C_CHECK        = (220,  40,  40, 160)
C_BG           = ( 40,  40,  40)
C_SIDEBAR_BG   = ( 50,  50,  60)
C_TEXT         = (230, 230, 230)
C_TEXT_DIM     = (160, 160, 160)
C_BUTTON       = ( 70, 130, 180)
C_BUTTON_HOVER = (100, 155, 210)
C_BUTTON_RED   = (180,  60,  60)
C_BLACK_PIECE  = ( 20,  20,  20)
C_WHITE_PIECE  = (255, 255, 255)
C_ACCENT       = (255, 200,  50)

# Clock presets shown in menu (minutes per side for the upcoming game).
TIME_PRESET_MINUTES = (1, 3, 5, 10, 15)

# ── Unicode glyphs ────────────────────────────────────────────────────────────
_GLYPH = {
    (KING,   WHITE): '♚', (QUEEN,  WHITE): '♛', (ROOK,   WHITE): '♜',
    (BISHOP, WHITE): '♝', (KNIGHT, WHITE): '♞', (PAWN,   WHITE): '♟',
    (KING,   BLACK): '♚', (QUEEN,  BLACK): '♛', (ROOK,   BLACK): '♜',
    (BISHOP, BLACK): '♝', (KNIGHT, BLACK): '♞', (PAWN,   BLACK): '♟',
}

def _glyph(p):
    if p == EMPTY:
        return ''
    return _GLYPH.get((piece_type(p), piece_color(p)), '?')

# ── Helpers ───────────────────────────────────────────────────────────────────
def sq_to_pixel(s, flipped=False):
    r, c = rc(s)
    if flipped:
        r = 7 - r
        c = 7 - c
    return MARGIN + c * SQ_SIZE, MARGIN + r * SQ_SIZE

def pixel_to_sq(px, py, flipped=False):
    c = (px - MARGIN) // SQ_SIZE
    r = (py - MARGIN) // SQ_SIZE
    if not (0 <= r < 8 and 0 <= c < 8):
        return -1
    if flipped:
        r = 7 - r
        c = 7 - c
    return sq(r, c)

def san_move(state_before, move, state_after=None):
    """
    Full SAN notation for a move, including disambiguation and +/# suffixes.
    state_after may be passed in to avoid recomputing it.
    """
    board  = state_before.board
    pt     = piece_type(board[move.from_sq])
    cols   = 'abcdefgh'
    fr, fc = rc(move.from_sq)
    tr, tc = rc(move.to_sq)
    PIECE_CH = {PAWN: '', KNIGHT: 'N', BISHOP: 'B', ROOK: 'R', QUEEN: 'Q', KING: 'K'}
    cap   = 'x' if (move.captured or move.is_ep) else ''
    promo = ('=' + {QUEEN: 'Q', ROOK: 'R', BISHOP: 'B', KNIGHT: 'N'}.get(move.promotion, '')
             if move.promotion else '')

    # ── Castling ──────────────────────────────────────────────────────────────
    if move.is_castle == 1:
        base = 'O-O'
    elif move.is_castle == -1:
        base = 'O-O-O'
    else:
        if pt == PAWN:
            prefix = cols[fc] if cap else ''
        else:
            pc = PIECE_CH[pt]
            color = piece_color(board[move.from_sq])
            # Find other same-type / same-color pieces that can also reach to_sq.
            other = [
                m.from_sq for m in state_before.get_legal_moves()
                if (m.from_sq != move.from_sq
                    and m.to_sq  == move.to_sq
                    and piece_type(board[m.from_sq])  == pt
                    and piece_color(board[m.from_sq]) == color)
            ]
            if not other:
                prefix = pc
            else:
                other_files = {rc(s)[1] for s in other}
                other_ranks = {rc(s)[0] for s in other}
                if fc not in other_files:          # unique file → use file
                    prefix = pc + cols[fc]
                elif fr not in other_ranks:        # unique rank → use rank
                    prefix = pc + str(8 - fr)
                else:                              # neither unique → use both
                    prefix = pc + cols[fc] + str(8 - fr)

        base = f"{prefix}{cap}{cols[tc]}{8-tr}{promo}"

    # ── Check / checkmate suffix ───────────────────────────────────────────────
    if state_after is None:
        state_after = state_before._apply(move)
    result = state_after.game_result()
    if result in ('white', 'black'):
        base += '#'
    elif state_after.is_in_check():
        base += '+'

    return base


# ── Button widget ──────────────────────────────────────────────────────────────
class Button:
    def __init__(self, rect, label, color=None, hover_color=None, font=None):
        self.rect        = pygame.Rect(rect)
        self.label       = label
        self.color       = color or C_BUTTON
        self.hover_color = hover_color or C_BUTTON_HOVER
        self._font       = font

    def draw(self, surface, font=None):
        f = font or self._font
        mouse = pygame.mouse.get_pos()
        col   = self.hover_color if self.rect.collidepoint(mouse) else self.color
        pygame.draw.rect(surface, col, self.rect, border_radius=6)
        pygame.draw.rect(surface, C_TEXT, self.rect, 1, border_radius=6)
        if f:
            txt = f.render(self.label, True, C_TEXT)
            tr  = txt.get_rect(center=self.rect.center)
            surface.blit(txt, tr)

    def is_clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))


# ── Main GUI class ─────────────────────────────────────────────────────────────
class ChessGUI:

    MODES = ['Human vs Human', 'Human vs AI (White)', 'Human vs AI (Black)', 'AI vs AI']

    # ── Sound helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _load_sounds():
        """Load Chess Titans WAV files.  Returns a dict or empty dict on error."""
        import random
        folder = os.path.join(os.path.dirname(__file__),
                              "PC _ Computer - Chess Titans - Sound Effects - Sounds")
        def _snd(name):
            path = os.path.join(folder, name)
            try:
                return pygame.mixer.Sound(path) if os.path.exists(path) else None
            except Exception:
                return None

        move_sounds = [s for s in (
            _snd("Move_Piece (1).WAV"), _snd("Move_Piece (2).WAV"),
            _snd("Move_Piece (3).WAV"), _snd("Move_Piece (4).WAV"),
            _snd("Move_Piece (5).WAV"), _snd("Move_Piece (6).WAV"),
        ) if s]

        return {
            "move":           move_sounds,
            "capture":        _snd("Capture.WAV"),
            "check":          _snd("Check.WAV"),
            "checkmate_win":  _snd("Checkmate_Win.WAV"),
            "checkmate_lose": _snd("Checkmate_Lose.WAV"),
            "illegal":        _snd("Illegal_Move.WAV"),
        }

    def _play(self, key):
        """Play a sound by key.  Silently ignores missing sounds."""
        import random
        if not self._sounds:
            return
        snd = self._sounds.get(key)
        if isinstance(snd, list):
            snd = random.choice(snd) if snd else None
        if snd:
            snd.play()

    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        pygame.display.set_caption("Chess — AI Assignment")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock  = pygame.time.Clock()

        # Sounds
        self._sounds = self._load_sounds()

        # Fonts
        self.font_piece   = pygame.font.SysFont('segoeuisymbol', SQ_SIZE - 12, bold=False)
        self.font_ui      = pygame.font.SysFont('segoeui',  18)
        self.font_ui_b    = pygame.font.SysFont('segoeui',  18, bold=True)
        self.font_small   = pygame.font.SysFont('segoeui',  14)
        self.font_title   = pygame.font.SysFont('segoeui',  22, bold=True)
        self.font_big     = pygame.font.SysFont('segoeui',  36, bold=True)

        self._overlay_surf = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)

        # Game state
        self.state        = GameState.from_fen(STARTING_FEN)
        self.pos_history  = {}          # FEN key → count (repetition)
        self.move_history = []          # list of (state_before, Move, san_str)
        self.selected_sq  = -1
        self.legal_moves  = []          # legal moves for selected piece
        self.flipped      = False       # flip board (black at bottom)
        self.last_move    = None

        # Modes / levels / clocks (minutes per side, chosen before Start Game)
        self.mode         = 'Human vs AI (Black)'   # default
        self.ai_level_w   = 10   # see agents_v4.SkillConfig (1 Random … 10 full v4)
        self.ai_level_b   = 10
        self.clock_w_min  = 5     # preset index resolves via TIME_PRESET_MINUTES
        self.clock_b_min  = 5
        self.clock_remaining_w = 0.0
        self.clock_remaining_b = 0.0
        self.agent_white  = None
        self.agent_black  = None
        self._rebuild_agents()

        # AI threading (+ cooperative stop when clock runs out / leaving game)
        self._ai_thread       = None
        self._ai_result       = None   # Move or None when done
        self._ai_thinking     = False
        self._ai_stop_event   = threading.Event()
        self._ai_generation   = 0  # bumped when aborting searches; stale threads must not publish moves
        self._ponder_epoch    = 0  # pondering threads bail if bumped (invalidate / menu / new game)
        self._ponder_thread   = None
        self._last_ponder_ms  = 0

        # Promotion
        self._promo_move  = None   # pending Move awaiting promotion choice

        # Screen state
        self._screen_state = 'menu'   # 'menu' | 'game' | 'gameover'
        self._result_text  = ''

        # AI-vs-AI auto-play speed (ms between moves)
        self.ai_delay_ms   = 400
        self._last_ai_tick = 0
        self._last_frame_ticks = pygame.time.get_ticks()

        self.animating = False
        self.anim_piece = EMPTY
        self.anim_start_sq = -1
        self.anim_end_sq = -1
        self.anim_progress = 0.0
        self.anim_duration = 0.15

        self._build_menu_buttons()
        self._build_game_buttons()

    # ── Agent factory ─────────────────────────────────────────────────────────
    def _rebuild_agents(self):
        self.agent_white = get_agent(self.ai_level_w)
        self.agent_black = get_agent(self.ai_level_b)

    def _invalidate_ai_sessions(self) -> None:
        """Invalidate in-flight daemon searches (timeout, resign, menu, new game)."""
        self._ai_generation += 1
        self._ponder_epoch += 1
        self._ai_stop_event.set()

    def _begin_ai_turn_search(self) -> int:
        """Fresh generation for the next AI think; clears stop latch for this move only."""
        self._ai_generation += 1
        self._ponder_epoch += 1  # discard in-flight pondering before real choose_move hits TT
        g = self._ai_generation
        self._ai_stop_event.clear()
        return g

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_menu_buttons(self):
        cx    = WIN_W // 2
        bw, bh = 280, 44
        self._menu_buttons = {}
        y = 180
        for mode in self.MODES:
            self._menu_buttons[mode] = Button((cx - bw//2, y, bw, bh), mode)
            y += bh + 10
        # Low enough to clear level + clock rows on the menu
        self._menu_buttons['Start'] = Button((cx - 100, 600, 200, 50), 'Start Game',
                                             color=(50, 150, 50), hover_color=(70, 180, 70))

    def _build_game_buttons(self):
        bx  = BOARD_PX + MARGIN * 2
        bw  = SIDEBAR_W - MARGIN
        bh  = 34
        y   = MARGIN
        self._btn_new     = Button((bx, y, bw, bh), 'New Game')
        y += bh + 6
        self._btn_flip    = Button((bx, y, bw, bh), 'Flip Board')
        y += bh + 6
        self._btn_resign  = Button((bx, y, bw, bh), 'Resign', color=C_BUTTON_RED,
                                   hover_color=(210, 80, 80))
        y += bh + 6
        self._btn_pgn     = Button((bx, y, bw, bh), 'Copy PGN',
                                   color=(60, 100, 140), hover_color=(80, 125, 170))
        self._pgn_copied_until = 0   # ticks timestamp until which "Copied!" is shown

    # ── New game ──────────────────────────────────────────────────────────────
    def new_game(self):
        self._invalidate_ai_sessions()
        self.state        = GameState.from_fen(STARTING_FEN)
        self.pos_history  = {self.state.position_key(): 1}
        self.move_history = []
        self.selected_sq  = -1
        self.legal_moves  = []
        self.last_move    = None
        self._promo_move  = None
        self._ai_result   = None
        self._ai_thinking = False
        self._ai_thread   = None
        self._result_text = ''
        self._last_ai_tick = 0
        self._rebuild_agents()
        self._screen_state = 'game'
        self._sync_clocks_from_menu()

    def _sync_clocks_from_menu(self) -> None:
        self.clock_remaining_w = float(self.clock_w_min) * 60.0
        self.clock_remaining_b = float(self.clock_b_min) * 60.0

    @staticmethod
    def _uci_move_to_gs_move(gs: GameState, uci: str | None):
        if not uci:
            return None
        for m in gs.get_legal_moves():
            if m.uci() == uci:
                return m
        return None

    def _clock_remaining_stm(self) -> float:
        """Whole seconds left on the clock for whoever is on move."""
        if self.state.turn == WHITE:
            return max(self.clock_remaining_w, 1.0 / 60.0)
        return max(self.clock_remaining_b, 1.0 / 60.0)

    def _online_style_move_budget_s(self, rem_STM: float) -> float:
        """
        Per-move budget for the AI (similar spirit to classical engines online):
        remainder / estimated moves-to-go, clamped — not the entire pool each turn.
        """
        r = max(float(rem_STM), 0.001)
        if r <= 4.0:
            return max(0.08, min(r * 0.82, r))
        if r <= 20.0:
            mtg = max(26.0, min(52.0, 32.0 + r / 20.0))
            return max(0.12, min(r / mtg, r * 0.34, r, 9.5))
        if r <= 120.0:
            mtg = max(34.0, min(62.0, 40.0 + r / 45.0))
            return max(0.14, min(r / mtg, r * 0.26, r, 12.0))
        mtg = max(36.0, min(76.0, 48.0 + r / 120.0))
        return max(0.18, min(r / mtg, r * 0.20, r, 16.5))

    def _ponder_budget_s_for_side(self, rem_bot: float) -> float:
        """Short bursts on the opponent's clock (does not decrement our clock yet)."""
        r = max(float(rem_bot), 0.001)
        return max(0.22, min(3.25, max(r * 0.032, r / 150.0) + 0.14))

    def _tick_clock(self, dt: float) -> None:
        """Count down whichever side is on move."""
        if self.state.turn == WHITE:
            self.clock_remaining_w = max(0.0, self.clock_remaining_w - dt)
            if self.clock_remaining_w <= 0.0:
                self._game_over('Black wins on time')
        else:
            self.clock_remaining_b = max(0.0, self.clock_remaining_b - dt)
            if self.clock_remaining_b <= 0.0:
                self._game_over('White wins on time')

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if self._screen_state == 'menu':
                    self._handle_menu_event(event)
                elif self._screen_state == 'game':
                    self._handle_game_event(event)
                elif self._screen_state == 'gameover':
                    self._handle_gameover_event(event)

            self.screen.fill(C_BG)
            if self._screen_state == 'menu':
                self._draw_menu()
            elif self._screen_state == 'game':
                self._tick_game()
                self._draw_game()
            elif self._screen_state == 'gameover':
                self._tick_game()
                self._draw_game()
                if not self.animating:
                    self._draw_gameover_overlay()

            pygame.display.flip()
            self.clock.tick(FPS)

    # ── Menu ──────────────────────────────────────────────────────────────────
    def _draw_menu(self):
        # Title
        title = self.font_big.render("Chess", True, C_ACCENT)
        self.screen.blit(title, title.get_rect(center=(WIN_W // 2, 90)))
        sub = self.font_ui.render("Select Game Mode", True, C_TEXT_DIM)
        self.screen.blit(sub, sub.get_rect(center=(WIN_W // 2, 135)))
        # hint = self.font_small.render(
        #     "AI levels: L10 = full v4 … L9 strip book/exp/Syzygy … L8–L2 handicap … L1 random",
        #     True, C_TEXT_DIM,
        # )
        # self.screen.blit(hint, hint.get_rect(center=(WIN_W // 2, 158)))

        for name, btn in self._menu_buttons.items():
            if name == self.mode:
                col = (50, 160, 80)
                btn_copy = Button(btn.rect, btn.label, color=col, hover_color=(70, 190, 100))
                btn_copy.draw(self.screen, self.font_ui_b)
            else:
                btn.draw(self.screen, self.font_ui)

        # Level sliders for AI modes
        cx = WIN_W // 2
        cx_offset = 80
        y_lv = 450
        if self.mode in ('Human vs AI (White)', 'Human vs AI (Black)', 'AI vs AI'):
            if self.mode == 'Human vs AI (White)':
                self._draw_level_selector(cx - cx_offset, y_lv, 'White AI preset', self.ai_level_w, 'w')
            if self.mode == 'Human vs AI (Black)':
                self._draw_level_selector(cx - cx_offset, y_lv, 'Black AI preset', self.ai_level_b, 'b')
            if self.mode == 'AI vs AI':
                self._draw_level_selector(cx - cx_offset, y_lv, 'White AI preset', self.ai_level_w, 'w')
                self._draw_level_selector(cx - cx_offset, y_lv + 28, 'Black AI preset', self.ai_level_b, 'b')

        y_tm = 520 if self.mode in ('Human vs AI (White)', 'Human vs AI (Black)', 'AI vs AI') else 440
        self._draw_clock_selector(cx - 220, y_tm, 'White clock', self.clock_w_min, 'cw')
        self._draw_clock_selector(cx + 40, y_tm, 'Black clock', self.clock_b_min, 'cb')

    def _cycle_clock_preset(self, current_min: int, delta: int) -> int:
        try:
            i = TIME_PRESET_MINUTES.index(int(current_min))
        except ValueError:
            i = TIME_PRESET_MINUTES.index(5)
        i = (i + delta) % len(TIME_PRESET_MINUTES)
        return TIME_PRESET_MINUTES[i]

    def _draw_clock_selector(self, x, y, label, current_min: int, tag: str):
        txt = self.font_small.render(label + f": {current_min} min", True, C_TEXT)
        self.screen.blit(txt, (x, y))
        bw, bh = 28, 26
        minus = pygame.Rect(x + 148, y - 2, bw, bh)
        plus  = pygame.Rect(x + 180, y - 2, bw, bh)
        for r, lbl in ((minus, '-'), (plus, '+')):
            col = C_BUTTON_HOVER if r.collidepoint(pygame.mouse.get_pos()) else C_BUTTON
            pygame.draw.rect(self.screen, col, r, border_radius=4)
            t = self.font_ui_b.render(lbl, True, C_TEXT)
            self.screen.blit(t, t.get_rect(center=r.center))
        if tag == 'cw':
            self._clock_minus_wt, self._clock_plus_wt = minus, plus
        else:
            self._clock_minus_bt, self._clock_plus_bt = minus, plus

    def _draw_level_selector(self, x, y, label, current, side):
        txt = self.font_small.render(label + f": {current}", True, C_TEXT)
        self.screen.blit(txt, (x, y))
        bw, bh = 28, 26
        minus = pygame.Rect(x + 130, y - 2, bw, bh)
        plus  = pygame.Rect(x + 162, y - 2, bw, bh)
        for r, lbl in ((minus, '-'), (plus, '+')):
            col = C_BUTTON_HOVER if r.collidepoint(pygame.mouse.get_pos()) else C_BUTTON
            pygame.draw.rect(self.screen, col, r, border_radius=4)
            t = self.font_ui_b.render(lbl, True, C_TEXT)
            self.screen.blit(t, t.get_rect(center=r.center))
        # Store rects for click handling
        if side == 'w':
            self._level_minus_w = minus
            self._level_plus_w  = plus
        else:
            self._level_minus_b = minus
            self._level_plus_b  = plus

    # ── Game events ───────────────────────────────────────────────────────────
    def _handle_menu_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            # Level buttons
            if hasattr(self, '_level_minus_w') and self._level_minus_w.collidepoint(pos):
                self.ai_level_w = max(1, self.ai_level_w - 1)
            elif hasattr(self, '_level_plus_w') and self._level_plus_w.collidepoint(pos):
                self.ai_level_w = min(10, self.ai_level_w + 1)
            if hasattr(self, '_level_minus_b') and self._level_minus_b.collidepoint(pos):
                self.ai_level_b = max(1, self.ai_level_b - 1)
            elif hasattr(self, '_level_plus_b') and self._level_plus_b.collidepoint(pos):
                self.ai_level_b = min(10, self.ai_level_b + 1)
            if hasattr(self, '_clock_minus_wt') and self._clock_minus_wt.collidepoint(pos):
                self.clock_w_min = self._cycle_clock_preset(self.clock_w_min, -1)
            elif hasattr(self, '_clock_plus_wt') and self._clock_plus_wt.collidepoint(pos):
                self.clock_w_min = self._cycle_clock_preset(self.clock_w_min, +1)
            if hasattr(self, '_clock_minus_bt') and self._clock_minus_bt.collidepoint(pos):
                self.clock_b_min = self._cycle_clock_preset(self.clock_b_min, -1)
            elif hasattr(self, '_clock_plus_bt') and self._clock_plus_bt.collidepoint(pos):
                self.clock_b_min = self._cycle_clock_preset(self.clock_b_min, +1)
            for name, btn in self._menu_buttons.items():
                if btn.is_clicked(event):
                    if name in self.MODES:
                        self.mode = name
                    elif name == 'Start':
                        self.new_game()

    def _handle_game_event(self, event):
        if self._btn_new.is_clicked(event):
            self._invalidate_ai_sessions()
            self._screen_state = 'menu'
            return
        if self._btn_flip.is_clicked(event):
            self.flipped = not self.flipped
            return
        if self._btn_resign.is_clicked(event):
            winner = 'black' if self.state.turn == WHITE else 'white'
            self._game_over(winner + ' wins (resign)')
            return
        if self._btn_pgn.is_clicked(event):
            if self._copy_to_clipboard(self._build_pgn()):
                self._pgn_copied_until = pygame.time.get_ticks() + 2000
            return

        if self._promo_move is not None:
            self._handle_promo_event(event)
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            s = pixel_to_sq(*event.pos, flipped=self.flipped)
            if s >= 0:
                self._handle_board_click(s)

    def _handle_board_click(self, s):
        if self.animating: return

        state  = self.state
        color  = state.turn
        is_human = self._is_human_turn()
        if not is_human:
            return

        piece_here = state.board[s]

        if self.selected_sq == -1:
            # Select a piece
            if piece_here != EMPTY and piece_color(piece_here) == color:
                self.selected_sq = s
                self.legal_moves = state.get_legal_moves_for(s)
        else:
            # Try to make a move
            move = self._find_move(self.selected_sq, s)
            if move is not None:
                self._execute_move(move)
            else:
                # Clicked a square that isn't a legal destination
                if piece_here != EMPTY and piece_color(piece_here) == color:
                    # Reselect own piece
                    self.selected_sq = s
                    self.legal_moves = state.get_legal_moves_for(s)
                else:
                    # Truly illegal — play feedback sound
                    self._play("illegal")
                    self.selected_sq = -1
                    self.legal_moves = []

    def _find_move(self, from_sq, to_sq):
        """Find legal move from from_sq to to_sq. Returns None if not found."""
        candidates = [m for m in self.legal_moves if m.from_sq == from_sq and m.to_sq == to_sq]
        if not candidates:
            return None
        # If promotion, pick queen by default (will open dialog)
        promo_cands = [m for m in candidates if m.promotion]
        if promo_cands:
            # Store all promo options and open dialog
            self._promo_pending = promo_cands
            self._promo_to_sq   = to_sq
            self._promo_from_sq = from_sq
            self._promo_move    = promo_cands[0]   # placeholder
            return None   # defer until promotion chosen
        return candidates[0]

    def _handle_promo_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            chosen = self._promo_hit_test(event.pos)
            if chosen is not None:
                move = next((m for m in self._promo_pending if m.promotion == chosen), None)
                self._promo_move = None
                if move:
                    self._execute_move(move)

    def _promo_hit_test(self, pos):
        """Returns piece type if click lands on promotion dialog, else None."""
        options = [QUEEN, ROOK, BISHOP, KNIGHT]
        bw, bh  = SQ_SIZE, SQ_SIZE
        total_w = len(options) * bw
        x0 = MARGIN + (BOARD_PX - total_w) // 2
        y0 = MARGIN + BOARD_PX // 2 - bh // 2
        for i, pt in enumerate(options):
            r = pygame.Rect(x0 + i * bw, y0, bw, bh)
            if r.collidepoint(pos):
                return pt
        return None

    def _execute_move(self, move):
        """Apply a validated legal move to the game state."""
        self.animating = True
        self.anim_piece = self.state.board[move.from_sq]
        self.anim_start_sq = move.from_sq
        self.anim_end_sq = move.to_sq
        self.anim_progress = 0.0

        state_before = self.state
        self.state   = self.state.make_move(move)
        san          = san_move(state_before, move, self.state)
        self.last_move    = move
        self.selected_sq  = -1
        self.legal_moves  = []
        self.move_history.append((state_before, move, san))

        # Track repetition
        key = self.state.position_key()
        self.pos_history[key] = self.pos_history.get(key, 0) + 1

        # ── Sound ─────────────────────────────────────────────────────────────
        result = self.state.game_result()
        if result in ('white', 'black'):
            pass  # checkmate sound played in _game_over_from_result
        elif move.captured or move.is_ep:
            self._play("capture")
        elif self.state.is_in_check():
            self._play("check")
        else:
            self._play("move")

        # Check game over
        if result:
            self._game_over_from_result(result)
            return
        if self.pos_history.get(key, 0) >= 3:
            self._game_over('Draw by repetition')
            return

    def _game_over_from_result(self, result):
        msgs = {'white': 'White wins!', 'black': 'Black wins!', 'draw': 'Draw!'}
        # Determine win/lose from the human's perspective (or White if AI vs AI)
        if result in ('white', 'black'):
            human_color = None
            if self.mode == 'Human vs AI (White)':
                human_color = WHITE
            elif self.mode == 'Human vs AI (Black)':
                human_color = BLACK
            winner_color = WHITE if result == 'white' else BLACK
            if human_color is not None:
                if winner_color == human_color:
                    self._play("checkmate_win")
                else:
                    self._play("checkmate_lose")
            else:
                self._play("checkmate_win")  # AI vs AI — just play win sound
        self._game_over(msgs.get(result, result))

    def _game_over(self, text):
        self._invalidate_ai_sessions()
        self._ai_result      = None
        self._result_text    = text
        self._screen_state   = 'gameover'
        self._ai_thinking    = False

    def _handle_gameover_event(self, event):
        if self._btn_new.is_clicked(event):
            self.new_game()
            return
        if self._btn_pgn.is_clicked(event):
            if self._copy_to_clipboard(self._build_pgn()):
                self._pgn_copied_until = pygame.time.get_ticks() + 2000
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if event.pos[0] < BOARD_PX + MARGIN * 2:
                self._invalidate_ai_sessions()
                self._screen_state = 'menu'

    # ── AI logic ──────────────────────────────────────────────────────────────
    def _is_human_turn(self):
        turn = self.state.turn
        if self.mode == 'Human vs Human':
            return True
        if self.mode == 'Human vs AI (White)':
            return turn == WHITE
        if self.mode == 'Human vs AI (Black)':
            return turn == BLACK
        return False   # AI vs AI

    def _is_ai_turn(self):
        return not self._is_human_turn()

    def _get_current_agent(self):
        return self.agent_white if self.state.turn == WHITE else self.agent_black

    def _maybe_spawn_ponder(self, now_ticks: int) -> None:
        """Cheap TT warmup on the opponent's clock (Human vs AI only)."""
        if self.mode in ('AI vs AI', 'Human vs Human'):
            return
        if self.animating or self._promo_move is not None or self._ai_thinking:
            return
        if not self._is_human_turn():
            return

        pt = getattr(self, '_ponder_thread', None)
        if pt is not None and pt.is_alive():
            return
        if now_ticks - self._last_ponder_ms < 38:
            return

        pondering_agent = self.agent_black if self.state.turn == WHITE else self.agent_white

        try:
            from agents_v4 import MinimaxAgent
        except ImportError:
            return
        if not isinstance(pondering_agent, MinimaxAgent):
            return

        epoch_frozen = self._ponder_epoch
        rem_bot = (
            self.clock_remaining_b
            if pondering_agent is self.agent_black
            else self.clock_remaining_w
        )
        wall = self._ponder_budget_s_for_side(rem_bot)
        fen_snap = self.state.to_fen()
        evt = self._ai_stop_event

        def _run_pe():
            if epoch_frozen != self._ponder_epoch:
                return
            if evt is not None and getattr(evt, "is_set", lambda: False)():
                return
            from position import Position

            pos = Position.from_fen(fen_snap)
            try:
                pondering_agent.ponder(pos, time_limit_s=wall, stop_event=evt)
            except Exception:
                pass

        self._last_ponder_ms = now_ticks
        self._ponder_thread = threading.Thread(target=_run_pe, daemon=True)
        self._ponder_thread.start()

    def _tick_game(self):
        """Called every frame; triggers AI move if it's the AI's turn."""
        now = pygame.time.get_ticks()
        dt = (now - self._last_frame_ticks) / 1000.0
        self._last_frame_ticks = now

        if self.animating:
            self.anim_progress += dt / self.anim_duration
            if self.anim_progress >= 1.0:
                self.animating = False
            else:
                return

        if self._screen_state != 'game':
            return

        if self._promo_move is not None:
            return

        self._tick_clock(dt)

        if self._screen_state != 'game':
            return

        # Collect result from AI thread (ignore if session ended mid-search)
        if self._ai_thinking and self._ai_thread and not self._ai_thread.is_alive():
            self._ai_thinking = False
            if self._screen_state != 'game':
                self._ai_result = None
                return
            if self._ai_result is not None:
                self._execute_move(self._ai_result)
                self._ai_result  = None
            return

        if self._ai_thinking:
            return

        self._maybe_spawn_ponder(now)

        if not self._is_ai_turn():
            return

        # AI vs AI: add a small delay for readability (clocks tick — models live thinking)
        if self.mode == 'AI vs AI' and now - self._last_ai_tick < self.ai_delay_ms:
            return
        self._last_ai_tick = now

        agent = self._get_current_agent()
        state = self.state
        # Online-style slice of remaining clock (full pool stays on the GUI; search uses tighter budget)
        rem_stm = self._clock_remaining_stm()
        tl = self._online_style_move_budget_s(rem_stm)
        g = self._begin_ai_turn_search()

        def _run():
            import random as _rnd
            from position import Position

            pos = Position.from_fen(state.to_fen())
            pm = agent.choose_move(
                pos,
                time_limit_s=tl,
                stop_event=self._ai_stop_event,
                repetition_base=None,
            )
            mv = self._uci_move_to_gs_move(state, pm.uci() if pm else None)
            if mv is None:
                lm = state.get_legal_moves()
                if lm:
                    mv = _rnd.choice(lm)
            if g == self._ai_generation:
                self._ai_result = mv

        self._ai_thinking = True
        self._ai_result    = None
        self._ai_thread    = threading.Thread(target=_run, daemon=True)
        self._ai_thread.start()

    # ── PGN export ────────────────────────────────────────────────────────────
    def _build_pgn(self) -> str:
        """Build a PGN string from the current game."""
        from datetime import date
        today = date.today().strftime('%Y.%m.%d')

        if self.mode == 'Human vs Human':
            white_name, black_name = 'Human', 'Human'
        elif self.mode == 'Human vs AI (White)':
            white_name, black_name = 'Human', self.agent_black.name
        elif self.mode == 'Human vs AI (Black)':
            white_name, black_name = self.agent_white.name, 'Human'
        else:
            white_name = self.agent_white.name
            black_name = self.agent_black.name

        txt = self._result_text
        if 'White wins' in txt:   result = '1-0'
        elif 'Black wins' in txt: result = '0-1'
        elif 'Draw' in txt or 'draw' in txt: result = '1/2-1/2'
        else:                     result = '*'

        header = (
            '[Event "AI Chess Assignment"]\n'
            '[Site "Local"]\n'
            f'[Date "{today}"]\n'
            '[Round "1"]\n'
            f'[White "{white_name}"]\n'
            f'[Black "{black_name}"]\n'
            f'[Result "{result}"]\n'
            '\n'
        )

        # Build move tokens, wrapping at ~80 chars
        tokens = []
        for i, (_, _, san) in enumerate(self.move_history):
            if i % 2 == 0:
                tokens.append(f'{i // 2 + 1}. {san}')
            else:
                tokens.append(san)
        tokens.append(result)

        lines, cur = [], ''
        for tok in tokens:
            if cur and len(cur) + 1 + len(tok) > 80:
                lines.append(cur)
                cur = tok
            else:
                cur = (cur + ' ' + tok).strip()
        if cur:
            lines.append(cur)

        return header + '\n'.join(lines)

    def _copy_to_clipboard(self, text: str) -> bool:
        """Copy text to the system clipboard. Returns True on success."""
        import subprocess
        try:
            if sys.platform == 'win32':
                proc = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
                proc.communicate(input=text.encode('utf-8'))
                return True
            elif sys.platform == 'darwin':
                proc = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
                proc.communicate(input=text.encode('utf-8'))
                return True
            else:   # Linux
                proc = subprocess.Popen(
                    ['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
                proc.communicate(input=text.encode('utf-8'))
                return True
        except Exception:
            pass
        # Fallback: pyperclip (if installed)
        try:
            import pyperclip
            pyperclip.copy(text)
            return True
        except Exception:
            return False

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw_game(self):
        self._draw_board()
        self._draw_pieces()
        self._draw_overlays()
        self._draw_sidebar()
        if self._promo_move is not None:
            self._draw_promo_dialog()
        # Buttons
        f = self.font_ui
        self._btn_new.draw(self.screen, f)
        self._btn_flip.draw(self.screen, f)
        self._btn_resign.draw(self.screen, f)
        self._btn_pgn.draw(self.screen, f)
        # "Copied!" feedback
        if pygame.time.get_ticks() < self._pgn_copied_until:
            fb = self.font_small.render("✓ Copied to clipboard!", True, (80, 220, 120))
            self.screen.blit(fb, fb.get_rect(
                centerx=self._btn_pgn.rect.centerx,
                top=self._btn_pgn.rect.bottom + 4))

    def _draw_board(self):
        for r in range(8):
            for c in range(8):
                color = C_LIGHT if (r + c) % 2 == 0 else C_DARK
                x = MARGIN + c * SQ_SIZE
                y = MARGIN + r * SQ_SIZE
                if self.flipped:
                    x = MARGIN + (7 - c) * SQ_SIZE
                    y = MARGIN + (7 - r) * SQ_SIZE
                pygame.draw.rect(self.screen, color, (x, y, SQ_SIZE, SQ_SIZE))

        # Coordinates
        for i in range(8):
            # File labels (a-h)
            fl = self.font_small.render('abcdefgh'[i if not self.flipped else 7-i], True, C_TEXT_DIM)
            self.screen.blit(fl, (MARGIN + i * SQ_SIZE + SQ_SIZE - 12,
                                  MARGIN + BOARD_PX - 14))
            # Rank labels (1-8)
            rl = self.font_small.render(str(8 - i if not self.flipped else i + 1), True, C_TEXT_DIM)
            self.screen.blit(rl, (MARGIN + 2, MARGIN + i * SQ_SIZE + 2))

    def _draw_overlays(self):
        surf = self._overlay_surf
        surf.fill((0, 0, 0, 0))

        # Last move highlight
        if self.last_move:
            for s in (self.last_move.from_sq, self.last_move.to_sq):
                r, c = rc(s)
                if self.flipped:
                    r, c = 7-r, 7-c
                pygame.draw.rect(surf, C_LAST_MOVE, (c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

        # King in check
        if self.state.is_in_check():
            k = self.state.find_king(self.state.turn)
            if k >= 0:
                r, c = rc(k)
                if self.flipped:
                    r, c = 7-r, 7-c
                pygame.draw.rect(surf, C_CHECK, (c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

        # Selected square
        if self.selected_sq >= 0:
            r, c = rc(self.selected_sq)
            if self.flipped:
                r, c = 7-r, 7-c
            pygame.draw.rect(surf, C_SELECTED, (c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

        # Legal move dots
        for move in self.legal_moves:
            r, c = rc(move.to_sq)
            if self.flipped:
                r, c = 7-r, 7-c
            cx = c * SQ_SIZE + SQ_SIZE // 2
            cy = r * SQ_SIZE + SQ_SIZE // 2
            if move.captured or move.is_ep:
                pygame.draw.circle(surf, C_LEGAL, (cx, cy), SQ_SIZE // 2 - 4, 5)
            else:
                pygame.draw.circle(surf, C_LEGAL, (cx, cy), SQ_SIZE // 5)

        self.screen.blit(surf, (MARGIN, MARGIN))

    def _draw_pieces(self):
        board = self.state.board

        # 1. Draw stationary pieces
        for s in range(64):
            # Skip the destination square if we are currently animating a piece there
            if self.animating and s == self.anim_end_sq:
                continue

            p = board[s]
            if p == EMPTY:
                continue
            r, c = rc(s)
            if self.flipped:
                r, c = 7-r, 7-c
            glyph  = _glyph(p)
            fg     = C_WHITE_PIECE if piece_color(p) == WHITE else C_BLACK_PIECE
            outline= C_BLACK_PIECE if piece_color(p) == WHITE else C_WHITE_PIECE
            cx     = MARGIN + c * SQ_SIZE + SQ_SIZE // 2
            cy     = MARGIN + r * SQ_SIZE + SQ_SIZE // 2

            for dx, dy in ((-1,-1),(-1,1),(1,-1),(1,1)):
                ot = self.font_piece.render(glyph, True, outline)
                self.screen.blit(ot, ot.get_rect(center=(cx+dx, cy+dy)))
            ft = self.font_piece.render(glyph, True, fg)
            self.screen.blit(ft, ft.get_rect(center=(cx, cy)))

        # 2. Draw the animating piece on top
        if self.animating:
            p = self.anim_piece
            r, c = rc(self.anim_start_sq)
            er, ec = rc(self.anim_end_sq)
            if self.flipped:
                r, c = 7-r, 7-c
                er, ec = 7-er, 7-ec

            start_x = MARGIN + c * SQ_SIZE + SQ_SIZE // 2
            start_y = MARGIN + r * SQ_SIZE + SQ_SIZE // 2
            end_x = MARGIN + ec * SQ_SIZE + SQ_SIZE // 2
            end_y = MARGIN + er * SQ_SIZE + SQ_SIZE // 2

            cx = start_x + (end_x - start_x) * self.anim_progress
            cy = start_y + (end_y - start_y) * self.anim_progress

            glyph  = _glyph(p)
            fg     = C_WHITE_PIECE if piece_color(p) == WHITE else C_BLACK_PIECE
            outline= C_BLACK_PIECE if piece_color(p) == WHITE else C_WHITE_PIECE

            for dx, dy in ((-1,-1),(-1,1),(1,-1),(1,1)):
                ot = self.font_piece.render(glyph, True, outline)
                self.screen.blit(ot, ot.get_rect(center=(cx+dx, cy+dy)))
            ft = self.font_piece.render(glyph, True, fg)
            self.screen.blit(ft, ft.get_rect(center=(cx, cy)))

    def _draw_sidebar(self):
        bx = BOARD_PX + MARGIN * 2
        # Background
        pygame.draw.rect(self.screen, C_SIDEBAR_BG,
                         (bx - 4, 0, SIDEBAR_W + 8, WIN_H))

        # Turn indicator
        turn_lbl = "White to move" if self.state.turn == WHITE else "Black to move"
        color_dot = C_WHITE_PIECE if self.state.turn == WHITE else (100, 100, 100)
        pygame.draw.circle(self.screen, color_dot, (bx + 10, 190), 8)
        t = self.font_ui_b.render(turn_lbl, True, C_TEXT)
        self.screen.blit(t, (bx + 24, 182))

        def _clk_line(label, sec):
            m, s = int(max(0, sec)) // 60, int(max(0, sec)) % 60
            return f"{label} {m}:{s:02d}"

        tc = self.font_small.render(_clk_line("W clock", self.clock_remaining_w), True, C_TEXT)
        self.screen.blit(tc, (bx, 208))
        tc2 = self.font_small.render(_clk_line("B clock", self.clock_remaining_b), True, C_TEXT)
        self.screen.blit(tc2, (bx + 138, 208))

        # Mode & level info
        by = 232
        t = self.font_small.render(f"Mode: {self.mode}", True, C_TEXT_DIM)
        self.screen.blit(t, (bx, by))
        by += 18
        if self.mode != 'Human vs Human':
            if self.mode in ('Human vs AI (Black)', 'AI vs AI'):
                t = self.font_small.render(f"White AI: {self.agent_white.name}", True, C_TEXT_DIM)
                self.screen.blit(t, (bx, by)); by += 18
            if self.mode in ('Human vs AI (White)', 'AI vs AI'):
                t = self.font_small.render(f"Black AI: {self.agent_black.name}", True, C_TEXT_DIM)
                self.screen.blit(t, (bx, by)); by += 18

        # AI thinking indicator
        if self._ai_thinking:
            anim = '.' * ((pygame.time.get_ticks() // 400) % 4)
            t = self.font_ui.render(f"Thinking{anim}", True, C_ACCENT)
            self.screen.blit(t, (bx, by + 4))
        by += 28

        # Move history
        t = self.font_ui_b.render("Move History", True, C_TEXT)
        self.screen.blit(t, (bx, by)); by += 22
        pygame.draw.line(self.screen, C_TEXT_DIM, (bx, by), (bx + SIDEBAR_W - MARGIN, by))
        by += 6

        line_h   = 18
        max_disp = (WIN_H - by - MARGIN) // line_h
        start    = max(0, len(self.move_history) - max_disp)
        for i in range(start, len(self.move_history)):
            state_b, move, san = self.move_history[i]
            move_num = (i // 2) + 1
            if i % 2 == 0:
                label = f"{move_num}. {san}"
            else:
                label = f"   ... {san}"
            col = C_TEXT if i == len(self.move_history) - 1 else C_TEXT_DIM
            t   = self.font_small.render(label, True, col)
            self.screen.blit(t, (bx, by))
            by += line_h

    def _draw_promo_dialog(self):
        """Draw a promotion piece-selection dialog centered on the board."""
        options = [QUEEN, ROOK, BISHOP, KNIGHT]
        bw, bh  = SQ_SIZE, SQ_SIZE
        total_w = len(options) * bw
        x0 = MARGIN + (BOARD_PX - total_w) // 2
        y0 = MARGIN + BOARD_PX // 2 - bh // 2

        # Background
        bg = pygame.Rect(x0 - 8, y0 - 36, total_w + 16, bh + 52)
        pygame.draw.rect(self.screen, (30, 30, 30, 230), bg, border_radius=10)
        pygame.draw.rect(self.screen, C_ACCENT, bg, 2, border_radius=10)

        lbl = self.font_ui_b.render("Promote to:", True, C_ACCENT)
        self.screen.blit(lbl, lbl.get_rect(center=(x0 + total_w // 2, y0 - 16)))

        color = self.state.turn  # the pawn's color (before the move)
        # Deduce pawn color from pending move
        pawn_color = piece_color(self._promo_pending[0].from_sq and
                                  self.move_history and
                                  self.move_history[-1][0].board[self._promo_from_sq] or
                                  WHITE)
        # Safer: look up the original state
        if self.move_history:
            pawn_color = piece_color(self.move_history[-1][0].board[self._promo_from_sq]) if \
                self.move_history else WHITE
        else:
            pawn_color = self.state.turn  # before move was recorded

        mouse = pygame.mouse.get_pos()
        for i, pt in enumerate(options):
            x = x0 + i * bw
            rect = pygame.Rect(x, y0, bw, bh)
            bg_col = (70, 70, 90) if rect.collidepoint(mouse) else (50, 50, 70)
            pygame.draw.rect(self.screen, bg_col, rect, border_radius=6)
            pygame.draw.rect(self.screen, C_TEXT_DIM, rect, 1, border_radius=6)
            # Piece glyph – use the actual moving side's color
            p = pt * (WHITE if pawn_color == WHITE else 1)
            # Build piece value for glyph lookup
            glyph_color = WHITE if self.state.turn == BLACK else BLACK  # turn already flipped
            g = _GLYPH.get((pt, glyph_color), '?')
            fg = C_WHITE_PIECE if glyph_color == WHITE else C_BLACK_PIECE
            outline = C_BLACK_PIECE if glyph_color == WHITE else C_WHITE_PIECE
            cx, cy = rect.centerx, rect.centery
            for dx, dy in ((-1,-1),(-1,1),(1,-1),(1,1)):
                ot = self.font_piece.render(g, True, outline)
                self.screen.blit(ot, ot.get_rect(center=(cx+dx, cy+dy)))
            ft = self.font_piece.render(g, True, fg)
            self.screen.blit(ft, ft.get_rect(center=(cx, cy)))

    def _draw_gameover_overlay(self):
        overlay = pygame.Surface((BOARD_PX + MARGIN * 2, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        t1 = self.font_big.render("Game Over", True, C_ACCENT)
        t2 = self.font_title.render(self._result_text, True, C_TEXT)
        t3 = self.font_ui.render("Click anywhere to return to menu", True, C_TEXT_DIM)

        cx = (BOARD_PX + MARGIN * 2) // 2
        cy = WIN_H // 2
        self.screen.blit(t1, t1.get_rect(center=(cx, cy - 50)))
        self.screen.blit(t2, t2.get_rect(center=(cx, cy + 10)))
        self.screen.blit(t3, t3.get_rect(center=(cx, cy + 60)))


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    gui = ChessGUI()
    gui.run()


if __name__ == '__main__':
    main()
