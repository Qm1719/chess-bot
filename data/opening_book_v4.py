"""opening_book_v4.py — v4-only opening book with randomized choice.

Keep v3 deterministic/pure, but allow v4 (NNUE experiments) to diversify openings
to avoid playing the same line every game.

Extra lines (e.g. from your games) live in `data/opening_lines.sqlite3`; they are
merged in when the file changes — no engine restart.
"""

from __future__ import annotations

import os
import sys
import secrets
from typing import Dict, List

_V5_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENGINE_DIR = os.path.join(_V5_ROOT, "engine")
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from position import Position, Move  # type: ignore


def _book_key_from_fen(fen: str) -> str:
    # Same matching policy as v3: ignore clocks and EP square differences.
    parts = fen.split()
    return " ".join(parts[:3])


def _book_key(pos: Position) -> str:
    return _book_key_from_fen(pos.to_fen())


def _uci_to_move(pos: Position, uci: str) -> Move | None:
    for m in pos.legal_moves():
        if m.uci() == uci:
            return m
    return None


def _build_book(lines: List[List[str]]) -> Dict[str, List[str]]:
    book: Dict[str, List[str]] = {}
    for line in lines:
        pos = Position.from_fen(Position.START_FEN)
        for uci in line:
            k = _book_key(pos)
            book.setdefault(k, [])
            if uci not in book[k]:
                book[k].append(uci)
            mv = _uci_to_move(pos, uci)
            if mv is None:
                break
            pos.push(mv)
    return book


# Copied from submission v3/opening_book.py
_LINES: List[List[str]] = [
    # e4 e5
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1", "f6e4", "f1e1", "e4d6"], # C67 Ruy Lopez: Berlin Defense, Rio Gambit Accepted
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"], # C84 Ruy Lopez: Closed
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "b7b5", "a4b3", "c8b7"], # C84 Ruy Lopez: Closed
    ["e2e4", "e7e5", "g1f3", "g8f6", "f3e5", "d7d6", "e5f3", "f6e4", "d2d4", "d6d5"], # C42 Petrov's Defense: Classical Attack
    ["e2e4", "e7e5", "g1f3", "g8f6", "f3e5", "f6e4", "d1e2", "d8e7", "e2e4", "d7d6"], # C42 Petrov's Defense: Damiano Variation, Kholmov Gambit
    ["e2e4", "e7e5", "g1f3", "d7d6", "d2d4", "e5d4", "f3d4", "g8f6", "b1c3", "b8c6"], # C41 Philidor Defense: Exchange Variation
    ["e2e4", "e7e5", "g1f3", "d7d6", "d2d4", "e5d4", "f3d4", "f8e7", "f1d3", "g8f6"], # C41 Philidor Defense: Exchange Variation
    ["e2e4", "e7e5", "g1f3", "d7d6", "d2d4", "b8d7", "f1c4", "e5d4", "f3d4", "g8f6"], # C41 Philidor Defense: Hanham Variation
    ["e2e4", "e7e5", "g1f3", "d7d6", "d2d4", "g8f6", "d4e5", "f6e4", "d1d5", "e4c5"], # C41 Philidor Defense: Nimzowitsch Variation, Rellstab Variation
    
    # e4 c5
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"], # B90 Sicilian Defense: Najdorf Variation
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "g8f6", "b1c3", "c5d4", "f3d4", "a7a6"], # B50 Sicilian Defense: Modern Variations, Anti-Qxd4 Move Order
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "d7d6"], # B56 Sicilian Defense: Classical Variation
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g7g6", "c2c4", "g8f6"], # B36 Sicilian Defense: Accelerated Dragon, Maróczy Bind
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "e7e6", "b1c3", "a7a6"], # B46 Sicilian Defense: Taimanov Variation
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "e7e5", "d4b5", "d7d6"], # B32 Sicilian Defense: Kalashnikov Variation
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c6d4", "f3d4", "c5d4", "d1d4", "e7e6"], # B32 Sicilian Defense: Open
    ["e2e4", "c7c5", "g1f3", "e7e6", "b1c3", "a7a6", "d2d4", "c5d4", "f3d4", "d8c7"], # B43 Sicilian Defense: Kan Variation, Knight Variation
    ["e2e4", "c7c5", "g1f3", "e7e6", "b1c3", "a7a6", "d2d4", "b7b5", "d4d5", "d7d6"], # B40 Sicilian Defense: French Variation
    ["e2e4", "c7c5", "g1f3", "e7e6", "b1c3", "b8c6", "d2d4", "c5d4", "f3d4", "a7a6"], # B46 Sicilian Defense: Taimanov Variation line 71
    ["e2e4", "c7c5", "g1f3", "e7e6", "b1c3", "b8c6", "d2d4", "c6d4", "f3d4", "c5d4"], # B40 Sicilian Defense: French Variation
    ["e2e4", "c7c5", "g1f3", "e7e6", "b1c3", "d7d6", "d2d4", "c5d4", "f3d4", "a7a6"], # B40 Sicilian Defense: French Variation
    
    # e4 e6
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6d7", "f2f4", "c7c5"], # C11 French Defense: Classical Variation, Steinitz Variation
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6e4", "c3e4", "d5e4"], # C11 French Defense: Classical Variation, Steinitz Variation
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5", "a2a3", "b4c3"], # C18 French Defense: Winawer Variation, Advance Variation
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5", "a2a3", "b4a5"], # C17 French Defense: Winawer Variation, Advance Variation
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "b8d7", "g1f3", "g8f6"], # C10 French Defense: Rubinstein Variation, Blackburne Defense
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8d7", "g1f3", "d7c6"], # C10 French Defense: Rubinstein Variation, Fort Knox Variation
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "g8f6", "e4f6", "d8f6"], # C10 French Defense: Rubinstein Variation
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "e6e5", "g1f3", "g8f6"], # C10 French Defense: Rubinstein Variation, Ellis Gambit
    ["e2e4", "e7e6", "d2d4", "c7c5", "g1f3", "c5d4", "f3d4", "a7a6", "b1c3", "d8c7"], # B43 Sicilian Defense: Kan Variation, Knight Variation line 74
    
    # e4 c6
    ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c8f5", "g1f3", "e7e6", "f1e2", "g8e7"], # B12 Caro-Kann Defense: Advance Variation, Short Variation
    ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c6c5", "g1f3", "c5d4", "f3d4", "b8c6"], # B12 Caro-Kann Defense: Advance Variation, Botvinnik-Carls Defense
    ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c6c5", "g1f3", "b8c6", "d4c5", "c8g4"], # B12 Caro-Kann Defense: Advance Variation, Botvinnik-Carls Defense
    
    # e4 d6
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "c1e3", "f8g7", "d1d2", "c7c6"], # B07 Pirc Defense
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "c1e3", "c7c6", "d1d2", "b7b5"], # B07 Pirc Defense
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "e7e5", "g1f3", "b8d7", "f1c4", "f8e7"], # C41 Philidor Defense: Lion Variation
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "b8d7", "f2f4", "e7e5", "g1f3", "e5d4"], # B07 Lion Defense: Anti-Philidor, Lion's Cave
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "c7c6", "f2f4", "d8a5", "f1d3", "e7e5"], # B07 Czech Defense
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "c7c6", "f2f4", "g7g6", "g1f3", "f8g7"], # B07 Czech Defense
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "c7c6", "f2f4", "h7h5", "g1f3", "d6d5"], # B07 Czech Defense
    
    # e4 g6
    ["e2e4", "g7g6", "d2d4", "f8g7", "b1c3", "d7d6", "c1e3", "a7a6", "d1d2", "b7b5"], # B06 Modern Defense: Standard Defense
    ["e2e4", "g7g6", "d2d4", "f8g7", "b1c3", "d7d6", "c1e3", "g8f6", "d1d2", "c7c6"], # B06 Modern Defense: Standard Defense
    ["e2e4", "g7g6", "d2d4", "f8g7", "b1c3", "c7c6", "f2f4", "d7d5", "e4e5", "h7h5"], # B06 Modern Defense: Gurgenidze Defense
    ["e2e4", "g7g6", "d2d4", "f8g7", "b1c3", "c7c6", "f2f4", "h7h5", "e4e5", "g8h6"], # B06 Modern Defense: Standard Line
    ["e2e4", "g7g6", "d2d4", "f8g7", "b1c3", "c7c6", "f2f4", "d7d6", "g1f3", "h7h5"], # B06 Modern Defense: Standard Line
    ["e2e4", "g7g6", "d2d4", "f8g7", "b1c3", "c7c6", "f2f4", "e7e6", "g1f3", "g8e7"], # B06 Modern Defense: Standard Line
    ["e2e4", "g7g6", "d2d4", "f8g7", "b1c3", "c7c6", "f2f4", "d8b6", "e4e5", "d7d6"], # B06 Modern Defense: Standard Line
    ["e2e4", "g7g6", "d2d4", "d7d6", "b1c3", "f8g7"], # B06 Modern Defense: Standard Defense line 94
    ["e2e4", "g7g6", "d2d4", "d7d6", "b1c3", "g8f6"], # B07 Pirc Defense line 85
    ["e2e4", "g7g6", "d2d4", "d7d6", "b1c3", "a7a6", "f2f4", "f8g7", "g1f3", "b7b5"], # B06 Modern Defense
    ["e2e4", "g7g6", "d2d4", "d7d6", "b1c3", "a7a6", "f2f4", "b7b5", "g1f3", "c8b7"], # B06 Modern Defense
    ["e2e4", "g7g6", "d2d4", "g8f6", "e4e5", "f6h5", "g1f3", "d7d6", "f1c4", "b8c6"], # B06 Modern Defense: Norwegian Defense
    
    # e4 d5
    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4", "g8f6", "g1f3", "c7c6"], # B01 Scandinavian Defense: Main Line, Mieses Variation
    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4", "c7c6", "f1c4", "c8f5"], # B01 Scandinavian Defense: Main Line
    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5d6", "d2d4", "g8f6", "g1f3", "c7c6"], # B01 Scandinavian Defense: Gubinsky-Melts Defense
    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5d8", "d2d4", "g8f6", "g1f3", "g7g6"], # B01 Scandinavian Defense: Valencian Variation
    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5d8", "d2d4", "c7c6", "f1c4", "c8f5"], # B01 Scandinavian Defense: Valencian Variation
    ["e2e4", "d7d5", "e4d5", "g8f6", "g1f3", "d8d5", "b1c3", "d5a5", "d2d4", "c7c6"], # B01 Scandinavian Defense: Modern Variation line 108
    ["e2e4", "d7d5", "e4d5", "g8f6", "g1f3", "d8d5", "b1c3", "d5d6", "d2d4", "c7c6"], # B01 Scandinavian Defense: Gubinsky-Melts Defense line 110
    ["e2e4", "d7d5", "e4d5", "g8f6", "g1f3", "d8d5", "b1c3", "d5d8", "d2d4", "g7g6"], # B01 Scandinavian Defense: Valencian Variation line 111
    ["e2e4", "d7d5", "e4d5", "g8f6", "g1f3", "f6d5", "d2d4", "g7g6", "c2c4", "d5b6"], # B01 Scandinavian Defense: Richter Variation
    ["e2e4", "d7d5", "e4d5", "g8f6", "g1f3", "f6d5", "d2d4", "c8g4", "c2c4", "g4f3"], # B01 Scandinavian Defense: Modern Variation, Gipslis Variation
    ["e2e4", "d7d5", "e4d5", "g8f6", "g1f3", "f6d5", "d2d4", "e7e6", "c2c4", "d5f6"], # B01 Scandinavian Defense: Modern Variation
    
    
    
    # d4 d5
    ["d2d4", "d7d5", "c2c4", "d5c4", "e2e4", "g8f6", "e4e5", "f6d5", "f1c4", "d5b6"], # D20 Queen's Gambit Accepted: Central Variation, Alekhine System
    ["d2d4", "d7d5", "c2c4", "d5c4", "e2e4", "g8f6", "e4e5", "f6d7", "e5e6", "f7e6"], # D20 Queen's Gambit Accepted: Central Variation, Alekhine System
    ["d2d4", "d7d5", "c2c4", "d5c4", "e2e4", "g8f6", "e4e5", "f6d7", "e5e6", "f7e6"], # D20 Queen's Gambit Accepted: Central Variation, Alekhine System
    ["d2d4", "d7d5", "c2c4", "d5c4", "e2e4", "e7e5", "g1f3", "f8b4", "b1c3", "e5d4"], # D20 Queen's Gambit Accepted: Central Variation, McDonnell Defense
    ["d2d4", "d7d5", "c2c4", "d5c4", "e2e4", "e7e5", "g1f3", "e5d4", "f1c4", "b8c6"], # D20 Queen's Gambit Accepted: Central Variation, McDonnell Defense
    ["d2d4", "d7d5", "c2c4", "d5c4", "e2e4", "b7b5", "a2a4", "c7c6", "a4b5", "c6b5"], # D20 Queen's Gambit Accepted: Central Variation, Greco Variation
    ["d2d4", "d7d5", "c2c4", "d5c4", "e2e4", "c7c5", "d4d5", "g8f6", "b1c3", "b7b5"], # D20 Queen's Gambit Accepted: Linares Variation
    ["d2d4", "d7d5", "c2c4", "d5c4", "e2e4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5"], # D20 Queen's Gambit Accepted: Central Variation, Rubinstein Defense
    ["d2d4", "d7d5", "c2c4", "d5c4", "e2e4", "b8c6", "g1f3", "c8g4", "d4d5", "c6e5"], # D20 Queen's Gambit Accepted: Central Variation, Modern Defense
    ["d2d4", "d7d5", "c2c4", "d5c4", "e2e4", "b8c6", "g1f3", "g8f6", "b1c3", "c8g4"], # D20 Queen's Gambit Accepted: Central Variation, Modern Defense
    ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "g8f6", "e2e3", "e7e6", "g1f3", "b8d7"], # D45 Semi-Slav Defense: Normal Variation
    ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "g8f6", "e2e3", "a7a6", "g1f3", "e7e6"], # D45 Semi-Slav Defense: Accelerated Meran Variation
    ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "g8f6", "e2e3", "g7g6", "g1f3", "f8g7"], # D10 Slav Defense
    ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "g8f6", "e2e3", "c8f5", "c4d5", "c6d5"], # D10 Slav Defense
    ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "d5c4", "e2e3", "b7b5", "a2a4", "b5b4"], # D10 Slav Defense
    ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "d5c4", "e2e3", "c8e6", "f1e2", "g8f6"], # D10 Slav Defense
    ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "d5c4", "e2e3", "e7e5", "f1c4", "e5d4"], # D10 Slav Defense
    ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "e7e5", "d4e5", "d5d4", "c3e4", "d8a5"], # D10 Slav Defense: Winawer Countergambit
    ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "e7e6", "e2e3", "g8f6"], # D10 Slav Defense: Normal Variation line 132
    ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "e7e6", "e2e3", "b8d7", "d1c2", "g8f6"], # D31 Semi-Slav Defense: Accelerated Move Order
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c4d5", "e6d5", "c1g5", "f8b4"], # D35 Queen's Gambit Declined: Exchange Variation, Positional Variation
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c4d5", "f6d5", "e2e4", "d5c3"], # D35 Queen's Gambit Declined: Exchange Variation
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "c7c6"], # D31 Semi-Slav Defense: Accelerated Move Order line 141
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "f8e7", "c4d5", "e6d5", "c1f4", "c7c6"], # D31 Queen's Gambit Declined: Charousek Variation
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "g8f6", "a2a3", "b4e7"], # D31 Queen's Gambit Declined: Queen's Knight Variation
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "c7c5", "g1f3", "g8f6"], # D31 Queen's Gambit Declined: Queen's Knight Variation
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "c7c5", "c4d5", "e6d5", "g1f3", "b8c6"], # D32 Tarrasch Defense: Two Knights Variation
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "c7c5", "c4d5", "c5d4", "d1a4", "c8d7"], # D32 Tarrasch Defense: Schara Gambit
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "a7a6", "c4d5", "e6d5", "c1f4", "c7c6"], # D31 Queen's Gambit Declined: Janowski Variation
    ["d2d4", "d7d5", "c2c4", "b8c6", "b1c3", "d5c4", "g1f3", "g8f6", "c1g5", "h7h6"], # D07 Queen's Gambit Declined: Chigorin Defense, Modern Gambit
    ["d2d4", "d7d5", "c2c4", "b8c6", "b1c3", "g8f6", "g1f3", "d5c4"], # D07 Queen's Gambit Declined: Chigorin Defense, Modern Gambit line 151
    ["d2d4", "d7d5", "c2c4", "b8c6", "b1c3", "g8f6", "g1f3", "e7e6", "c1g5", "d5c4"], # D07 Queen's Gambit Declined: Chigorin Defense
    ["d2d4", "d7d5", "c2c4", "b8c6", "b1c3", "g8f6", "g1f3", "c8g4", "c4d5", "g4f3"], # D07 Queen's Gambit Declined: Chigorin Defense
    ["d2d4", "d7d5", "c2c4", "b8c6", "b1c3", "e7e5", "c4d5", "e5d4", "d5c6", "d4c3"], # D07 Queen's Gambit Declined: Chigorin Defense, Tartakower Gambit
    ["d2d4", "d7d5", "c2c4", "b8c6", "b1c3", "e7e5", "c4d5", "c6d4", "e2e3", "d4f5"], # D07 Queen's Gambit Declined: Chigorin Defense, Tartakower Gambit
    ["d2d4", "d7d5", "c2c4", "b8c6", "b1c3", "e7e6", "g1f3", "g8f6"], # D07 Queen's Gambit Declined: Chigorin Defense line 153
    ["d2d4", "d7d5", "c2c4", "b8c6", "b1c3", "e7e6", "g1f3", "f8b4", "c1g5", "b4c3"], # D07 Queen's Gambit Declined: Chigorin Defense
    ["d2d4", "d7d5", "c2c4", "b8c6", "b1c3", "e7e6", "g1f3", "a7a6", "c1g5", "f8e7"], # D07 Queen's Gambit Declined: Chigorin Defense
    ["d2d4", "d7d5", "c2c4", "e7e5", "d4e5", "d5d4", "g1f3", "b8c6", "g2g3", "g8e7"], # D20 Queen's Gambit Accepted: Central Variation, Alekhine System
    ["d2d4", "d7d5", "c2c4", "e7e5", "d4e5", "g8e7", "g1f3", "b8c6", "c4d5", "d8d5"], # D08 Queen's Gambit Declined: Albin Countergambit
    
    # d4 Nf6
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "d7d5", "g2g3", "f8e7", "f1g2", "e8g8"], # E06 Catalan Opening: Closed
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "d7d5", "g2g3", "f8b4", "c1d2", "b4e7"], # E10 Indian Defense: Anti-Nimzo-Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "d7d5", "g2g3", "c7c6", "f1g2", "f8e7"], # E10 Indian Defense: Anti-Nimzo-Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "d7d5", "g2g3", "d5c4", "f1g2", "f8e7"], # E05 Catalan Opening: Open Defense, Classical Line
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3", "c8a6", "d1c2", "c7c5"], # E15 Queen's Indian Defense: Fianchetto Variation, Nimzowitsch Variation
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3", "c8b7", "f1g2", "f8e7"], # E17 Queen's Indian Defense: Traditional Variation
    ["d2d4", "g8f6", "c2c4", "c7c6", "b1c3", "d7d5"], # D45 Semi-Slav Defense: Normal Variation line 145
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8"], # E90 King's Indian Defense: Normal Variation, Rare Defenses
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "e8g8", "g1f3", "d7d6"], # E90 King's Indian Defense: Normal Variation line 184
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5", "e2e4", "d5c3"], # D85 Grünfeld Defense: Exchange Variation
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d6", "e2e4", "f8g7"], # E90 King's Indian Defense: Normal Variation line 184
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d6", "e2e4", "e7e5", "g1f3", "b8d7"], # E61 King's Indian Defense
    ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "b7b5", "c4b5", "a7a6", "e2e3", "g7g6"], # A57 Benko Gambit Accepted: Modern Variation
    ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5", "c4d5", "d7d6"], # A60 Benoni Defense: Modern Variation
    ["d2d4", "g8f6", "c2c4", "d7d6", "b1c3", "c8g4", "h2h3", "g4h5", "g2g4", "h5g6"], # A53 Old Indian Defense
]

# Built-in book (in-memory). Additional lines load from `data/opening_lines.sqlite3` and merge
# when the DB file mtime changes (no process restart). See `opening_lines_db.add_line` and
# `tools/add_opening_line_from_pgn.py`.
_STATIC_BOOK: Dict[str, List[str]] = _build_book(_LINES)

_SQL_MTIME: float = -1.0
_MERGED_CACHE: Dict[str, List[str]] | None = None


def _merge_line_dicts(base: Dict[str, List[str]], extra: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {k: list(v) for k, v in base.items()}
    for k, v in extra.items():
        L = out.setdefault(k, [])
        for u in v:
            if u not in L:
                L.append(u)
    return out


def _get_merged_opening_book() -> Dict[str, List[str]]:
    """Main opening map: static _LINES + SQLite lines; reloads when `opening_lines.sqlite3` changes."""
    global _SQL_MTIME, _MERGED_CACHE
    try:
        import opening_lines_db as _old_lines  # type: ignore
    except Exception:
        return _STATIC_BOOK
    p = _old_lines.db_path()
    mtime = os.path.getmtime(p) if os.path.isfile(p) else 0.0
    if _MERGED_CACHE is not None and mtime == _SQL_MTIME:
        return _MERGED_CACHE
    _SQL_MTIME = mtime
    try:
        _old_lines.ensure_db()
        sql_lines = _old_lines.load_all_lines() if mtime else []
    except Exception:
        sql_lines = []
    extra: Dict[str, List[str]] = _build_book(sql_lines) if sql_lines else {}
    _MERGED_CACHE = _merge_line_dicts(_STATIC_BOOK, extra)
    return _MERGED_CACHE

# Temporary: if we get Black and White doesn't start with 1.e4/1.d4, use a tiny
# reply book to avoid awkward early play.
_BLACK_REPLY_LINES_NON_E4D4: List[List[str]] = [
    # 1. c4
    ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "d7d5", "c4d5", "f6d5"],
    ["c2c4", "g8f6", "b1c3", "e7e6", "g1f3", "d7d5", "d2d4", "f8b4", "c1d2", "e8g8"],
    ["c2c4", "c7c5", "g1f3", "b8c6", "b1c3", "g7g6", "g2g3", "f8g7", "f1g2", "e7e6"],
    # 1. Nf3
    ["g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "e7e6", "e1g1", "f8e7", "d2d3", "e8g8"],
    ["g1f3", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "f8e7", "e1g1", "e8g8"],
    # 1. g3
    ["g2g3", "c7c5", "f1g2", "g8f6", "c2c4", "d7d5", "c4d5", "f6d5", "d2d4", "c5d4"],
    ["g2g3", "c7c5", "f1g2", "g8f6", "c2c4", "d7d5", "c4d5", "f6d5", "b1c3", "d5c7"],
    # 1. b3
    ["b2b3", "e7e5", "c1b2", "b8c6", "g1f3", "g8f6", "e2e3", "f8d6", "f1b5", "e8g8"],
    ["b2b3", "d7d5", "c1b2", "g8f6", "g1f3", "e7e6", "e2e3", "f8e7", "f1e2", "e8g8"],
    # 1. f4
    ["f2f4", "d7d5", "g1f3", "g8f6", "e2e3", "c7c5", "f1b5", "b8c6", "e1g1", "e7e6"],
    
]
_BOOK_BLACK_REPLY_NON_E4D4 = _build_book(_BLACK_REPLY_LINES_NON_E4D4)

# v4-only RNG: diversify openings across games
_RNG = secrets.SystemRandom()


def pick_book_move(pos: Position, max_fullmove: int = 6) -> Move | None:
    """Return a randomized book move if position matches and early enough."""
    if pos.fullmove > max_fullmove:
        return None

    # If we're Black and this position matches our "non-e4/d4" reply book, use it.
    # Originally this only applied at fullmove==1, but we also want Black to stay in-book
    # on subsequent plies of these lines (e.g. 1.g3 c5 2.Bg2 ...).
    if getattr(pos, "side", 0) == 1:
        try:
            moves = _BOOK_BLACK_REPLY_NON_E4D4.get(_book_key(pos))
            if moves:
                for uci in moves:
                    mv = _uci_to_move(pos, uci)
                    if mv is not None:
                        return mv
        except Exception:
            pass

    moves = _get_merged_opening_book().get(_book_key(pos))
    if not moves:
        return None
    # If we're Black and multiple book replies exist (book was built from white lines),
    # prefer the upper/first line to keep black play stable.
    if getattr(pos, "side", 0) == 1:
        for uci in moves:
            mv = _uci_to_move(pos, uci)
            if mv is not None:
                return mv
        return None

    legal: list[Move] = []
    for uci in moves:
        mv = _uci_to_move(pos, uci)
        if mv is not None:
            legal.append(mv)
    if not legal:
        return None
    return _RNG.choice(legal)

