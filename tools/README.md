# tools

This folder contains small scripts we used to debug/analyze the bot and maintain the local DBs.

Unless noted, run these from `root folder/`:

## 1) Quick smoke check (engine is sane)

```bash
python tools/smoke_selfplay.py
```

## 2) Analyze a single position (FEN)

Edit the FEN at the top of `tools/analyze_fen.py`, then run:

```bash
python tools/analyze_fen.py
```
Or
```bash
python tools/analyze_fen.py "5Q2/8/8/7K/7P/2k3P1/8/8 w - - 1 85"
```

## 3) Return Syzygy tablebase probe (WDL/DTZ dump)

Same as 2)

## 4) Local Stockfish analysis for one game id

Edit the game ID at the top of `tools/two_pass_local_sf_test.py`, then run:

```bash
python tools/two_pass_local_sf_test.py
```
Or
```bash
python tools/two_pass_local_sf_test.py "7Bq5YXRjCPOJ"
```

Notes:
- This script accepts either a full game id (`7Bq5YXRj`) or an analysis link id (`7Bq5YXRjCPOJ`) and will auto-trim to 8 chars.
- It needs `python-chess` and a Stockfish binary (`STOCKFISH_PATH` or `engine/stockfish/`).

## 5) Opening book maintenance (`data/opening_lines.sqlite3`)

These write into `data/opening_lines.sqlite3`. The engine hot-reloads this file (no need to restart).

### 5.1) Add a line from a PGN move string (SAN → UCI)

```bash
python tools/add_opening_line_from_pgn.py --pgn "1. d4 Nf6 2. c4 d6 { A53 Old Indian } 3. Nc3 Bg4 4. h3 Bh5"
```

### 5.2) Create a line from a Lichess game id (Stockfish-assisted)

```bash
python tools/add_opening_book_line.py <gameId>
```

## 6) Experience book maintenance (`data/experience_book.sqlite3`)

### 6.1) Merge two experience books

```bash
python tools/merge_experience_books.py <src.sqlite3> <dst.sqlite3>
```

### 6.2) Self-add lines to experience book

```bash
python ../engine/experience_book.py add --fen "2r2rk1/1p1b1p1p/2p1p1pQ/qp2P2P/1b1P4/1R1B1N2/5PP1/4R1K1 w - - 4 25" --move f3g5 --depth 18
```

## 7) Tournament-derived training

```bash
python tools/tournament_train.py
```

This writes `tools/tournament_train_report.txt` and can optionally add rows to `data/experience_book.sqlite3`.
