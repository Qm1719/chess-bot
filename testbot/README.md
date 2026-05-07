## testbot

This folder is for testing the bot levels (mainly levels 2–10 in `engine/agents_v4.py`) with a small round-robin tournament.

## 1) Requirements

- Python environment same as `root folder/`
- `python-chess` installed
- Optional: `STOCKFISH_PATH` if you want Stockfish-based adjudication

## 2) Run

From `root folder/testbot/`:

```bash
python tournament.py
```

## 3) Output

- `out/results.pgn`: all games
- `out/summary.txt`: final table + per-pair summary