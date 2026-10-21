# Chess bot (NNUE-assisted)

This repository is a Python chess bot written for an assignment for the Introduction to AI course at HCMUT. (It got full mark btw, hehe, thank teach!)
It features a GUI for local play, and a Lichess integration so you can try the bot online (credit to [lichess-bot-devs](https://github.com/lichess-bot-devs/lichess-bot)).

## Try it out on Lichess

- **Bot profile**: `https://lichess.org/@/nmai252withnnue`
- **Play it**: open the profile → Challenge (no login needed; due to limitations of my VPS, it can only play 1 match at a time).

## Code structure

- **Main engine**: `engine/agents_v4.py` (levels 1–10)
  - Level 10 is the full bot, levels 2–9 are handicapped, level 1 is pure random.
  - Levels were tested and assigned to comply with the requirements (see `testbot/out/summary.txt`).
  - Opening book (`data/opening_book_v4.py`) + extra lines from `data/opening_lines.sqlite3`.
  - Experience book from `data/experience_book.sqlite3` as the main ML-based reinforcement-learning mechanism (play → analyze → store better moves).
  - Syzygy endgame table up to 5 pieces (`data/3-4-5_pieces_Syzygy/`).
  - Stockfish NNUE (`engine/sf_nnue_bridge/`) controlled by `SF_NNUE`, `SF_NNUE_FILE`, `SF_NNUE_BLEND`.
- **Lichess bridge**: `lichess-bot/engines/submission_uci_bridge_v4.py`
  - Exposes UCI option `Level` (1–10) and forwards clocks/time controls.

## Setup

### 1) Install dependencies

Requirements:

- **Python 3.10+** for `lichess-bot/`
- `python-chess`, `PyYAML`, `requests`, etc. from `lichess-bot/requirements.txt`.
- `pygame` if local play

From `root folder/` (create a venv if needed):

```bash
python -m pip install -r lichess-bot/requirements.txt
python -m pip install pygame
```

### 2) Local play

From `root folder/`:
```bash
python main.py
```

### 3) Create a Lichess OAuth token (bot:play)

To operate lichess bot, you need a bot account and an API token.

- Create a new lichess account and confirm it. Don't play any games on it, though.
- Go to `https://lichess.org/api`. Click BOT, then Upgrade BOT account.
- Create a personal API token
- Check all needed boxes (I'd recommend Read incoming challenges, Send, accept and reject challenges, Create many games at once for other players and Play games using the Bot API)
- Copy the code it gives you (start with lip_)
- Open Terminal
- Type this text: `curl -d '' lichess.org/api/bot/account/upgrade -H "Authorization: Bearer YourTokenCodeHere"`
- Press enter. The text "YourTokenCodeHere" is where you should put your token code from the API access token.
- View your profile. Your New Username should have a "BOT" in front.

### 4) Configure env (`LICHESS_TOKEN`, `LICHESS_BOT_NAME`)

- Copy `selftrain.env.example` into `selftrain.env`
- Fill:
  - `LICHESS_TOKEN` with your API token
  - `LICHESS_BOT_NAME` with your bot account name


### 5) Configure lichess-bot engine bridge

`lichess-bot/config.yml` is already set up the path.
You only need to setup the bot behavior from line 127.

### 6) Run the Lichess bot

From `root folder/`:

```bash
python main.py lichess
```

This runs lichess-bot with `matchmaking.allow_matchmaking: false` injected at runtime so that it will only receive challenge from others.

### 7) Reinforcement learning (bot play → analyze by Stockfish → experience book)

From `root folder/`:

```bash
python main.py selftrain
```

What it does (`ops/selftrain.py`):

- Runs lichess-bot until `SELFTRAIN_MAX_GAMES` (default is 15) new PGNs appear.
- Analyzes and ingests all draw/loss games, then writes suggestions into the experience book.
- Stops when `SELFTRAIN_TRAINING_GAMES` is reached. Note: a bot can only play up to 100 games per day (daily reset is at 00:00 UTC, ~07:00 in Ho Chi Minh City).
- Writes logs to `data/selftrain_ingest.log`.
- Log from lichess-bot can be read at `data/selftrain_lichess_bot.log`. Delete the file if it's too long.

To achieve speed and correctness, self-train uses local Stockfish (default at `engine/stockfish/`). It has both Windows and Ubuntu versions. However, on older computers, please check out other builds at [here](https://stockfishchess.org/download/).

You can check which CPU features are compatible with Stockfish builds:
On Windows:

```bash
iwr https://download.sysinternals.com/files/Coreinfo.zip -OutFile $env:TEMP\Coreinfo.zip; Expand-Archive $env:TEMP\Coreinfo.zip -DestinationPath $env:TEMP\Coreinfo -Force; & "$env:TEMP\Coreinfo\coreinfo64.exe" -accepteula -f 2>&1 | % { $_ -replace [char]0,'' } | sls 'AVX2|BMI1|BMI2|AVX-512|VNNI' | ? { $_ -match '\s\*\s' }
```

On Ubuntu:

```bash
lscpu | grep -oiE 'avx2|bmi1|bmi2|avx512[^ ]*|vnni' | sort -u
```

### (Ubuntu) Optional: build the Stockfish NNUE bridge (`sf_nnue_bridge.so`)

On Linux, to use NNUE, a shared library `sf_nnue_bridge.so` is needed.

From `root folder/`:

```bash
sudo apt install -y build-essential
```

The build script expects Stockfish sources at `engine/stockfish/src` (relative to `root folder/`) and will output `engine/sf_nnue_bridge/sf_nnue_bridge.so`:

```bash
cd engine/sf_nnue_bridge
./build_linux.sh
```

If your Stockfish sources are in a different location, edit `engine/sf_nnue_bridge/build_linux.sh` (variable `sf=.../engine/stockfish/src`).


## Other folders

- `tools/`: contains tools for debug and analyze, read its README at `tools/README.md`
- `testbot/`: testing to assign level 2 to 9, read its README at `testbot/README.md`
