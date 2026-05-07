from __future__ import annotations

import os
import sys

fen = "5Q2/8/8/7K/7P/2k3P1/8/8 w - - 1 85"

def _add_engine_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    v5 = os.path.abspath(os.path.join(here, ".."))
    engine = os.path.join(v5, "engine")
    for p in (engine, v5):
        if p not in sys.path:
            sys.path.insert(0, p)


def main(argv: list[str]) -> int:
    global fen
    if len(argv) >= 2:
        fen = " ".join(argv[1:]).strip()
    _add_engine_path()

    from syzygy_345 import find_syzygy_path, probe_info, probe_move_list  # type: ignore

    tb = find_syzygy_path()
    if not tb:
        print("syzygy: not found (set SYZYGY_PATH or place TBs under submission v5/data/...)")
        return 1

    info = probe_info(fen)
    moves = probe_move_list(fen)
    print(f"fen: {fen}")
    print(f"tb_path: {tb}")
    print(f"info: {info}")
    if not moves:
        print("no move list (out of TB scope?)")
        return 0

    print("")
    print("uci\twdl\tdtz")
    for d in moves:
        print(f"{d.get('uci')}\t{d.get('wdl')}\t{d.get('dtz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

