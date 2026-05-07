"""
test_discord_webhook.py

Send a test message to DISCORD_WEBHOOK_URL.

Usage:
  py test_discord_webhook.py
  (reads submission v5/selftrain.env or .env via the same logic as selftrain)
"""

from __future__ import annotations

import os
import sys

V5 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(V5, "ops"))

from selftrain import _resolve_env_path, _discord_send  # type: ignore


def _load_env(p: str) -> None:
    if not os.path.isfile(p):
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(p, override=False)
    except Exception:
        for line in open(p, encoding="utf-8", errors="replace"):
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            if k and k not in os.environ:
                os.environ[k] = v


def main() -> int:
    env_path = _resolve_env_path()
    _load_env(str(env_path))
    os.environ["SELFTRAIN_NOTIFY_DEBUG"] = "1"
    wh = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not wh:
        print("DISCORD_WEBHOOK_URL is empty.", file=sys.stderr)
        return 2
    _discord_send("selftrain: discord webhook test message")
    print("sent (check channel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

