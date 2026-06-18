#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pause the NNUE preprocessing pipeline while the Eclipse lichess bot is
mid-game, so the engine isn't fighting preprocessing workers for CPU.

Polls https://lichess.org/api/account/playing with the bot's own token --
decoupled from lichess-bot's internals on purpose, so it works regardless of
which lichess-bot process/config is active. Creates the flag file the moment
a game is ongoing, removes it the moment none are.

Usage:
    LICHESS_BOT_TOKEN=... python scripts/bot_pause_watcher.py \\
        --flag /tmp/eclipse_preprocess.pause [--poll-seconds 5]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests


def is_playing(token: str) -> bool:
    resp = requests.get(
        "https://lichess.org/api/account/playing",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    return bool(resp.json().get("nowPlaying"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flag", required=True, help="pause-flag file path")
    p.add_argument("--poll-seconds", type=float, default=5.0)
    args = p.parse_args()

    token = os.environ.get("LICHESS_BOT_TOKEN")
    if not token:
        sys.exit("LICHESS_BOT_TOKEN is not set in the environment")

    paused = os.path.exists(args.flag)
    print(f"watching account play-state, flag={args.flag}, "
          f"starting paused={paused}")

    while True:
        try:
            playing = is_playing(token)
        except Exception as e:
            print(f"poll failed (leaving flag state unchanged): {e}")
            time.sleep(args.poll_seconds)
            continue

        if playing and not paused:
            with open(args.flag, "w") as f:
                f.write("")
            paused = True
            print("game active -> pausing preprocessing")
        elif not playing and paused:
            os.remove(args.flag)
            paused = False
            print("no game active -> resuming preprocessing")

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
