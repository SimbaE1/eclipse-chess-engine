#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pause the NNUE preprocessing pipeline only when the Eclipse lichess bot
has TWO simultaneous games running (an incoming challenge accepted on top
of the normal matchmaking game) -- a single game uses its own 4 threads and
doesn't need preprocessing's 4 freed up. Only the second, concurrent game
needs that headroom.

Polls https://lichess.org/api/account/playing with the bot's own token --
decoupled from lichess-bot's internals on purpose, so it works regardless of
which lichess-bot process/config is active. Creates the flag file the
moment 2+ games are ongoing, removes it once back down to <=1.

Usage:
    LICHESS_BOT_TOKEN=... python scripts/bot_pause_watcher.py \\
        --flag /tmp/eclipse_preprocess.pause [--poll-seconds 5] \\
        [--pause-at-games 2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests


def games_in_progress(token: str) -> int:
    resp = requests.get(
        "https://lichess.org/api/account/playing",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    return len(resp.json().get("nowPlaying") or [])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flag", required=True, help="pause-flag file path")
    p.add_argument("--poll-seconds", type=float, default=5.0)
    p.add_argument("--pause-at-games", type=int, default=2,
                   help="pause once nowPlaying count reaches this many")
    args = p.parse_args()

    token = os.environ.get("LICHESS_BOT_TOKEN")
    if not token:
        sys.exit("LICHESS_BOT_TOKEN is not set in the environment")

    paused = os.path.exists(args.flag)
    print(f"watching account play-state, flag={args.flag}, "
          f"pause-at-games={args.pause_at_games}, starting paused={paused}")

    while True:
        try:
            n_games = games_in_progress(token)
        except Exception as e:
            print(f"poll failed (leaving flag state unchanged): {e}")
            time.sleep(args.poll_seconds)
            continue

        should_pause = n_games >= args.pause_at_games
        if should_pause and not paused:
            with open(args.flag, "w") as f:
                f.write("")
            paused = True
            print(f"{n_games} games active -> pausing preprocessing")
        elif not should_pause and paused:
            os.remove(args.flag)
            paused = False
            print(f"{n_games} games active -> resuming preprocessing")

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
