#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gracefully pause the Eclipse lichess bot WITHOUT interrupting a live game.

Stopping lichess-bot mid-game (e.g. a bare SIGINT while it is searching) kills
the engine subprocess, so the bot stops moving and forfeits on time -- even
from a winning position. This script avoids that: it polls the account's
play-state and only signals the bot once NO game is in progress, i.e. in the
gap between games (matchmaking's `challenge_timeout` leaves ~1 min of idle
before the next challenge is created), so the current game always finishes
normally first and no new game is started.

"Pause" here means: stop the lichess-bot process cleanly between games. There
is no runtime "idle but alive" mode in lichess-bot, so resuming means starting
the bot again.

Usage:
    LICHESS_BOT_TOKEN=... python scripts/pause_bot_after_game.py \\
        [--config config-eclipse.yml] [--pid PID] \\
        [--poll-seconds 4] [--signal INT] [--confirm-idle 2]
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
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


def find_bot_pid(config: str) -> int | None:
    """Locate the running lichess-bot process for the given config file."""
    try:
        out = subprocess.check_output(["pgrep", "-f", f"lichess-bot.py.*{config}"],
                                      text=True)
    except subprocess.CalledProcessError:
        return None
    pids = [int(x) for x in out.split()]
    return pids[0] if pids else None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config-eclipse.yml",
                   help="config filename used to find the bot process")
    p.add_argument("--pid", type=int, default=None,
                   help="bot PID (overrides --config lookup)")
    p.add_argument("--poll-seconds", type=float, default=4.0)
    p.add_argument("--signal", default="INT",
                   help="signal to send when idle (INT or TERM)")
    p.add_argument("--confirm-idle", type=int, default=2,
                   help="require this many consecutive idle polls before signaling, "
                        "so a momentary API blip can't trigger a stop mid-game")
    args = p.parse_args()

    token = os.environ.get("LICHESS_BOT_TOKEN")
    if not token:
        sys.exit("LICHESS_BOT_TOKEN is not set in the environment")

    sig = getattr(signal, f"SIG{args.signal.upper().removeprefix('SIG')}")

    pid = args.pid if args.pid is not None else find_bot_pid(args.config)
    if pid is None:
        sys.exit(f"no running lichess-bot process found for config '{args.config}' "
                 f"(already paused?). Use --pid to specify one.")
    if not pid_alive(pid):
        sys.exit(f"PID {pid} is not running (already paused?).")

    print(f"pausing bot PID {pid} after current game; "
          f"poll={args.poll_seconds}s, signal=SIG{args.signal.upper()}")

    idle_streak = 0
    while True:
        if not pid_alive(pid):
            print(f"bot PID {pid} exited on its own; nothing to do.")
            return
        try:
            n = games_in_progress(token)
        except Exception as e:
            print(f"poll failed (waiting, will not signal): {e}")
            idle_streak = 0
            time.sleep(args.poll_seconds)
            continue

        if n > 0:
            if idle_streak:
                print(f"{n} game(s) back in progress -- resetting idle count")
            idle_streak = 0
            print(f"{n} game(s) in progress -- waiting for it to finish...")
        else:
            idle_streak += 1
            print(f"no game in progress ({idle_streak}/{args.confirm_idle})")
            if idle_streak >= args.confirm_idle:
                print(f"idle gap confirmed -- sending SIG{args.signal.upper()} to PID {pid}")
                try:
                    os.kill(pid, sig)
                except OSError as e:
                    sys.exit(f"failed to signal PID {pid}: {e}")
                # Confirm it actually stops.
                for _ in range(10):
                    time.sleep(1)
                    if not pid_alive(pid):
                        print(f"bot PID {pid} stopped cleanly between games.")
                        return
                print(f"signal sent but PID {pid} still alive after 10s; "
                      f"check the bot log.")
                return

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
