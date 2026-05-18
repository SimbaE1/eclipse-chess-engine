#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Stream-sampler for Lichess monthly PGN dumps.

Reads PGN from stdin (typically piped from `curl ... | zstd -dc`), filters for
high-Elo non-blitz games, and writes random-ply FENs to stdout, one per line.
Stops cleanly when --target FENs have been emitted - the upstream curl gets a
SIGPIPE and shuts down without finishing the 30GB download.

Why this shape: the official Lichess monthly dump is huge but only ~1% of games
match the TCEC-grade quality bar we want. Streaming + early-exit means we never
materialise more than a few hundred MB to disk regardless of how big the input
is.

Sampling strategy:
  - filter by Elo (both players >= --min-elo)
  - filter by time control (base time >= --min-tc-seconds)
  - pick --samples-per-game ply indices uniformly at random from
    [--min-ply, len(game))
  - for a fraction --random-noise of those samples, play one extra random LEGAL
    move before recording the FEN. This intentionally puts the model in
    off-optimal-line positions so the eval landscape doesn't have a hard
    discontinuity at the boundary of "stuff Stockfish has seen in a 2400 game".

Output is plain text, one FEN per line, suitable as input to
label_with_stockfish.py.
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Optional

try:
    import chess
    import chess.pgn
except ImportError:
    print("error: requires python-chess. pip install python-chess", file=sys.stderr)
    sys.exit(1)


def parse_tc_base(tc: str) -> Optional[int]:
    """Lichess TimeControl headers look like "600+5" (10min + 5s increment) or
    "-" for correspondence. Returns the base time in seconds, or None if
    unparseable / correspondence."""
    if not tc or tc == "-":
        return None
    try:
        base = tc.split("+", 1)[0]
        return int(base)
    except (ValueError, IndexError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, required=True,
                   help="emit this many FENs and exit")
    p.add_argument("--min-elo", type=int, default=2400,
                   help="reject games where either player rated below this")
    p.add_argument("--min-tc-seconds", type=int, default=300,
                   help="reject games whose TimeControl base is below this. "
                        "300=5min keeps rapid+ and rejects blitz/bullet.")
    p.add_argument("--samples-per-game", type=int, default=5,
                   help="random plies sampled per accepted game")
    p.add_argument("--min-ply", type=int, default=10,
                   help="don't sample positions before this ply (skip the "
                        "opening-book region where every game is identical)")
    p.add_argument("--random-noise", type=float, default=0.05,
                   help="fraction of samples to perturb with one extra random "
                        "legal move before recording. 0.0 = pure on-line "
                        "positions, 1.0 = always-perturbed.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-every", type=int, default=10000)
    args = p.parse_args()

    rng = random.Random(args.seed)
    emitted = 0
    games_scanned = 0
    games_accepted = 0
    started_emitting = False

    # Read PGN headers + moves directly from stdin. python-chess is lazy here -
    # it reads only enough bytes to materialise one game at a time, so memory
    # stays flat.
    while emitted < args.target:
        try:
            game = chess.pgn.read_game(sys.stdin)
        except UnicodeDecodeError:
            # Occasionally a partial multibyte sequence at a buffer boundary.
            # Skip the malformed game and keep going - one bad row in millions
            # isn't worth aborting over.
            continue
        if game is None:
            print(f"  EOF from stdin after {games_scanned} games; emitted={emitted}",
                  file=sys.stderr)
            break
        games_scanned += 1

        h = game.headers
        try:
            welo = int(h.get("WhiteElo", 0))
            belo = int(h.get("BlackElo", 0))
        except ValueError:
            continue
        if min(welo, belo) < args.min_elo:
            continue

        tc_base = parse_tc_base(h.get("TimeControl", ""))
        if tc_base is None or tc_base < args.min_tc_seconds:
            continue

        # Walk the mainline. For huge games this is the dominant cost; we
        # collect moves once and then sample from them.
        moves = list(game.mainline_moves())
        if len(moves) < args.min_ply + 1:
            continue
        games_accepted += 1

        candidates = list(range(args.min_ply, len(moves)))
        n_pick = min(args.samples_per_game, len(candidates))
        picks = set(rng.sample(candidates, n_pick))

        board = game.board()
        for i, mv in enumerate(moves):
            board.push(mv)
            if i not in picks:
                continue

            # Maybe inject noise by playing one extra random legal move on top
            # of the current position before recording.
            if rng.random() < args.random_noise:
                legal = list(board.legal_moves)
                if legal:
                    noisy = chess.Board(board.fen())
                    noisy.push(rng.choice(legal))
                    print(noisy.fen())
                else:
                    print(board.fen())
            else:
                print(board.fen())

            emitted += 1
            if emitted % args.report_every == 0:
                print(f"  emitted={emitted} games_scanned={games_scanned} "
                      f"games_accepted={games_accepted}", file=sys.stderr)
                sys.stdout.flush()
            if emitted >= args.target:
                break

    print(f"done: emitted {emitted} FENs, scanned {games_scanned} games, "
          f"accepted {games_accepted}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Downstream consumer closed the pipe (e.g. user hit Ctrl-C on the
        # shell). Quietly exit so we don't dump a stack trace.
        sys.exit(0)
