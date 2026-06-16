#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Per-game move accuracy report using Stockfish as the reference engine.

Evaluates every position in each PGN game with Stockfish at a fixed depth,
converts centipawn scores to win% (lichess formula), and reports each side's
average move accuracy and average centipawn loss (ACPL).

Uses the same raw-fd UCI driver as bench_vs_stockfish.py: python-chess's
asyncio chess.engine driver hangs on macOS (a select()/TextIOWrapper buffering
mismatch when an engine flushes many lines at once). Runs single-threaded by
default so it can share a spare core without stealing search threads from a
concurrently-running engine match.

    python3 scripts/analyze_accuracy.py data/v3_vs_v2.pgn --depth 14
"""

from __future__ import annotations

import argparse
import math
import sys

import chess
import chess.pgn

from bench_vs_stockfish import UciEngine


class Analyzer(UciEngine):
    def eval_cp(self, fen: str, depth: int) -> int:
        """Stockfish eval in centipawns from the side-to-move's POV.

        Mate scores are mapped to a large bounded value so downstream win% /
        accuracy math stays finite.
        """
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        score = 0
        while True:
            line = self._readline()
            if line.startswith("info") and " score " in line:
                t = line.split()
                try:
                    si = t.index("score")
                    kind, val = t[si + 1], int(t[si + 2])
                    if kind == "cp":
                        score = val
                    elif kind == "mate":
                        score = (30000 - abs(val)) if val > 0 else -(30000 - abs(val))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("bestmove"):
                return score


def win_percent(cp: int) -> float:
    """Lichess win% formula, from White's perspective."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(loss_win: float) -> float:
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * loss_win) - 3.1669))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--cap-cp", type=int, default=1000,
                    help="clamp |eval| to this before ACPL/win%% so a decided "
                         "game (and mate scores) don't dominate the averages")
    args = ap.parse_args()
    cap = args.cap_cp

    def clamp(cp: int) -> int:
        return max(-cap, min(cap, cp))

    sf = Analyzer(args.stockfish, "sf-analyze")
    sf.configure({"Threads": args.threads, "Hash": args.hash})

    # name -> running lists across all games
    tot_acpl: dict[str, list[float]] = {}
    tot_acc: dict[str, list[float]] = {}

    with open(args.pgn) as f:
        game_idx = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if not list(game.mainline_moves()):
                continue  # unfinished / empty game
            game_idx += 1
            white = game.headers.get("White", "White")
            black = game.headers.get("Black", "Black")

            # Evaluate every distinct position once (White-POV cp), cached by FEN.
            evals: dict[str, int] = {}

            def white_cp(board: chess.Board) -> int:
                fen = board.fen()
                if fen not in evals:
                    cp = sf.eval_cp(fen, args.depth)            # side-to-move POV
                    evals[fen] = cp if board.turn == chess.WHITE else -cp
                return evals[fen]

            board = game.board()
            cp_before = clamp(white_cp(board))
            win_before = win_percent(cp_before)

            acpl = {chess.WHITE: [], chess.BLACK: []}
            acc = {chess.WHITE: [], chess.BLACK: []}

            for move in game.mainline_moves():
                mover = board.turn
                board.push(move)
                cp_after = clamp(white_cp(board))
                win_after = win_percent(cp_after)

                if mover == chess.WHITE:
                    loss_cp = max(0, cp_before - cp_after)
                    loss_win = max(0.0, win_before - win_after)
                else:
                    loss_cp = max(0, cp_after - cp_before)
                    loss_win = max(0.0, win_after - win_before)

                acpl[mover].append(loss_cp)
                acc[mover].append(move_accuracy(loss_win))
                cp_before, win_before = cp_after, win_after

            print(f"\n=== Game {game_idx}: {white} (White) vs {black} (Black) "
                  f"[{game.headers.get('Result', '?')}] ===", flush=True)
            for side, name in ((chess.WHITE, white), (chess.BLACK, black)):
                if acc[side]:
                    avg_acc = sum(acc[side]) / len(acc[side])
                    avg_acpl = sum(acpl[side]) / len(acpl[side])
                    print(f"  {name:>12} ({'White' if side == chess.WHITE else 'Black'}): "
                          f"accuracy={avg_acc:5.1f}%  ACPL={avg_acpl:4.0f}cp  "
                          f"({len(acc[side])} moves)", flush=True)
                    tot_acpl.setdefault(name, []).extend(acpl[side])
                    tot_acc.setdefault(name, []).extend(acc[side])

    sf.quit()

    print(f"\n=== Aggregate over {game_idx} game(s), Stockfish depth {args.depth} ===")
    for name in sorted(tot_acpl):
        avg_acc = sum(tot_acc[name]) / len(tot_acc[name])
        avg_acpl = sum(tot_acpl[name]) / len(tot_acpl[name])
        print(f"  {name:>12}: accuracy={avg_acc:5.1f}%  ACPL={avg_acpl:4.0f}cp  "
              f"({len(tot_acpl[name])} moves)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
