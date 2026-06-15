#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Per-game move accuracy report using Stockfish as the reference engine.

Evaluates every position in each PGN game with Stockfish at a fixed depth,
converts centipawn scores to win% (lichess formula), and reports each side's
average move accuracy and average centipawn loss (ACPL).

    python3 scripts/analyze_accuracy.py data/v3_vs_v2.pgn --depth 16
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn


def win_percent(cp: int) -> float:
    """Lichess win% formula, from White's perspective."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(loss: float) -> float:
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * loss) - 3.1669))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--depth", type=int, default=16)
    args = ap.parse_args()

    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    limit = chess.engine.Limit(depth=args.depth)

    with open(args.pgn) as f:
        game_idx = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_idx += 1
            white = game.headers.get("White", "White")
            black = game.headers.get("Black", "Black")

            board = game.board()
            info = sf.analyse(board, limit)
            cp_before = info["score"].white().score(mate_score=10000)
            win_before = win_percent(cp_before)

            acpl = {chess.WHITE: [], chess.BLACK: []}
            acc = {chess.WHITE: [], chess.BLACK: []}

            for move in game.mainline_moves():
                mover = board.turn
                board.push(move)
                info = sf.analyse(board, limit)
                cp_after = info["score"].white().score(mate_score=10000)
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
                  f"[{game.headers.get('Result', '?')}] ===")
            for side, name in ((chess.WHITE, white), (chess.BLACK, black)):
                if acc[side]:
                    avg_acc = sum(acc[side]) / len(acc[side])
                    avg_acpl = sum(acpl[side]) / len(acpl[side])
                    print(f"  {name:>4} ({'White' if side == chess.WHITE else 'Black'}): "
                          f"accuracy={avg_acc:.1f}%  ACPL={avg_acpl:.0f}cp  "
                          f"({len(acc[side])} moves)")

    sf.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
