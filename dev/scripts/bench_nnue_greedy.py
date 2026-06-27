#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Head-to-head between two NNUE files with NO search: for each position, every
legal move is applied and the resulting position is scored with a single
NNUE forward pass (relative to the side to move after the move). The mover
picks the move that minimizes the opponent's resulting eval (i.e. greedy
1-ply minimax with NNUE as the leaf evaluator).

    python3 scripts/bench_nnue_greedy.py \
        --eval1 data/eclipse_halfkav2_v3.nnue --label1 v3 \
        --eval2 data/eclipse_halfkav2_v2.nnue --label2 v2 \
        --games 4 --pgn data/v3_vs_v2_greedy.pgn
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_runner import OPENINGS  # noqa: E402


class NNUEEvalServer:
    def __init__(self, binary: str, nnue_path: str):
        self.proc = subprocess.Popen(
            [binary, nnue_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        # First line of stdout is the "NNUE loaded" banner.
        self.proc.stdout.readline()

    def eval_fen(self, fen: str) -> int:
        self.proc.stdin.write(fen + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        return int(line)

    def best_move(self, board: chess.Board) -> chess.Move:
        best_move = None
        best_score = None
        for move in board.legal_moves:
            board.push(move)
            # eval is relative to the side now to move (the opponent);
            # the mover wants to minimize that.
            score = self.eval_fen(board.fen())
            board.pop()
            if best_score is None or score < best_score:
                best_score = score
                best_move = move
        return best_move

    def quit(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=5)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=str(repo / "build/tests/nnue_eval_server"))
    ap.add_argument("--server1")
    ap.add_argument("--server2")
    ap.add_argument("--eval1", required=True)
    ap.add_argument("--eval2", required=True)
    ap.add_argument("--label1", default="E1")
    ap.add_argument("--label2", default="E2")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--pgn", default=str(repo / "data/greedy_h2h.pgn"))
    args = ap.parse_args()

    server1 = args.server1 or args.server
    server2 = args.server2 or args.server

    e1 = NNUEEvalServer(server1, args.eval1)
    e2 = NNUEEvalServer(server2, args.eval2)

    wins = draws = losses = 0  # from e1's perspective
    pgn_path = Path(args.pgn)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    pgn_fh = pgn_path.open("w")

    try:
        for g in range(args.games):
            e1_is_white = (g % 2 == 0)
            white = e1 if e1_is_white else e2
            black = e2 if e1_is_white else e1
            white_name = args.label1 if e1_is_white else args.label2
            black_name = args.label2 if e1_is_white else args.label1

            opening = OPENINGS[g % len(OPENINGS)].split()
            board = chess.Board()
            for uci in opening:
                board.push_uci(uci)

            move_count = 0
            while not board.is_game_over(claim_draw=True) and move_count < args.max_moves:
                engine = white if board.turn == chess.WHITE else black
                move = engine.best_move(board)
                if move is None:
                    break
                board.push(move)
                move_count += 1

            outcome = board.outcome(claim_draw=True)
            result = outcome.result() if outcome else "*"

            if result == "1-0":
                if e1_is_white: wins += 1
                else: losses += 1
            elif result == "0-1":
                if e1_is_white: losses += 1
                else: wins += 1
            else:
                draws += 1

            game = chess.pgn.Game.from_board(board)
            game.headers["White"] = white_name
            game.headers["Black"] = black_name
            game.headers["Result"] = result
            pgn_fh.write(str(game) + "\n\n")
            pgn_fh.flush()

            n = wins + draws + losses
            print(f"[{g+1:>3}/{args.games}] {result}  "
                  f"{args.label1}: W:{wins} D:{draws} L:{losses}  "
                  f"({move_count} moves)", flush=True)
    finally:
        pgn_fh.close()
        e1.quit()
        e2.quit()

    print(f"\nPGN: {pgn_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
