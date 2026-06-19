#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Head-to-head benchmark between two local Eclipse configurations (e.g. two
NNUE files) using python-chess with a real clock-based time control.

    python3 scripts/bench_h2h_tc.py \
        --eval1 data/eclipse_halfkav2_v3.nnue --label1 v3 \
        --eval2 data/eclipse_halfkav2_v2.nnue --label2 v2 \
        --tc 180+2 --games 4 --pgn data/v3_vs_v2.pgn

Plays sequentially (concurrency 1), alternating colours each game, using the
same opening book as match_runner.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_runner import OPENINGS  # noqa: E402
from bench_vs_stockfish import parse_tc, elo_from_score  # noqa: E402


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(repo / "build/src/eclipse"))
    ap.add_argument("--eval1", required=True)
    ap.add_argument("--eval2", required=True)
    ap.add_argument("--label1", default="E1")
    ap.add_argument("--label2", default="E2")
    ap.add_argument("--policy", default=str(repo / "data/policy.onnx"))
    ap.add_argument("--tc", default="180+2", help="base+inc seconds, e.g. 180+2")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--pgn", default=str(repo / "data/h2h.pgn"))
    args = ap.parse_args()

    base, inc = parse_tc(args.tc)

    e1 = chess.engine.SimpleEngine.popen_uci(args.engine)
    e1.configure({"EvalFile": args.eval1, "PolicyFile": args.policy})

    e2 = chess.engine.SimpleEngine.popen_uci(args.engine)
    e2.configure({"EvalFile": args.eval2, "PolicyFile": args.policy})

    wins = draws = losses = 0  # from e1's perspective
    pgn_path = Path(args.pgn)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    pgn_fh = pgn_path.open("w")
    t_start = time.time()

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

            wclock, bclock = base, base
            outcome: str | None = None
            while not board.is_game_over(claim_draw=True):
                engine = white if board.turn == chess.WHITE else black
                limit = chess.engine.Limit(
                    white_clock=wclock, black_clock=bclock,
                    white_inc=inc, black_inc=inc,
                )
                t0 = time.time()
                result = engine.play(board, limit)
                elapsed = time.time() - t0
                white_to_move = (board.turn == chess.WHITE)
                if white_to_move:
                    wclock = wclock - elapsed + inc
                else:
                    bclock = bclock - elapsed + inc
                if wclock <= 0.0 or bclock <= 0.0:
                    outcome = "0-1" if white_to_move else "1-0"
                    if result.move is not None:
                        board.push(result.move)
                    break
                if result.move is None:
                    break
                board.push(result.move)
            if outcome is None:
                outcome = board.result(claim_draw=True)

            if outcome == "1-0":
                if e1_is_white: wins += 1
                else: losses += 1
            elif outcome == "0-1":
                if e1_is_white: losses += 1
                else: wins += 1
            else:
                draws += 1

            game = chess.pgn.Game.from_board(board)
            game.headers["White"] = white_name
            game.headers["Black"] = black_name
            game.headers["Result"] = outcome
            game.headers["TimeControl"] = f"{int(base)}+{inc:g}"
            pgn_fh.write(str(game) + "\n\n")
            pgn_fh.flush()

            n = wins + draws + losses
            score = (wins + 0.5 * draws) / n
            elo, ci = elo_from_score(score, n)
            elapsed_min = (time.time() - t_start) / 60.0
            print(f"[{g+1:>3}/{args.games}] {outcome}  "
                  f"{args.label1}: W:{wins} D:{draws} L:{losses}  "
                  f"score={score:.3f}  Elo={elo:+.0f} +/-{ci:.0f}  "
                  f"t={elapsed_min:.1f}min", flush=True)
    finally:
        pgn_fh.close()
        e1.quit()
        e2.quit()

    n = wins + draws + losses
    if n == 0:
        print("no games played", file=sys.stderr)
        return 1
    score = (wins + 0.5 * draws) / n
    elo, ci = elo_from_score(score, n)
    print()
    print(f"Final: {args.label1} {wins}W {draws}D {losses}L  "
          f"score={score:.3f}  Elo diff = {elo:+.0f} +/- {ci:.0f}  (95% CI, n={n})")
    print(f"PGN: {pgn_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
