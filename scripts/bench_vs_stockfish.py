#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Engine-vs-engine benchmark using python-chess. Stands in for cutechess-cli on
machines where cutechess is not installed.

Usage (from the eclipse/ repo root):

    python3 scripts/bench_vs_stockfish.py \
        --games 20 --tc 60+1 --opponent-elo 1800 \
        --pgn data/eclipse_vs_sf1800.pgn

Plays an even number of games, alternating colours each pair. Sets Stockfish's
UCI_LimitStrength + UCI_Elo to the requested rating. Prints per-game results
and a final Elo estimate with a 95% Wilson CI.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn


def parse_tc(s: str) -> tuple[float, float]:
    """`'60+1'` -> (60.0, 1.0). Both numbers in seconds."""
    base, _, inc = s.partition("+")
    return float(base), float(inc or 0.0)


def elo_from_score(score: float, n: int) -> tuple[float, float]:
    """Bayeselo-style Elo estimate + 95% half-width from a fraction in [0,1].

    Saturates at +/- 800 Elo for sweeps (0 or 1).
    """
    if score <= 0.0:
        return -800.0, 0.0
    if score >= 1.0:
        return 800.0, 0.0
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    # Wilson 95% interval on the score, mapped through the same logistic.
    z = 1.96
    p = score
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    lo = max(1e-6, centre - margin)
    hi = min(1 - 1e-6, centre + margin)
    elo_lo = -400.0 * math.log10(1.0 / lo - 1.0)
    elo_hi = -400.0 * math.log10(1.0 / hi - 1.0)
    return elo, (elo_hi - elo_lo) / 2.0


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--tc", default="60+1", help="base+inc seconds, e.g. 60+1")
    ap.add_argument("--opponent-elo", type=int, default=1800)
    ap.add_argument("--eclipse", default=str(repo / "build/src/eclipse"))
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--nnue", default=str(repo / "data/eclipse.nnue"))
    ap.add_argument("--policy", default=str(repo / "data/policy.onnx"))
    ap.add_argument("--pgn", default=str(repo / "data/eclipse_vs_sf.pgn"))
    args = ap.parse_args()

    base, inc = parse_tc(args.tc)

    eclipse = chess.engine.SimpleEngine.popen_uci(args.eclipse)
    eclipse.configure({"EvalFile": args.nnue, "PolicyFile": args.policy})

    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.opponent_elo})

    wins = draws = losses = 0  # from Eclipse's perspective
    pgn_path = Path(args.pgn)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    pgn_fh = pgn_path.open("w")
    t_start = time.time()

    try:
        for g in range(args.games):
            eclipse_is_white = (g % 2 == 0)
            white = eclipse if eclipse_is_white else sf
            black = sf if eclipse_is_white else eclipse
            white_name = "Eclipse" if eclipse_is_white else f"SF-{args.opponent_elo}"
            black_name = f"SF-{args.opponent_elo}" if eclipse_is_white else "Eclipse"

            board = chess.Board()
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
                    # Whoever just moved blew their clock.
                    outcome = "0-1" if white_to_move else "1-0"
                    if result.move is not None:
                        board.push(result.move)
                    break
                if result.move is None:
                    # Engine resigned or has no legal move - let board.result
                    # decide (it will handle checkmate/stalemate correctly).
                    break
                board.push(result.move)
            if outcome is None:
                outcome = board.result(claim_draw=True)

            # Score from Eclipse's side
            if outcome == "1-0":
                if eclipse_is_white: wins += 1
                else: losses += 1
            elif outcome == "0-1":
                if eclipse_is_white: losses += 1
                else: wins += 1
            else:
                draws += 1

            # PGN
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
                  f"W:{wins} D:{draws} L:{losses}  "
                  f"score={score:.3f}  Elo={elo:+.0f} ±{ci:.0f}  "
                  f"t={elapsed_min:.1f}min", flush=True)
    finally:
        pgn_fh.close()
        eclipse.quit()
        sf.quit()

    n = wins + draws + losses
    if n == 0:
        print("no games played", file=sys.stderr)
        return 1
    score = (wins + 0.5 * draws) / n
    elo, ci = elo_from_score(score, n)
    print()
    print(f"Final: {wins}W {draws}D {losses}L  "
          f"score={score:.3f}  Elo diff vs SF-{args.opponent_elo} = "
          f"{elo:+.0f} ± {ci:.0f}  (95% CI, n={n})")
    print(f"PGN: {pgn_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
