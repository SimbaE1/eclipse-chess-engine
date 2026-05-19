#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Eclipse self-play driver. Spawns one engine, alternates sides, writes PGN.

A single subprocess is enough since only one side is thinking at any moment;
running two would just contend for the same CPU. Threads is passed through to
the engine, which internally splits AB (1 thread) and MCTS (N-1 threads) per
the post-MCTS verifier wiring in src/search.cpp.
"""

import argparse
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", default="build/src/eclipse")
    p.add_argument("--nnue",   default="data/eclipse.nnue")
    p.add_argument("--policy", default=None,
                   help="optional policy.onnx; omit to keep PolicyMode=nnue (default)")
    p.add_argument("--threads", type=int, default=8,
                   help="UCI Threads. Engine splits 1 for AB verifier, rest for MCTS.")
    p.add_argument("--time-per-move", type=float, default=5.0,
                   help="seconds per move (wall clock)")
    p.add_argument("--max-plies", type=int, default=400,
                   help="hard cap so a stalled game doesn't run forever")
    p.add_argument("--out", type=Path, required=True, help="output PGN path")
    args = p.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    opts = {"EvalFile": args.nnue, "Threads": args.threads}
    if args.policy:
        opts["PolicyFile"] = args.policy
    engine.configure(opts)

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"]  = "Eclipse self-play"
    game.headers["White"]  = "Eclipse (HalfKAv2)"
    game.headers["Black"]  = "Eclipse (HalfKAv2)"
    game.headers["TimeControl"] = f"{args.time_per_move:g}s/move"
    node = game

    limit = chess.engine.Limit(time=args.time_per_move)
    t0 = time.time()
    for ply in range(args.max_plies):
        if board.is_game_over(claim_draw=True):
            break
        side = "W" if board.turn == chess.WHITE else "B"
        t_move = time.time()
        info = engine.play(board, limit, info=chess.engine.INFO_ALL)
        elapsed = time.time() - t_move
        score = info.info.get("score")
        depth = info.info.get("depth")
        nodes = info.info.get("nodes")
        nps   = info.info.get("nps")
        print(f"  ply {ply+1:>3} {side} {board.san(info.move):<7} "
              f"score={score} depth={depth} nodes={nodes} nps={nps} "
              f"({elapsed:.1f}s)", file=sys.stderr, flush=True)
        node = node.add_variation(info.move,
                                  comment=f"{score} d{depth} n{nodes}")
        board.push(info.move)

    game.headers["Result"] = board.result(claim_draw=True)
    # claim_draw=True above means the loop exited as soon as either 3-fold rep
    # or the 50-move rule was claimable, so we need to check those before the
    # automatic-draw conditions.
    game.headers["Termination"] = (
        "checkmate"             if board.is_checkmate() else
        "stalemate"             if board.is_stalemate() else
        "insufficient material" if board.is_insufficient_material() else
        "fivefold repetition"   if board.is_fivefold_repetition() else
        "75-move rule"          if board.is_seventyfive_moves() else
        "threefold repetition"  if board.can_claim_threefold_repetition() else
        "50-move rule"          if board.can_claim_fifty_moves() else
        "ply cap"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        print(game, file=f)
    engine.quit()
    print(f"\n=== Result {game.headers['Result']} ({game.headers['Termination']}) "
          f"in {time.time()-t0:.0f}s, PGN -> {args.out} ===", file=sys.stderr)


if __name__ == "__main__":
    main()
