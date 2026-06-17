#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Tiny interactive driver: play a game against ./build/src/eclipse.

Usage from the eclipse/ directory:
    python scripts/play_human.py [--engine-time-ms 5000] [--side w|b]

Tracks the move list, talks UCI on stdin/stdout, asks you for each of your
moves in plain UCI (e.g. `e2e4`, `e7e8q`). Doesn't try to be a full GUI -
just enough to get a smoke-test game out of the engine without typing
`position startpos moves ...` by hand every turn.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import chess
except ImportError:
    print("error: requires python-chess. pip install python-chess", file=sys.stderr)
    sys.exit(1)


def send(p, line: str):
    assert p.stdin is not None
    p.stdin.write(line + "\n")
    p.stdin.flush()


def read_until(p, token: str):
    """Return the list of stdout lines emitted up to and including the one
    containing `token`. Raises if the engine closes its stdout first."""
    assert p.stdout is not None
    out = []
    while True:
        line = p.stdout.readline()
        if not line:
            raise RuntimeError("engine exited unexpectedly. tail:\n" + "\n".join(out[-20:]))
        line = line.rstrip()
        out.append(line)
        if token in line:
            return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", default="./build/src/eclipse",
                        help="path to engine binary (default: ./build/src/eclipse)")
    parser.add_argument("--nnue", default="data/eclipse.nnue")
    parser.add_argument("--policy", default="data/policy.onnx")
    parser.add_argument("--engine-time-ms", type=int, default=5000,
                        help="ms per move the engine spends thinking (used only if --tc is 'fixed')")
    parser.add_argument("--side", choices=["w", "b"], default="w",
                        help="which side YOU play (default: white)")
    parser.add_argument("--tc", default="5+3",
                        help="time control (e.g., 5+3 for 5m+3s, or 'fixed' to use --engine-time-ms)")
    parser.add_argument("--syzygy", default="/Users/ezra/syzygy",
                        help="path to Syzygy tablebases")
    parser.add_argument("--threads", type=int, default=4,
                        help="number of search threads")
    parser.add_argument("--ab-threads", type=int, default=1,
                        help="number of alpha-beta verification threads")
    parser.add_argument("--hash", type=int, default=256,
                        help="transposition table size in MB")
    args = parser.parse_args()

    eng_path = Path(args.engine).resolve()
    if not eng_path.exists():
        sys.exit(f"engine not found at {eng_path}. Build first: cmake --build build")

    nnue = Path(args.nnue).resolve()
    policy = Path(args.policy).resolve()

    use_clock = (args.tc.lower() != "fixed")
    wtime, btime, winc, binc = 0, 0, 0, 0
    if use_clock:
        tc_match = re.match(r"^(\d+)(?:\+(\d+))?$", args.tc)
        if not tc_match:
            sys.exit("Invalid time control format. Use e.g. 5+3 or 5 or fixed")
        initial_mins = int(tc_match.group(1))
        inc_secs = int(tc_match.group(2)) if tc_match.group(2) else 0
        wtime = initial_mins * 60 * 1000
        btime = initial_mins * 60 * 1000
        winc = inc_secs * 1000
        binc = inc_secs * 1000

    print(f"Engine:     {eng_path}")
    print(f"NNUE:       {nnue}")
    print(f"Policy:     {policy}")
    print(f"You play:   {'White' if args.side == 'w' else 'Black'}")
    if use_clock:
        print(f"Clock TC:   {args.tc} (White: {wtime/1000:.0f}s, Black: {btime/1000:.0f}s, Inc: {winc/1000:.0f}s)")
    else:
        print(f"Engine ms:  {args.engine_time_ms}")
    print(f"Threads:    {args.threads}")
    print(f"AbThreads:  {args.ab_threads}")
    print(f"Hash:       {args.hash} MB")
    if args.syzygy:
        print(f"Syzygy:     {args.syzygy}")
    print()

    p = subprocess.Popen([str(eng_path)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)

    send(p, "uci")
    read_until(p, "uciok")
    send(p, f"setoption name EvalFile value {nnue}")
    send(p, f"setoption name PolicyFile value {policy}")
    send(p, f"setoption name Threads value {args.threads}")
    send(p, f"setoption name AbThreads value {args.ab_threads}")
    send(p, f"setoption name Hash value {args.hash}")
    send(p, "setoption name Ponder value true")
    if args.syzygy:
        syzygy_path = Path(args.syzygy).resolve()
        if syzygy_path.exists():
            send(p, f"setoption name SyzygyPath value {syzygy_path}")
        else:
            print(f"Warning: Syzygy path '{args.syzygy}' not found, playing without tablebases.")
    send(p, "isready")
    read_until(p, "readyok")

    board = chess.Board()
    move_history: list[str] = []
    user_is_white = (args.side == "w")

    # Ponder state. While it's the human's turn the engine searches the position
    # it predicts will arise (history + its predicted reply, `ponder_move`). If
    # the human plays that move we send `ponderhit` and the in-flight search
    # becomes the engine's reply; otherwise we `stop` it and discard the result.
    pondering = False
    ponder_move: str | None = None
    engine_search_active = False
    engine_search_start = 0.0

    def go_cmd(prefix: str = "") -> str:
        if use_clock:
            return f"go {prefix}wtime {wtime} btime {btime} winc {winc} binc {binc}"
        return f"go {prefix}movetime {args.engine_time_ms}"

    try:
        while not board.is_game_over():
            print(board.unicode(borders=True, empty_square="."))
            if use_clock:
                print(f"Clock -> White: {wtime/1000:.1f}s | Black: {btime/1000:.1f}s")
            stm = "White" if board.turn == chess.WHITE else "Black"
            print(f"\n{stm} to move (ply {board.ply()})")

            if (board.turn == chess.WHITE) == user_is_white:
                # Human's turn
                start_time = time.time()
                while True:
                    raw = input("Your move (UCI, or 'quit'): ").strip()
                    if raw.lower() in ("quit", "q", "exit"):
                        send(p, "quit")
                        return
                    try:
                        mv = chess.Move.from_uci(raw)
                    except (chess.InvalidMoveError, ValueError):
                        print("  bad UCI format. example: e2e4, e7e8q")
                        continue
                    if mv not in board.legal_moves:
                        print("  illegal in this position. legal moves:",
                              " ".join(m.uci() for m in list(board.legal_moves)[:20]),
                              "..." if board.legal_moves.count() > 20 else "")
                        continue
                    break
                
                elapsed = int((time.time() - start_time) * 1000)
                if use_clock:
                    if user_is_white:
                        wtime = wtime - elapsed + winc
                        if wtime <= 0:
                            wtime = 0
                            print("You flagged! Game over.")
                            break
                    else:
                        btime = btime - elapsed + binc
                        if btime <= 0:
                            btime = 0
                            print("You flagged! Game over.")
                            break
                
                board.push(mv)
                move_history.append(mv.uci())

                # Resolve an in-flight ponder search against what the human did.
                if pondering:
                    if mv.uci() == ponder_move:
                        # Ponder hit: the engine is already searching this exact
                        # position. Put it on the clock and reuse the result as
                        # the engine's reply (collected in the engine block).
                        send(p, "ponderhit")
                        engine_search_active = True
                        engine_search_start = time.time()
                    else:
                        # Ponder miss: discard the wasted search.
                        send(p, "stop")
                        read_until(p, "bestmove")
                    pondering = False
                    ponder_move = None
            else:
                # Engine's turn
                if engine_search_active:
                    # A ponderhit search is already running for this position;
                    # just collect its result. The clock started at ponderhit.
                    start_time = engine_search_start
                    lines = read_until(p, "bestmove")
                    engine_search_active = False
                else:
                    pos_cmd = "position startpos"
                    if move_history:
                        pos_cmd += " moves " + " ".join(move_history)
                    send(p, pos_cmd)

                    start_time = time.time()
                    send(p, go_cmd())
                    lines = read_until(p, "bestmove")
                elapsed = int((time.time() - start_time) * 1000)
                
                if use_clock:
                    if user_is_white: # engine is black
                        btime = btime - elapsed + binc
                        if btime <= 0:
                            btime = 0
                            print("Engine flagged! You win.")
                            break
                    else: # engine is white
                        wtime = wtime - elapsed + winc
                        if wtime <= 0:
                            wtime = 0
                            print("Engine flagged! You win.")
                            break

                # bestmove line is the last one
                bestmove_line = next(l for l in reversed(lines) if l.startswith("bestmove"))
                tokens = bestmove_line.split()
                if len(tokens) < 2 or tokens[1] == "0000":
                    print("Engine has no move (game over from its POV).")
                    break
                uci = tokens[1]
                # `bestmove X ponder Y` — Y is the reply the engine predicts.
                predicted = None
                if "ponder" in tokens:
                    pi = tokens.index("ponder")
                    if pi + 1 < len(tokens) and tokens[pi + 1] != "0000":
                        predicted = tokens[pi + 1]
                print(f"Engine plays: {uci}" + (f" (pondering on {predicted})" if predicted else ""))
                # Optionally show the engine's info chatter from the lines
                for ln in lines:
                    if ln.startswith("info nodes") or ln.startswith("info string"):
                        print(f"  {ln}")
                mv = chess.Move.from_uci(uci)
                if mv not in board.legal_moves:
                    print(f"!! engine returned illegal move {uci} — aborting")
                    break
                board.push(mv)
                move_history.append(uci)

                # Start pondering on the predicted reply while the human thinks.
                if predicted and not board.is_game_over():
                    try:
                        pm = chess.Move.from_uci(predicted)
                    except (chess.InvalidMoveError, ValueError):
                        pm = None
                    if pm is not None and pm in board.legal_moves:
                        send(p, "position startpos moves "
                                + " ".join(move_history + [predicted]))
                        send(p, go_cmd("ponder "))
                        pondering = True
                        ponder_move = predicted
    except KeyboardInterrupt:
        print("\n(Ctrl-C) aborting.")
    finally:
        print()
        print(board.unicode(borders=True, empty_square="."))
        print(f"Result: {board.result()}")
        print(f"PGN moves: {' '.join(move_history)}")
        send(p, "quit")
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            p.kill()


if __name__ == "__main__":
    main()
