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
import subprocess
import sys
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
                        help="ms per move the engine spends thinking")
    parser.add_argument("--side", choices=["w", "b"], default="w",
                        help="which side YOU play (default: white)")
    args = parser.parse_args()

    eng_path = Path(args.engine).resolve()
    if not eng_path.exists():
        sys.exit(f"engine not found at {eng_path}. Build first: cmake --build build")

    nnue = Path(args.nnue).resolve()
    policy = Path(args.policy).resolve()
    print(f"Engine:    {eng_path}")
    print(f"NNUE:      {nnue}")
    print(f"Policy:    {policy}")
    print(f"You play:  {'White' if args.side == 'w' else 'Black'}")
    print(f"Engine ms: {args.engine_time_ms}")
    print()

    p = subprocess.Popen([str(eng_path)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)

    send(p, "uci")
    read_until(p, "uciok")
    send(p, f"setoption name EvalFile value {nnue}")
    send(p, f"setoption name PolicyFile value {policy}")
    send(p, "isready")
    read_until(p, "readyok")

    board = chess.Board()
    move_history: list[str] = []
    user_is_white = (args.side == "w")

    try:
        while not board.is_game_over():
            print(board.unicode(borders=True, empty_square="."))
            stm = "White" if board.turn == chess.WHITE else "Black"
            print(f"\n{stm} to move (ply {board.ply()})")

            if (board.turn == chess.WHITE) == user_is_white:
                # Human's turn
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
                board.push(mv)
                move_history.append(mv.uci())
            else:
                # Engine's turn
                pos_cmd = "position startpos"
                if move_history:
                    pos_cmd += " moves " + " ".join(move_history)
                send(p, pos_cmd)
                send(p, f"go movetime {args.engine_time_ms}")
                lines = read_until(p, "bestmove")
                # bestmove line is the last one
                bestmove_line = next(l for l in reversed(lines) if l.startswith("bestmove"))
                tokens = bestmove_line.split()
                if len(tokens) < 2 or tokens[1] == "0000":
                    print("Engine has no move (game over from its POV).")
                    break
                uci = tokens[1]
                print(f"Engine plays: {uci}")
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
