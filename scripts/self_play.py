#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Eclipse self-play driver. Spawns one engine, alternates sides, writes PGN.

A single subprocess is enough since only one side is thinking at any moment;
running two would just contend for the same CPU. Threads is passed through to
the engine, which internally splits AB (1 thread) and MCTS (N-1 threads) per
the post-MCTS verifier wiring in src/search.cpp.

Drives UCI manually (rather than via python-chess SimpleEngine) so that the
`info string AB ...` lines emitted by the post-MCTS verifier survive to the
user instead of being collapsed into INFO_ALL's "latest string only" slot.
"""

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess
import chess.pgn


@dataclass
class MoveStats:
    move:        chess.Move
    score_cp:    Optional[int]
    nodes:       Optional[int]
    nps:         Optional[int]
    ab_line:     Optional[str]   # raw "info string AB verify/override: ..." text


_INFO_NODES_RE = re.compile(
    r"^info\s+nodes\s+(\d+)\s+nps\s+(\d+)\s+time\s+\d+\s+score\s+cp\s+(-?\d+)\b"
)
_INFO_AB_RE    = re.compile(r"^info\s+string\s+AB\s+\S+:\s.*\(d=(\d+),\s*nodes=(\d+)\)\s*$")


class UciEngine:
    """Minimal UCI driver focused on streaming info lines. Not a general-purpose
    UCI client — only implements what self-play needs."""

    def __init__(self, path: str):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, sentinel_prefix: str):
        """Reads stdout until a line starts with sentinel_prefix. Returns all
        lines read (including the sentinel)."""
        out = []
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            out.append(line)
            if line.startswith(sentinel_prefix):
                return out
        raise RuntimeError(f"engine exited before emitting '{sentinel_prefix}'")

    def configure(self, options: dict) -> None:
        for name, value in options.items():
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._read_until("readyok")

    def go(self, moves_uci: list[str], time_ms: int) -> MoveStats:
        """Set position to startpos + moves, run `go movetime`, parse output."""
        pos = "position startpos"
        if moves_uci:
            pos += " moves " + " ".join(moves_uci)
        self._send(pos)
        self._send(f"go movetime {time_ms}")

        last_main = None        # most recent "info nodes ... score cp ..." line
        ab_line   = None        # last "info string AB ..." line seen
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("bestmove"):
                bm = line.split()[1]
                # Engine ships e.g. "bestmove e2e4 ponder ..." — first token only.
                move = chess.Move.from_uci(bm)
                score_cp = nodes = nps = None
                if last_main is not None:
                    m = _INFO_NODES_RE.match(last_main)
                    if m:
                        nodes, nps, score_cp = (int(g) for g in m.groups())
                return MoveStats(move, score_cp, nodes, nps, ab_line)
            if _INFO_NODES_RE.match(line):
                last_main = line
            elif _INFO_AB_RE.match(line) or line.startswith("info string AB "):
                ab_line = line
        raise RuntimeError("engine exited mid-search")

    def quit(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


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

    engine = UciEngine(args.engine)
    opts = {"EvalFile": args.nnue, "Threads": args.threads}
    if args.policy:
        opts["PolicyFile"] = args.policy
    engine.configure(opts)

    board = chess.Board()
    game  = chess.pgn.Game()
    game.headers["Event"]       = "Eclipse self-play"
    game.headers["White"]       = "Eclipse (HalfKAv2)"
    game.headers["Black"]       = "Eclipse (HalfKAv2)"
    game.headers["TimeControl"] = f"{args.time_per_move:g}s/move"
    node = game
    move_history: list[str] = []

    time_ms = int(round(args.time_per_move * 1000))
    t0 = time.time()
    for ply in range(args.max_plies):
        if board.is_game_over(claim_draw=True):
            break
        side    = "W" if board.turn == chess.WHITE else "B"
        t_move  = time.time()
        stats   = engine.go(move_history, time_ms)
        elapsed = time.time() - t_move
        san     = board.san(stats.move)

        # Pretty-print one line per move. The AB line, if any, goes on its own
        # indented sub-line so the main move table stays scannable.
        print(f"  ply {ply+1:>3} {side} {san:<7} "
              f"score={stats.score_cp}cp nodes={stats.nodes} nps={stats.nps} "
              f"({elapsed:.1f}s)", file=sys.stderr, flush=True)
        if stats.ab_line:
            print(f"           {stats.ab_line}", file=sys.stderr, flush=True)

        comment = f"{stats.score_cp}cp n{stats.nodes}"
        if stats.ab_line:
            # Strip the "info string AB " prefix for PGN compactness.
            comment += " | " + stats.ab_line.removeprefix("info string AB ").strip()
        node = node.add_variation(stats.move, comment=comment)
        board.push(stats.move)
        move_history.append(stats.move.uci())

    game.headers["Result"] = board.result(claim_draw=True)
    # claim_draw=True means the loop exited as soon as 3-fold rep or the
    # 50-move rule was claimable, so check those before the automatic-draw
    # conditions in the termination string.
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
