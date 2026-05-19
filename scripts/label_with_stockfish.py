#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Stockfish-driven label generator for HalfKAv2 training.

Reads FENs (one per line) from --fens or stdin, runs Stockfish at the given
depth, and writes `<fen>;<score_cp>` lines to --out for consumption by
scripts/train_halfkav2.py.

Mate scores are converted to a saturating cp value (kMateCp) so they don't
break the sigmoid loss. Positions that Stockfish refuses (illegal/checkmated)
are skipped with a warning.

Typical use:

    # Sample 200k unique FENs from a PGN dump (one easy way to get positions)
    python scripts/label_with_stockfish.py --fens fens.txt --out training.txt \\
        --depth 8 --stockfish-path /opt/homebrew/bin/stockfish

For a serious PoC you want ~1M positions at depth 8-10. That's ~30 minutes of
Stockfish time on a fast modern CPU - run it overnight on a real corpus.

The script is deliberately single-threaded UCI driver - if you need throughput,
shard the FEN file across processes and concatenate the outputs.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

# Saturating value for mate-in-N scores. Sigmoid(1500/410) ~= 0.97, which is
# close enough to a "winning" target without making the loss explode.
MATE_CP = 1500

# Pattern matches Stockfish "info ... score cp <N>" or "score mate <N>".
_SCORE_RE = re.compile(r"score (cp|mate) (-?\d+)")


def stream_fens(args) -> Iterable[str]:
    """Yields stripped FEN lines from --fens or stdin. Skips blanks / '#' comments."""
    if args.fens is not None:
        f = args.fens.open("r")
        close = True
    else:
        f = sys.stdin
        close = False
    try:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line
    finally:
        if close:
            f.close()


class Stockfish:
    """Minimal blocking UCI wrapper. Each `evaluate(fen, depth)` runs
    `position fen` + `go depth` and reads through `bestmove`, returning the
    last cp score reported.

    Not optimized: spawns one process for the lifetime of this script and
    serializes evaluations through stdin/stdout. Good enough for offline
    labeling at any sane corpus size."""

    def __init__(self, path: str, threads: int = 1, hash_mb: int = 64):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, line: str):
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str) -> list[str]:
        """Reads stdout until a line containing `token` appears. Returns all
        lines seen (including the terminating one)."""
        assert self.proc.stdout is not None
        out = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish exited unexpectedly")
            out.append(line.rstrip())
            if token in line:
                return out

    def evaluate(self, fen: str, depth: int) -> Optional[int]:
        """Returns the last reported cp score from Stockfish's perspective of
        the side to move, or None if the position is terminal (no bestmove
        or no score line)."""
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        lines = self._wait_for("bestmove")
        score = None
        for line in lines:
            m = _SCORE_RE.search(line)
            if m:
                kind, val = m.group(1), int(m.group(2))
                if kind == "cp":
                    score = val
                else:  # mate
                    score = MATE_CP if val > 0 else -MATE_CP
        return score

    def close(self):
        if self.proc.poll() is None:
            try:
                self._send("quit")
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fens", type=Path, default=None,
                   help="text file of FENs (one per line); reads stdin if omitted")
    p.add_argument("--out", type=Path, required=True,
                   help="output training file (<fen>;<score_cp>)")
    p.add_argument("--depth", type=int, default=8,
                   help="Stockfish search depth per position (default 8). 8-10 "
                        "is the sweet spot for PoC training speed vs. label quality.")
    p.add_argument("--stockfish-path", type=str,
                   default=shutil.which("stockfish") or "stockfish")
    p.add_argument("--threads", type=int, default=1,
                   help="Stockfish threads per evaluation (1 is fine for shallow search)")
    p.add_argument("--hash-mb", type=int, default=64)
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many positions (useful for quick test runs)")
    p.add_argument("--report-every", type=int, default=1000)
    args = p.parse_args()

    sf = Stockfish(args.stockfish_path, threads=args.threads, hash_mb=args.hash_mb)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    started = time.time()
    try:
        with args.out.open("w") as outf:
            for fen in stream_fens(args):
                if args.limit is not None and written >= args.limit:
                    break
                try:
                    score = sf.evaluate(fen, args.depth)
                except RuntimeError as e:
                    print(f"engine error on '{fen}': {e}", file=sys.stderr)
                    break
                if score is None:
                    skipped += 1
                    continue
                outf.write(f"{fen};{score}\n")
                written += 1
                if written % args.report_every == 0:
                    elapsed = time.time() - started
                    rate = written / max(elapsed, 1e-6)
                    print(f"  {written} labeled, {skipped} skipped, "
                          f"{rate:.1f} pos/s")
    finally:
        sf.close()

    elapsed = time.time() - started
    print(f"done: {written} positions written to {args.out} "
          f"({skipped} skipped, {elapsed:.1f}s total)")


if __name__ == "__main__":
    main()
