#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-engine search depth and time-per-move, read from cutechess PGN comments.

cutechess writes `{eval/depth time}` after each move; this aggregates depth and
time by engine across all completed games.

    python3 scripts/match_depth.py /tmp/ep2c5_vs_ep2c3_5p3.pgn
"""
import re
import sys
import chess.pgn
from collections import defaultdict

_COMMENT = re.compile(r'([-+]?\d+\.\d+|[-+]?M\d+)/(\d+)\s+([\d.]+)s')


def main(path):
    depth = defaultdict(list)
    tsec = defaultdict(list)
    with open(path) as f:
        while (g := chess.pgn.read_game(f)):
            if g.headers.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
                continue
            names = {chess.WHITE: g.headers["White"], chess.BLACK: g.headers["Black"]}
            node, stm = g, chess.WHITE
            while node.variations:
                node = node.variation(0)
                m = _COMMENT.search(node.comment or "")
                if m:
                    depth[names[stm]].append(int(m.group(2)))
                    tsec[names[stm]].append(float(m.group(3)))
                stm = not stm
    if not depth:
        print("  (no move comments found)")
        return
    for e in sorted(depth):
        dd, tt = depth[e], tsec[e]
        print(f"  {e:>6}: depth avg={sum(dd)/len(dd):.1f} max={max(dd)} | "
              f"time/move avg={sum(tt)/len(tt):.2f}s max={max(tt):.1f}s  "
              f"({len(dd)} moves)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: match_depth.py <pgn>")
    main(sys.argv[1])
