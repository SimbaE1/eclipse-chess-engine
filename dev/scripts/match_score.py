#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Print per-game results and the running match score for a cutechess PGN.

    python3 scripts/match_score.py /tmp/ep2c5_vs_ep2c3_5p3.pgn
"""
import sys
import chess.pgn
from collections import defaultdict


def main(path):
    pts = defaultdict(float)
    wdl = defaultdict(lambda: [0, 0, 0])  # wins, draws, losses
    rows = []
    with open(path) as f:
        n = 0
        while (g := chess.pgn.read_game(f)):
            r = g.headers.get("Result")
            if r not in ("1-0", "0-1", "1/2-1/2"):
                continue
            n += 1
            w, b = g.headers["White"], g.headers["Black"]
            rows.append(f"  G{n:>2}: {w:>6} (W) vs {b:>6} (B)  {r}")
            if r == "1-0":
                pts[w] += 1; wdl[w][0] += 1; wdl[b][2] += 1
            elif r == "0-1":
                pts[b] += 1; wdl[b][0] += 1; wdl[w][2] += 1
            else:
                pts[w] += .5; pts[b] += .5; wdl[w][1] += 1; wdl[b][1] += 1
    print("\n".join(rows))
    print(f"\n  Score ({n} games):")
    for e in sorted(pts, key=lambda x: -pts[x]):
        win, draw, loss = wdl[e]
        print(f"    {e:>6}: {pts[e]:.1f}  (+{win} ={draw} -{loss})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: match_score.py <pgn>")
    main(sys.argv[1])
