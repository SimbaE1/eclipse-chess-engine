#!/usr/bin/env python3
"""Eclipse vs Stockfish 1000–1800 Elo, 1s/move, 6 threads."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from match_runner import UCIEngine, StockfishEngine, run_match, elo_diff, calibrate_elo, estimate_elo_from_calibration
from pathlib import Path

ROOT     = Path("/Users/ezra/eclipse-chess-engine")
ENGINE   = str(ROOT / "build/src/eclipse")
EVAL     = str(ROOT / "data/eclipse_halfkav2_v2.nnue")
RESULTS  = ROOT / "results" / "benchmark"
RESULTS.mkdir(parents=True, exist_ok=True)

MOVETIME_MS = 1000
THREADS     = 6
HASH_MB     = 128
GAMES       = 20
SF_ELOS     = [1000, 1200, 1400, 1600, 1800]

log_lines = []
def log(msg=""):
    print(msg, flush=True)
    log_lines.append(msg)

log("=" * 65)
log(f"Eclipse vs Stockfish — 1s/move, {THREADS} threads")
log("=" * 65)

results = calibrate_elo(
    eclipse_bin=ENGINE,
    eval_file=EVAL,
    stockfish_bin="stockfish",
    sf_elos=SF_ELOS,
    games_per_level=GAMES,
    movetime_ms=MOVETIME_MS,
    engine_label="Eclipse",
    threads=THREADS,
    hash_mb=HASH_MB,
)

log("\n" + "=" * 65)
log("Summary:")
for sf_elo, stats in results:
    diff, err = elo_diff(stats.wins_e1, stats.wins_e2, stats.draws)
    pct = stats.score_rate_e1() * 100
    log(f"  vs SF~{sf_elo}: {stats.wins_e1}W {stats.draws}D {stats.wins_e2}L  "
        f"({pct:.1f}%)  Elo diff: {diff:+.0f} ±{err:.0f}")

estimated = estimate_elo_from_calibration(results)
if estimated:
    log(f"\n  Estimated Eclipse Elo: ~{estimated:.0f}")

log("\nDone.")

out_path = RESULTS / "bench_1000_1800_summary.txt"
out_path.write_text("\n".join(log_lines))
log(f"Summary → {out_path}")
