#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Full benchmark runner:
  1. HalfKAv2 (improved search) vs 3.26M model (old HalfKP engine) — via Stockfish calibration
  2. Improved search vs baseline search (both on HalfKAv2) — head-to-head
  3. All engines vs Stockfish at multiple levels — absolute Elo calibration

Results saved to results/benchmark/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from match_runner import (
    UCIEngine, StockfishEngine, run_match, calibrate_elo,
    estimate_elo_from_calibration, elo_diff
)
from pathlib import Path
import time

# ---------------------------------------------------------------------------
# Paths — all absolute so the script runs from any cwd
# ---------------------------------------------------------------------------
ROOT       = Path("/Users/ezra/eclipse-chess-engine")
DATA       = ROOT / "data"
RESULTS    = ROOT / "results" / "benchmark"

ENGINE_NEW  = str(ROOT / "build/src/eclipse")          # improved search
ENGINE_BASE = "/tmp/eclipse-baseline-build/src/eclipse" # baseline (pre-improvements)
ENGINE_OLD  = "/tmp/eclipse-halfkp-build/src/eclipse"  # old HalfKP engine

EVAL_NEW    = str(DATA / "eclipse_halfkav2_v2.nnue")   # halfkav2.pt converted
EVAL_OLD_HP = str(DATA / "eclipse_v3_3.26M.nnue")      # old 3.26M HalfKP model

STOCKFISH   = "stockfish"

# Benchmark parameters
MOVETIME_MS  = 500   # 500ms/move
GAMES_H2H    = 40    # head-to-head games (improved vs baseline, same NNUE)
GAMES_CAL    = 20    # games per Stockfish level for calibration
SF_ELOS      = [800, 1200, 1600, 2000, 2400]
THREADS      = 6     # MCTS threads (5 workers + 1 AB in parallel mode)
HASH_MB      = 64

RESULTS.mkdir(parents=True, exist_ok=True)

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(msg)
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# 1. Search improvement: improved vs baseline (same HalfKAv2 NNUE)
# ---------------------------------------------------------------------------
log("=" * 65)
log("PHASE 1: Search improvement — HalfKAv2 improved vs baseline")
log("=" * 65)

e_impr  = UCIEngine(ENGINE_NEW.split(),  name_hint="HalfKAv2-ImprSearch")
e_base  = UCIEngine(ENGINE_BASE.split(), name_hint="HalfKAv2-Baseline")
e_impr.start();  e_base.start()
e_impr.uci_init(eval_file=EVAL_NEW, hash_mb=HASH_MB, threads=THREADS, ab_threads=1)
e_base.uci_init(eval_file=EVAL_NEW, hash_mb=HASH_MB, threads=1, ab_threads=1)

h2h_search = run_match(e_impr, e_base, GAMES_H2H, MOVETIME_MS,
                       out_pgn=str(RESULTS / "improved_vs_baseline.pgn"),
                       label_e1="ImprovedSearch", label_e2="BaselineSearch")
e_impr.quit(); e_base.quit()

h2h_diff, h2h_err = elo_diff(h2h_search.wins_e1, h2h_search.wins_e2, h2h_search.draws)
log(f"\n  Search improvement Elo delta: {h2h_diff:+.0f} ± {h2h_err:.0f}")

# ---------------------------------------------------------------------------
# 2. Calibrate HalfKAv2 (improved) vs Stockfish
# ---------------------------------------------------------------------------
log("\n" + "=" * 65)
log("PHASE 2: Elo calibration — HalfKAv2 improved search vs Stockfish")
log("=" * 65)

cal_new = calibrate_elo(
    eclipse_bin=ENGINE_NEW,
    eval_file=EVAL_NEW,
    stockfish_bin=STOCKFISH,
    sf_elos=SF_ELOS,
    games_per_level=GAMES_CAL,
    movetime_ms=MOVETIME_MS,
    engine_label="HalfKAv2-Impr",
    threads=THREADS,
    hash_mb=HASH_MB,
)
elo_new = estimate_elo_from_calibration(cal_new)

# ---------------------------------------------------------------------------
# 3. Calibrate HalfKAv2 (baseline) vs Stockfish
# ---------------------------------------------------------------------------
log("\n" + "=" * 65)
log("PHASE 3: Elo calibration — HalfKAv2 baseline search vs Stockfish")
log("=" * 65)

cal_base = calibrate_elo(
    eclipse_bin=ENGINE_BASE,
    eval_file=EVAL_NEW,
    stockfish_bin=STOCKFISH,
    sf_elos=SF_ELOS,
    games_per_level=GAMES_CAL,
    movetime_ms=MOVETIME_MS,
    engine_label="HalfKAv2-Base",
    threads=1,
    hash_mb=32,
)
elo_base = estimate_elo_from_calibration(cal_base)

# ---------------------------------------------------------------------------
# 4. Calibrate old 3.26M HalfKP model vs Stockfish (old engine)
# ---------------------------------------------------------------------------
log("\n" + "=" * 65)
log("PHASE 4: Elo calibration — 3.26M HalfKP model (old engine) vs Stockfish")
log("=" * 65)

cal_old = calibrate_elo(
    eclipse_bin=ENGINE_OLD,
    eval_file=EVAL_OLD_HP,
    stockfish_bin=STOCKFISH,
    sf_elos=SF_ELOS,
    games_per_level=GAMES_CAL,
    movetime_ms=MOVETIME_MS,
    engine_label="3.26M-HalfKP",
)
elo_old = estimate_elo_from_calibration(cal_old)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
lines = [
    "",
    "=" * 65,
    "ECLIPSE BENCHMARK SUMMARY",
    f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
    "=" * 65,
    "",
    "SEARCH IMPROVEMENT (HalfKAv2 Improved vs Baseline, same NNUE)",
    f"  +{h2h_search.wins_e1} ={h2h_search.draws} -{h2h_search.wins_e2}"
    f"  ({h2h_search.games_played} games)",
    f"  Score: {h2h_search.score_rate_e1()*100:.1f}%",
    f"  Elo delta (improved - baseline): {h2h_diff:+.0f} ± {h2h_err:.0f}",
    "",
    "ELO CALIBRATION vs STOCKFISH",
    "",
    "  HalfKAv2 (improved search):",
]
for sf_elo, stats in cal_new:
    diff, err = elo_diff(stats.wins_e1, stats.wins_e2, stats.draws)
    rate = stats.score_rate_e1()
    lines.append(f"    vs SF~{sf_elo:4d}: {rate*100:5.1f}%  diff {diff:+.0f}±{err:.0f}")
if elo_new:
    lines.append(f"  => Estimated Elo: ~{elo_new:.0f}")

lines += ["", "  HalfKAv2 (baseline search):"]
for sf_elo, stats in cal_base:
    diff, err = elo_diff(stats.wins_e1, stats.wins_e2, stats.draws)
    rate = stats.score_rate_e1()
    lines.append(f"    vs SF~{sf_elo:4d}: {rate*100:5.1f}%  diff {diff:+.0f}±{err:.0f}")
if elo_base:
    lines.append(f"  => Estimated Elo: ~{elo_base:.0f}")

lines += ["", "  3.26M HalfKP (old engine, original search):"]
for sf_elo, stats in cal_old:
    diff, err = elo_diff(stats.wins_e1, stats.wins_e2, stats.draws)
    rate = stats.score_rate_e1()
    lines.append(f"    vs SF~{sf_elo:4d}: {rate*100:5.1f}%  diff {diff:+.0f}±{err:.0f}")
if elo_old:
    lines.append(f"  => Estimated Elo: ~{elo_old:.0f}")

if elo_new and elo_old:
    lines += [
        "",
        "TOTAL IMPROVEMENT (new model + search vs old 3.26M model):",
        f"  Estimated: {elo_new - elo_old:+.0f} Elo  (~{elo_new:.0f} vs ~{elo_old:.0f})",
    ]
if elo_new and elo_base:
    lines += [
        f"  Search alone: {elo_new - elo_base:+.0f} Elo"
        f"  (also cross-check head-to-head: {h2h_diff:+.0f}±{h2h_err:.0f})",
    ]

lines += ["", "=" * 65]

for l in lines:
    log(l)

summary_path = RESULTS / "benchmark_summary.txt"
with open(summary_path, "w") as f:
    f.write("\n".join(log_lines) + "\n")
print(f"\nFull log → {summary_path}")
