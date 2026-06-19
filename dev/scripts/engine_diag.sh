#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Live engine diagnostics: nps / depth / seldepth / score from one search.
#
#   scripts/engine_diag.sh <net.nnue> [movetime_ms] [threads]
#
# Examples:
#   scripts/engine_diag.sh /tmp/eclipse_v7_ep2c5.nnue
#   scripts/engine_diag.sh /tmp/eclipse_v7_ep2c3.nnue 5000 4
set -euo pipefail

NET=${1:?usage: engine_diag.sh <net.nnue> [movetime_ms] [threads]}
MOVETIME=${2:-3000}
THREADS=${3:-4}
SYZYGY=${SYZYGY:-/Users/ezra/syzygy}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$REPO/build/src/eclipse"
SLEEP_S=$(awk "BEGIN{print $MOVETIME/1000 + 1.5}")

{
  printf 'uci\n'
  printf 'setoption name EvalFile value %s\n' "$NET"
  printf 'setoption name Threads value %s\n' "$THREADS"
  printf 'setoption name SyzygyPath value %s\n' "$SYZYGY"
  printf 'isready\n'
  printf 'position startpos\n'
  printf 'go movetime %s\n' "$MOVETIME"
  sleep "$SLEEP_S"
  printf 'quit\n'
} | "$ENGINE" 2>&1 | grep -E '^info (depth|string)'
