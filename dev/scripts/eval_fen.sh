#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Probe a net's eval on one FEN, or every FEN/EPD line in a file.
#
#   scripts/eval_fen.sh <net.nnue> "<fen>" [movetime_ms] [threads]
#   scripts/eval_fen.sh <net.nnue> --file data/blindspot_g10.epd [movetime_ms] [threads]
#
# Prints the last info line (eval/depth/nps/pv) per position. Use it to compare
# an old vs new net on known blind-spot positions: a fixed eval blind spot
# shows up as a confidently-wrong score (e.g. +0.8 in a lost position).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$REPO/build/src/eclipse"
SYZYGY="${SYZYGY:-/Users/ezra/syzygy}"

NET=${1:?usage: eval_fen.sh <net.nnue> <fen|--file path> [movetime_ms] [threads]}
SRC=${2:?need a FEN or --file <path>}
if [ "$SRC" = "--file" ]; then FILE=${3:?--file needs a path}; MOVETIME=${4:-2000}; THREADS=${5:-4}
else FEN=$SRC; MOVETIME=${3:-2000}; THREADS=${4:-4}; fi

probe() {  # $1 = fen
  local fen=$1
  { printf 'uci\nsetoption name EvalFile value %s\n' "$NET"
    printf 'setoption name Threads value %s\n' "$THREADS"
    printf 'setoption name SyzygyPath value %s\nisready\n' "$SYZYGY"
    printf 'position fen %s\ngo movetime %s\n' "$fen" "$MOVETIME"
    sleep "$(awk "BEGIN{print $MOVETIME/1000 + 1.0}")"; printf 'quit\n'
  } | "$ENGINE" 2>/dev/null | grep -E '^info depth' | tail -1
}

if [ -n "${FILE:-}" ]; then
  # EPD: FEN is the first 4 fields; strip opcodes after them. Skip blanks.
  while IFS= read -r line; do
    [ -z "${line// }" ] && continue
    fen=$(echo "$line" | awk '{print $1, $2, $3, $4}')
    note=$(echo "$line" | sed -n 's/.*c0 "\([^"]*\)".*/\1/p')
    printf '%-72s\n  %s | %s\n' "$fen" "$(probe "$fen 0 1")" "$note"
  done < "$FILE"
else
  probe "$FEN"
fi
