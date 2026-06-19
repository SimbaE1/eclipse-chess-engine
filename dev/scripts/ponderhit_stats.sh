#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Ponderhit count + average nps from a cutechess -debug log.
#
# ponderhits and nps are NOT in the PGN. To capture them, launch the match
# with cutechess's -debug flag and tee its output, e.g.:
#
#   cutechess-cli -debug <...engines...> 2>&1 | tee /tmp/match_debug.log
#
# then:
#   scripts/ponderhit_stats.sh /tmp/match_debug.log
set -euo pipefail

LOG=${1:?usage: ponderhit_stats.sh <cutechess_debug.log>}
[ -f "$LOG" ] || { echo "no such log: $LOG" >&2; exit 1; }

echo "  ponderhits sent : $(grep -c 'ponderhit' "$LOG" || true)"
grep -oE 'nps [0-9]+' "$LOG" \
  | awk '{s+=$2; n++} END{if(n) printf "  avg nps         : %d  (%d samples)\n", s/n, n; else print "  avg nps         : (no nps lines)"}'
echo "  bestmove 0000   : $(grep -c 'bestmove 0000' "$LOG" || true)"
