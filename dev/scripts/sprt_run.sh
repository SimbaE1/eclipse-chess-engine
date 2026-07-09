#!/usr/bin/env bash
# SPRT self-play harness: is <new> stronger than <old>?
#
# Wraps cutechess-cli in a sequential probability ratio test. The run stops
# by itself as soon as the result is statistically decided:
#   H1 accepted (LLR >= ~2.94)  -> new is at least ELO1 stronger: ship it.
#   H0 accepted (LLR <= ~-2.94) -> gain (if any) is below ELO0: reject.
# A run that hits -games without crossing either bound is inconclusive; the
# change is probably worth < ELO1 Elo.
#
# Usage:
#   dev/scripts/sprt_run.sh <new_binary> <old_binary> [options]
#
# Options (defaults tuned for the 8-core dev iMac):
#   -t TC        time control          (default 20+0.2)
#   -0 ELO0      H0: gain <= this      (default 0)
#   -1 ELO1      H1: gain >= this      (default 5)
#   -g GAMES     max games             (default 2000)
#   -c CONC      concurrent games      (default 4)
#   -T THREADS   MCTS threads/engine   (default 1; sequential AB. 2+ with
#                                       -A 1 runs AB in PARALLEL like the
#                                       production bot -- budget ~cores /
#                                       (THREADS+ABTHREADS) for -c)
#   -A ABTHREADS AB threads/engine     (default 1)
#   -n NNUE      value net             (default data/eclipse.nnue)
#   -x "ARGS"    extra setoptions for BOTH engines, cutechess option.X=Y syntax
#   -s           smoke test: 4 games at st=0.15, no SPRT (verifies wiring)
#
# Output lands in dev/sprt_runs/<timestamp>/ (games.pgn + log.txt). The
# directory is gitignored; keep the log line with the final LLR/Elo when
# recording results.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"

TC="20+0.2"
ELO0=0
ELO1=5
MAXGAMES=2000
CONC=4
THREADS=1
ABTHREADS=1
NNUE="$REPO/data/eclipse.nnue"
BOOK="$REPO/data/books/UHO_Lichess_4852_v1.epd"
EXTRA=""
SMOKE=0

usage() { sed -n '2,30p' "$0"; exit 1; }

[[ $# -lt 2 ]] && usage
NEW="$1"; OLD="$2"; shift 2

while getopts "t:0:1:g:c:T:A:n:x:s" opt; do
    case $opt in
        t) TC="$OPTARG" ;;
        0) ELO0="$OPTARG" ;;
        1) ELO1="$OPTARG" ;;
        g) MAXGAMES="$OPTARG" ;;
        c) CONC="$OPTARG" ;;
        T) THREADS="$OPTARG" ;;
        A) ABTHREADS="$OPTARG" ;;
        n) NNUE="$OPTARG" ;;
        x) EXTRA="$OPTARG" ;;
        s) SMOKE=1 ;;
        *) usage ;;
    esac
done

for bin in "$NEW" "$OLD"; do
    [[ -x "$bin" ]] || { echo "error: engine binary not executable: $bin" >&2; exit 1; }
done
[[ -f "$NNUE" ]] || { echo "error: NNUE net not found: $NNUE" >&2; exit 1; }
[[ -f "$BOOK" ]] || { echo "error: opening book not found: $BOOK" >&2; exit 1; }

OUT="$REPO/dev/sprt_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# Both sides get identical settings; only the binary differs. The default
# Threads=1 + AbThreads=1 exercises the full MCTS+AB hybrid in sequential
# mode (one core per engine); -T 2 -A 1 switches to the production-style
# parallel topology (dedicated AB thread alongside MCTS).
COMMON_OPTS=(
    proto=uci
    tc="$TC"
    timemargin=100
    option.Threads="$THREADS"
    option.AbThreads="$ABTHREADS"
    option.Hash=128
    option.MctsHash=64
    option.EvalFile="$NNUE"
)
# shellcheck disable=SC2206  # word-splitting of EXTRA is intentional
COMMON_OPTS+=($EXTRA)

MODE_ARGS=(-sprt elo0="$ELO0" elo1="$ELO1" alpha=0.05 beta=0.05
           -games 2 -rounds $((MAXGAMES / 2)))
if [[ $SMOKE -eq 1 ]]; then
    # Fixed 150ms/move, 4 games, no SPRT: just prove the plumbing works.
    COMMON_OPTS=("${COMMON_OPTS[@]/tc=$TC/st=0.15}")
    MODE_ARGS=(-games 2 -rounds 2)
fi

echo "SPRT: $NEW vs $OLD"
echo "  tc=$TC elo0=$ELO0 elo1=$ELO1 max_games=$MAXGAMES conc=$CONC"
echo "  out=$OUT"

cutechess-cli \
    -engine name=new cmd="$NEW" dir="$REPO" \
    -engine name=old cmd="$OLD" dir="$REPO" \
    -each "${COMMON_OPTS[@]}" \
    "${MODE_ARGS[@]}" \
    -openings file="$BOOK" format=epd order=random \
    -repeat \
    -draw movenumber=40 movecount=8 score=10 \
    -resign movecount=3 score=600 \
    -recover \
    -concurrency "$CONC" \
    -ratinginterval 10 \
    -pgnout "$OUT/games.pgn" \
    2>&1 | tee "$OUT/log.txt"
