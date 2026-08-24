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
# To test a NET rather than a code change, pass the same binary twice and give
# each side its own EvalFile with -N/-O. That holds the code exactly constant,
# which two separately-built binaries cannot promise.
#
# Options (defaults tuned for the 8-core dev iMac: 4 P-cores + 4 E-cores):
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
#   -n NNUE      value net, both sides (default data/eclipse.nnue)
#   -M TREEMB    MctsTreeMB/engine     (default 256)
#   -x "ARGS"    extra setoptions for BOTH engines, cutechess option.X=Y syntax
#   -N "ARGS"    extra setoptions for the NEW side only
#   -O "ARGS"    extra setoptions for the OLD side only
#   -l LABEL     run-directory suffix, e.g. -l net_narrow_vs_wide
#   -s           smoke test: 4 games at st=0.15, no SPRT (verifies wiring)
#
# Output lands in dev/sprt_runs/<timestamp>[_LABEL]/ (games.pgn + log.txt).
# The directory is gitignored; keep the log line with the final LLR/Elo when
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
# Overridable, and falls back to whatever book this checkout actually has.
# The hardcoded UHO_Lichess_4852_v1.epd is not in the repo and is not on the
# dev box; every run had to be launched with the name edited in by hand or it
# died at the `-f "$BOOK"` guard below before playing a game.
BOOK="${BOOK:-}"
if [[ -z "$BOOK" ]]; then
    for cand in "$REPO/data/books/UHO_Lichess_4852_v1.epd" \
                "$REPO/data/books/UHO_4060_v2.epd"; do
        [[ -f "$cand" ]] && { BOOK="$cand"; break; }
    done
    BOOK="${BOOK:-$REPO/data/books/UHO_Lichess_4852_v1.epd}"
fi
# Overridable: the deployment box does not necessarily keep tablebases where the
# dev iMac does, and a wrong path silently drops Syzygy adjudication (the
# ADJUDICATE array below strips those two entries when the directory is absent)
# rather than failing, which quietly lengthens every endgame in the run.
SYZYGY="${SYZYGY:-/Users/ezra/syzygy}"
TREEMB=256
EXTRA=""
NEW_EXTRA=""
OLD_EXTRA=""
LABEL=""
SMOKE=0

usage() { sed -n '2,39p' "$0"; exit 1; }

[[ $# -lt 2 ]] && usage
NEW="$1"; OLD="$2"; shift 2

while getopts "t:0:1:g:c:T:A:n:M:x:N:O:l:s" opt; do
    case $opt in
        t) TC="$OPTARG" ;;
        0) ELO0="$OPTARG" ;;
        1) ELO1="$OPTARG" ;;
        g) MAXGAMES="$OPTARG" ;;
        c) CONC="$OPTARG" ;;
        T) THREADS="$OPTARG" ;;
        A) ABTHREADS="$OPTARG" ;;
        n) NNUE="$OPTARG" ;;
        M) TREEMB="$OPTARG" ;;
        x) EXTRA="$OPTARG" ;;
        N) NEW_EXTRA="$OPTARG" ;;
        O) OLD_EXTRA="$OPTARG" ;;
        l) LABEL="$OPTARG" ;;
        s) SMOKE=1 ;;
        *) usage ;;
    esac
done

for bin in "$NEW" "$OLD"; do
    [[ -x "$bin" ]] || { echo "error: engine binary not executable: $bin" >&2; exit 1; }
done
[[ -f "$NNUE" ]] || { echo "error: NNUE net not found: $NNUE" >&2; exit 1; }
[[ -f "$BOOK" ]] || { echo "error: opening book not found: $BOOK" >&2; exit 1; }

# ---- preflight: prove each side actually loaded the net we asked it to ------
# The net architecture is a COMPILE-TIME constant (kFtOutSize in
# accumulator.hpp, kL1OutSize/kL2OutSize in nnue.hpp). `setoption EvalFile`
# cannot change it: handed a net of the wrong shape the engine prints
# "NNUE load failed: arch mismatch" and then keeps running on whatever net it
# already had. That is silent from cutechess's point of view -- a 2026-08-12
# run spent 2.5 hours and 30 games measuring the wide net against itself.
# So: boot each binary with its net and refuse to start unless the last NNUE
# line is a successful load of exactly that file.
preflight() {
    local bin="$1" net="$2" who="$3" out last
    out=$(printf 'uci\nsetoption name EvalFile value %s\nisready\nquit\n' "$net" \
          | "$bin" 2>&1 || true)
    last=$(echo "$out" | grep -E "NNUE (loaded|load failed)" | tail -1)
    if [[ "$last" != *"NNUE loaded: $net"* ]]; then
        echo "error: $who ($bin) did not load $net" >&2
        echo "       last NNUE line: ${last:-<none>}" >&2
        echo "       a net whose architecture does not match the binary is" >&2
        echo "       rejected; rebuild the binary for that architecture." >&2
        exit 1
    fi
    echo "  preflight ok: $who -> $(echo "$last" | sed 's/.*(\(.*\)).*/\1/')"
}

# Whichever EvalFile each side will really end up with: the per-engine
# override if one was given, otherwise the shared -n net.
new_net="$NNUE"; old_net="$NNUE"
[[ "$NEW_EXTRA" == *EvalFile=* ]] && new_net="${NEW_EXTRA##*EvalFile=}" && new_net="${new_net%% *}"
[[ "$OLD_EXTRA" == *EvalFile=* ]] && old_net="${OLD_EXTRA##*EvalFile=}" && old_net="${old_net%% *}"
preflight "$NEW" "$new_net" "new"
preflight "$OLD" "$old_net" "old"

OUT="$REPO/dev/sprt_runs/$(date +%Y%m%d_%H%M%S)${LABEL:+_$LABEL}"
mkdir -p "$OUT"

# Both sides get identical settings; only the binary (or the -N/-O overrides)
# differs. The default Threads=1 + AbThreads=1 exercises the full MCTS+AB
# hybrid in sequential mode (one core per engine); -T 2 -A 1 switches to the
# production-style parallel topology (dedicated AB thread alongside MCTS).
#
# restart=on tears the engine process down between games. The node pool never
# returns slab memory to the OS, so without this a long run's RSS is the
# high-water mark of every game so far -- a 600+15 pair was observed at 3.07 GB
# each on a 16 GB machine. MctsTreeMB caps the live tree on top of that;
# together they make per-process memory a constant, which is what lets several
# games run concurrently without the machine falling into swap and corrupting
# the time control for both sides.
COMMON_OPTS=(
    proto=uci
    tc="$TC"
    timemargin=100
    restart=on
    option.Threads="$THREADS"
    option.AbThreads="$ABTHREADS"
    option.Hash=128
    option.MctsHash=64
    option.MctsTreeMB="$TREEMB"
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

# Adjudication. Syzygy ends dead endings the moment they enter 5-piece range
# instead of grinding out the 50-move rule -- a 138-move rook ending cost ~35
# minutes of a 600+15 pair for a result the tablebase knew at move 133.
# -resign is only safe now that MCTS reports real centipawns rather than
# tanh-compressed ones; before that fix nothing ever reached 600.
# -draw stays deliberately tight (score=10): both nets share a ~1.5-pawn
# misjudgement of 3v2 rook endings, so a loose threshold would adjudicate real
# positions as draws on a shared error rather than on agreement.
ADJUDICATE=(
    -tb "$SYZYGY" -tbpieces 5
    -draw movenumber=40 movecount=8 score=10
    -resign movecount=4 score=600
    -maxmoves 250
)
[[ -d "$SYZYGY" ]] || ADJUDICATE=("${ADJUDICATE[@]:2}")

# Per-side extra setoptions. These are usually EMPTY, and that is the case that
# used to break: macOS ships bash 3.2, where `"${arr[@]}"` on an empty array is
# an unbound-variable error under `set -u`. The 2026-08-12 queue died on exactly
# that at launch ("NEW_OPTS[@]: unbound variable"), so leg 2 measured nothing and
# left a 0-byte log. The `${arr[@]+...}` guard below expands to nothing at all
# when the array is empty instead of tripping `set -u`; it is used at the
# cutechess call site too.
# shellcheck disable=SC2206  # word-splitting of the option string is intentional
NEW_OPTS=($NEW_EXTRA)
# shellcheck disable=SC2206
OLD_OPTS=($OLD_EXTRA)

echo "SPRT: $NEW vs $OLD"
echo "  tc=$TC elo0=$ELO0 elo1=$ELO1 max_games=$MAXGAMES conc=$CONC"
echo "  threads=$THREADS abthreads=$ABTHREADS treemb=$TREEMB"
echo "  book=$BOOK"
[[ -n "$NEW_EXTRA" ]] && echo "  new-only: $NEW_EXTRA"
[[ -n "$OLD_EXTRA" ]] && echo "  old-only: $OLD_EXTRA"
echo "  out=$OUT"

cutechess-cli \
    -engine name=new cmd="$NEW" dir="$REPO" ${NEW_OPTS[@]+"${NEW_OPTS[@]}"} \
    -engine name=old cmd="$OLD" dir="$REPO" ${OLD_OPTS[@]+"${OLD_OPTS[@]}"} \
    -each "${COMMON_OPTS[@]}" \
    "${MODE_ARGS[@]}" \
    -openings file="$BOOK" format=epd order=random \
    -repeat \
    "${ADJUDICATE[@]}" \
    -recover \
    -concurrency "$CONC" \
    -ratinginterval 10 \
    -pgnout "$OUT/games.pgn" \
    2>&1 | tee "$OUT/log.txt"
