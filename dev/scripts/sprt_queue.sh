#!/usr/bin/env bash
# Run the queued 180+2 SPRT legs one after another.
#
# Sequential, not parallel, on purpose. Each leg already uses every core the
# machine can give an engine without distorting its own time control; running
# two legs at once would halve the cores per game and change the very thing
# under test. Leg 2 starts when leg 1 crosses an SPRT bound or exhausts -g.
#
# Launch detached -- these legs run for days:
#   nohup dev/scripts/sprt_queue.sh > dev/sprt_runs/queue.log 2>&1 & disown
#
# The baseline is BUILT FROM A GIT REF, not read from a path someone had lying
# around. The previous version of this script hardcoded an absolute scratchpad
# directory from the session that wrote it; that path is gone, so the script was
# only ever runnable on one machine on one day. Now:
#
#   BASE_REF=origin/main dev/scripts/sprt_queue.sh
#
# checks that ref out into a git worktree under $SPRT_WORK, configures and
# builds it with the same CMake settings as the main build, and plays the
# current build against it. Any machine, any ref, no manual setup.
#
# Env overrides:
#   BASE_REF    git ref to use as "old"      (default origin/main)
#   SPRT_WORK   scratch dir for the baseline (default $REPO/.sprt-work)
#   TC/CONC/ELO0/ELO1/TREEMB/GAMES           (see the defaults below)

set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

BASE_REF="${BASE_REF:-origin/main}"
SPRT_WORK="${SPRT_WORK:-$REPO/.sprt-work}"

TC="${TC:-180+2}"
CONC="${CONC:-2}"   # 2 games x 2 engines x (1 MCTS + 1 AB) = 8 threads = 8 logical cores
ELO0="${ELO0:-0}"
ELO1="${ELO1:-10}"
GAMES="${GAMES:-1000}"
TREEMB="${TREEMB:-256}"

# [0,10] rather than the more usual [0,5]: at 180+2 with -c 2 this machine plays
# roughly 30 games/hour, and an elo1=5 test needs on the order of thousands of
# games to resolve. elo1=10 trades the ability to detect a very small gain for a
# run that finishes in days instead of weeks. A bundle this size should clear 10
# Elo or be reconsidered.

CUR="$REPO/build/src/eclipse"
NET="${NET:-$REPO/data/eclipse.nnue}"

banner() { echo; echo "==================== $* ===================="; date; echo; }

# ---- build the baseline from BASE_REF ---------------------------------------
BASE_SHA="$(git rev-parse --short "$BASE_REF")"
WT="$SPRT_WORK/base-$BASE_SHA"
BASE="$WT/build/src/eclipse"

banner "BASELINE  $BASE_REF ($BASE_SHA)"
if [[ -x "$BASE" ]]; then
    echo "  already built: $BASE"
else
    mkdir -p "$SPRT_WORK"
    [[ -d "$WT" ]] || git worktree add --detach "$WT" "$BASE_REF"
    # A fresh worktree does NOT inherit submodule contents, and extern/fathom is
    # a submodule -- without this the CMake configure fails on a missing
    # tbprobe.c rather than on anything to do with the engine.
    git -C "$WT" submodule update --init --recursive
    # Same flags as the main build (Release + -march=native). A baseline built
    # with different optimisation settings measures the compiler, not the code.
    cmake -S "$WT" -B "$WT/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$WT/build" -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
          --target eclipse >/dev/null
    echo "  built: $BASE"
fi

[[ -x "$CUR" ]]  || { echo "error: current build missing: $CUR (run cmake --build build)" >&2; exit 1; }
[[ -x "$BASE" ]] || { echo "error: baseline build failed: $BASE" >&2; exit 1; }

# ---- leg 1: the whole code bundle -------------------------------------------
# Net held constant on both sides, so this measures only the code: the TT
# rewrite (lockless hashing + 4-way buckets), the MCTS fixes (clamped
# AB-injected Q, real centipawns out of MCTS, bounded node pool), the AB search
# changes (lazy move selection, skip-depth lazy SMP, aspiration 50->15,
# search-driven time extension), and the time-management rework.
#
# If this fails, do NOT conclude the whole bundle is bad -- it is several
# independent bets in one binary. Leg 2 splits off the one that can be isolated
# without a rebuild; the rest need their own worktrees.
banner "LEG 1/2  code: this build vs $BASE_REF ($BASE_SHA)"
dev/scripts/sprt_run.sh "$CUR" "$BASE" \
    -t "$TC" -0 "$ELO0" -1 "$ELO1" -g "$GAMES" -c "$CONC" -M "$TREEMB" \
    -n "$NET" \
    -l "code_vs_$BASE_SHA"

# ---- leg 2: time management, isolated ---------------------------------------
# MoveHorizon is a UCI spin, so the single most behaviour-changing part of the
# bundle can be tested with ONE binary and no rebuild: same code both sides,
# only the divisor differs. 25 is the new default, 40 is what main effectively
# shipped (the MLH head was never live, so the clamp pinned the divisor to its
# floor of 40 every move of every game).
#
# This is the leg most likely to move: it changes how much clock every move
# gets, and D=8 flagged a game in 2026-06-18 testing.
banner "LEG 2/2  time management: MoveHorizon 25 vs 40 (same binary)"
dev/scripts/sprt_run.sh "$CUR" "$CUR" \
    -t "$TC" -0 "$ELO0" -1 "$ELO1" -g "$GAMES" -c "$CONC" -M "$TREEMB" \
    -n "$NET" \
    -N "option.MoveHorizon=25" \
    -O "option.MoveHorizon=40" \
    -l movehorizon_25_vs_40

banner "QUEUE COMPLETE"
