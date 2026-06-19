#!/usr/bin/env bash
# Fill empty watcher slots with the next unstarted/uncompleted month,
# going from 2026-05 backwards to 2017-01.
# Quality: all Lichess [%eval] annotations are Stockfish depth ≥22 (cloud eval).
# Called by the cron monitor every 30 minutes to keep all 5 slots busy.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO/scripts"
DATA="$REPO/data"
MAX_SLOTS=20  # keep 20 months queued; curl contention is handled by backoff

# All months newest-first, back to 2017-01 (~115 months total).
all_months=()
for year in $(seq 2026 -1 2017); do
    if [ "$year" -eq 2026 ]; then max_m=5; else max_m=12; fi
    for m in $(seq "$max_m" -1 1); do
        all_months+=( "$(printf '%04d-%02d' "$year" "$m")" )
    done
done

# Scan months in order: skip completed ones, track running ones, launch until
# we have MAX_SLOTS months queued (running + newly launched).
queued=0
launched=0
for MONTH in "${all_months[@]}"; do
    [ "$queued" -lt "$MAX_SLOTS" ] || break

    MM="${MONTH//-/_}"
    OUTFILE="$DATA/eval_${MM}.txt"
    LOGFILE="$REPO/extract_eval_${MONTH}.log"
    PIDFILE="$REPO/extract_eval_${MONTH}.pid"

    # Already finished — don't count toward queue, just skip.
    grep -q "COMPLETE" "$LOGFILE" 2>/dev/null && continue

    # Already running — counts toward queue, move on.
    if [ -f "$PIDFILE" ]; then
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            queued=$((queued + 1))
            continue
        fi
    fi

    # Not running and not complete — launch it.
    echo "launching $MONTH"
    nohup bash "$SCRIPTS/run_month_eval.sh" \
        "$MONTH" "$OUTFILE" "$LOGFILE" "$PIDFILE" > /dev/null 2>&1 &
    queued=$((queued + 1))
    launched=$((launched + 1))
    sleep 5
done

echo "queued=$queued launched=$launched"
