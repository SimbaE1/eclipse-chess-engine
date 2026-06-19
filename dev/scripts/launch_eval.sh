#!/usr/bin/env bash
# Launch SF-eval extraction for recent months, newest first.
# Uses run_month_eval.sh watchers with 1800+ ELO, 180s+ TC. Runs to EOF (no target cap).
# Stagger 15s between launches to avoid flooding the 5-connection cap.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO/scripts"
DATA="$REPO/data"

MONTHS=(
    2026-05 2026-04 2026-03 2026-02 2026-01
    2025-12 2025-11 2025-10 2025-09 2025-08
    2025-07 2025-06 2025-05 2025-04 2025-03
    2025-02 2025-01
)

for MONTH in "${MONTHS[@]}"; do
    MM="${MONTH//-/_}"
    OUTFILE="$DATA/eval_${MM}.txt"
    LOGFILE="$REPO/extract_eval_${MONTH}.log"
    PIDFILE="$REPO/extract_eval_${MONTH}.pid"

    # Skip if already complete (pid file gone = completed)
    if [ ! -f "$PIDFILE" ] && [ -f "$OUTFILE" ]; then
        lines=$(wc -l < "$OUTFILE" 2>/dev/null || echo 0)
        if [ "$lines" -gt 100000 ]; then
            echo "[$MONTH] looks done ($lines lines), skipping"
            continue
        fi
    fi

    if [ -f "$PIDFILE" ]; then
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[$MONTH] already running (PID $pid), skipping"
            continue
        fi
    fi

    echo "[$MONTH] launching..."
    nohup bash "$SCRIPTS/run_month_eval.sh" \
        "$MONTH" "$OUTFILE" "$LOGFILE" "$PIDFILE" \
        > /dev/null 2>&1 &

    sleep 15
done

echo "All eval watchers launched."
