#!/usr/bin/env bash
# Usage: run_month_eval.sh <YYYY-MM> <output_file> <log_file> <pid_file>
# Loops until python exits 0 (EOF = done). Retries curl failures with
# exponential backoff. Checkpoint in <output_file>.ckpt enables mid-stream
# resume — curl fast-forwards through already-processed bytes on restart.
set -euo pipefail

MONTH="$1"
OUTFILE="$2"
LOGFILE="$3"
PIDFILE="$4"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO/scripts"
URL="https://database.lichess.org/standard/lichess_db_standard_rated_${MONTH}.pgn.zst"

# Ignore SIGTERM/SIGPIPE from killed pipeline children.
trap '' SIGTERM SIGPIPE

echo "$$" > "$PIDFILE"
echo "[eval-watcher/$MONTH] started PID=$$" >> "$LOGFILE"

delay=30

while true; do
    extra=""
    if [ ! -f "${OUTFILE}.ckpt" ]; then
        extra="--no-resume"
    fi

    set +e
    curl -fsSL --speed-limit 512 --speed-time 30 "$URL" \
      | zstd -dc \
      | python3 "$SCRIPTS/extract_lichess_evals.py" \
          --output "$OUTFILE" \
          --min-elo 1800 \
          --min-tc-seconds 180 \
          $extra \
      >> "$LOGFILE" 2>&1
    pipe_status=("${PIPESTATUS[@]}")
    set -e

    curl_rc=${pipe_status[0]}
    py_rc=${pipe_status[2]}

    last_emitted=$(grep -oP 'emitted=\K[0-9]+' "$LOGFILE" | tail -1 || echo 0)
    # curl_rc=0: natural EOF (expected; python runs to data exhaustion).
    # curl_rc>=128: killed by signal — tolerated for forward-compat.
    if [ "$py_rc" -eq 0 ] && [ "${last_emitted:-0}" -gt 0 ] && \
       { [ "$curl_rc" -eq 0 ] || [ "$curl_rc" -ge 128 ]; }; then
        echo "[eval-watcher/$MONTH] COMPLETE at $(date)" >> "$LOGFILE"
        rm -f "$PIDFILE"
        exit 0
    fi

    echo "[eval-watcher/$MONTH] exited py_rc=$py_rc curl_rc=$curl_rc, sleeping ${delay}s then retry" >> "$LOGFILE"
    sleep "$delay"
    delay=$(( delay < 300 ? delay * 2 : 300 ))
done
