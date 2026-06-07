#!/usr/bin/env bash
# Usage: run_month.sh <YYYY-MM> <output_file> <log_file> <pid_file>
# Loops until python exits 0 (EOF = done). Retries curl failures with
# exponential backoff. Does NOT delete the data file on failure so that
# the checkpoint can resume where it left off.
set -euo pipefail

MONTH="$1"
OUTFILE="$2"
LOGFILE="$3"
PIDFILE="$4"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO/scripts"
URL="https://database.lichess.org/standard/lichess_db_standard_rated_${MONTH}.pgn.zst"

printf 'WhiteElo >= "2500"\nBlackElo >= "2500"\n' > /tmp/elo_filter.pgn

echo "$$" > "$PIDFILE"
echo "[watcher/$MONTH] started PID=$$" >> "$LOGFILE"

delay=30

while true; do
    # Use --no-resume only when no checkpoint exists yet.
    extra=""
    if [ ! -f "${OUTFILE}.ckpt" ]; then
        extra="--no-resume"
    fi

    # Disable errexit around the pipeline so curl/zstd failures don't kill the
    # watcher before PIPESTATUS can be read and the retry loop can fire.
    set +e
    curl -fsSL "$URL" \
      | zstd -dc \
      | pgn-extract -s --quiet -t /tmp/elo_filter.pgn 2>/dev/null \
      | python3 "$SCRIPTS/extract_lichess_wdl.py" \
          --target 5_000_000 --output "$OUTFILE" $extra \
      >> "$LOGFILE" 2>&1
    pipe_status=("${PIPESTATUS[@]}")
    set -e

    curl_rc=${pipe_status[0]}
    py_rc=${pipe_status[3]}

    # Treat curl failure OR zero positions emitted as a non-success so we retry.
    # (curl returning 0 bytes → python gets empty stdin → py_rc=0 with 0 positions)
    last_emitted=$(grep -oP 'emitted=\K[0-9]+' "$LOGFILE" | tail -1 || echo 0)
    if [ "$py_rc" -eq 0 ] && [ "$curl_rc" -eq 0 ] && [ "${last_emitted:-0}" -gt 0 ]; then
        echo "[watcher/$MONTH] COMPLETE at $(date)" >> "$LOGFILE"
        rm -f "$PIDFILE"
        exit 0
    fi

    echo "[watcher/$MONTH] exited py_rc=$py_rc, sleeping ${delay}s then retry" >> "$LOGFILE"
    sleep "$delay"
    delay=$(( delay < 300 ? delay * 2 : 300 ))
done
