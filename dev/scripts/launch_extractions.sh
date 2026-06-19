#!/usr/bin/env bash
# Launch (or resume) monthly WDL extractions toward 15M total positions.
#
# Feb-May 2025: resume from existing checkpoints (auto-detected by the script).
# Jun-Aug 2025: fresh starts.
#
# Each job: curl | zstd -dc | pgn-extract (Elo pre-filter) | extract_lichess_wdl.py
# nohup + & keeps jobs alive after terminal disconnect.
# PIDs written to extract_YYYY_MM.pid for monitoring.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$REPO/data"
SCRIPTS="$REPO/scripts"
BASE_URL="https://database.lichess.org/standard"

printf 'WhiteElo >= "2500"\nBlackElo >= "2500"\n' > /tmp/elo_filter.pgn

# args: YYYY-MM  output-file  log-file  pid-file
launch() {
    local month="$1" out="$2" log="$3" pid_file="$4"
    local url="$BASE_URL/lichess_db_standard_rated_${month}.pgn.zst"

    # If a checkpoint exists the script resumes automatically.
    # If no checkpoint exists but the output file does, the script would
    # refuse (safety guard). --no-resume is only needed for a deliberate restart.
    nohup bash -c "
        curl -fsSL \"$url\" \
          | zstd -dc \
          | pgn-extract -s --quiet -t /tmp/elo_filter.pgn 2>/dev/null \
          | python3 \"$SCRIPTS/extract_lichess_wdl.py\" \
              --target 5_000_000 \
              --output \"$out\"
    " >> "$log" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"
    echo "launched $month  PID=$pid  out=$(basename "$out")"
}

# ── Resume Feb-May (checkpoints already exist) ─────────────────────────────
launch 2025-02 "$DATA/wdl_feb2025.txt"    "$REPO/extract_feb2025.log"    "$REPO/extract_feb2025.pid"
launch 2025-03 "$DATA/wdl_2025_03.txt"   "$REPO/extract_2025_03.log"   "$REPO/extract_2025_03.pid"
launch 2025-04 "$DATA/wdl_2025_04.txt"   "$REPO/extract_2025_04.log"   "$REPO/extract_2025_04.pid"
launch 2025-05 "$DATA/wdl_2025_05.txt"   "$REPO/extract_2025_05.log"   "$REPO/extract_2025_05.pid"

# ── Fresh months to reach 15M ───────────────────────────────────────────────
launch 2025-06 "$DATA/wdl_2025_06.txt"   "$REPO/extract_2025_06.log"   "$REPO/extract_2025_06.pid"
launch 2025-07 "$DATA/wdl_2025_07.txt"   "$REPO/extract_2025_07.log"   "$REPO/extract_2025_07.pid"
launch 2025-08 "$DATA/wdl_2025_08.txt"   "$REPO/extract_2025_08.log"   "$REPO/extract_2025_08.pid"

echo ""
echo "All jobs launched. Monitor with:"
echo "  tail -f \$REPO/extract_2025_03.log   # example"
echo "  for p in \$REPO/extract_2025_??.pid \$REPO/extract_feb2025.pid; do pid=\$(cat \$p); kill -0 \$pid 2>/dev/null && echo \"\$(basename \$p .pid): running\" || echo \"\$(basename \$p .pid): stopped\"; done"
