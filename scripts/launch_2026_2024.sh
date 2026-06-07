#!/usr/bin/env bash
# Launch WDL extractions for Jan-May 2026 and all 12 months of 2024.
# Uses run_month.sh (watcher with retry) for robustness.
# Each job runs to EOF since per-month yield is ~300-600K (well under target).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$REPO/data"
SCRIPTS="$REPO/scripts"

launch() {
    local month="$1" out="$2" log="$3" pid_file="$4"
    nohup bash "$SCRIPTS/run_month.sh" "$month" "$out" "$log" "$pid_file" &
    echo "launched $month  out=$(basename "$out")"
}

# ── 2026: Jan–May (Jun not published yet) ──────────────────────────────────
launch 2026-01 "$DATA/wdl_2026_01.txt" "$REPO/extract_2026_01.log" "$REPO/extract_2026_01.pid"
launch 2026-02 "$DATA/wdl_2026_02.txt" "$REPO/extract_2026_02.log" "$REPO/extract_2026_02.pid"
launch 2026-03 "$DATA/wdl_2026_03.txt" "$REPO/extract_2026_03.log" "$REPO/extract_2026_03.pid"
launch 2026-04 "$DATA/wdl_2026_04.txt" "$REPO/extract_2026_04.log" "$REPO/extract_2026_04.pid"
launch 2026-05 "$DATA/wdl_2026_05.txt" "$REPO/extract_2026_05.log" "$REPO/extract_2026_05.pid"

# ── 2024: all 12 months ─────────────────────────────────────────────────────
launch 2024-01 "$DATA/wdl_2024_01.txt" "$REPO/extract_2024_01.log" "$REPO/extract_2024_01.pid"
launch 2024-02 "$DATA/wdl_2024_02.txt" "$REPO/extract_2024_02.log" "$REPO/extract_2024_02.pid"
launch 2024-03 "$DATA/wdl_2024_03.txt" "$REPO/extract_2024_03.log" "$REPO/extract_2024_03.pid"
launch 2024-04 "$DATA/wdl_2024_04.txt" "$REPO/extract_2024_04.log" "$REPO/extract_2024_04.pid"
launch 2024-05 "$DATA/wdl_2024_05.txt" "$REPO/extract_2024_05.log" "$REPO/extract_2024_05.pid"
launch 2024-06 "$DATA/wdl_2024_06.txt" "$REPO/extract_2024_06.log" "$REPO/extract_2024_06.pid"
launch 2024-07 "$DATA/wdl_2024_07.txt" "$REPO/extract_2024_07.log" "$REPO/extract_2024_07.pid"
launch 2024-08 "$DATA/wdl_2024_08.txt" "$REPO/extract_2024_08.log" "$REPO/extract_2024_08.pid"
launch 2024-09 "$DATA/wdl_2024_09.txt" "$REPO/extract_2024_09.log" "$REPO/extract_2024_09.pid"
launch 2024-10 "$DATA/wdl_2024_10.txt" "$REPO/extract_2024_10.log" "$REPO/extract_2024_10.pid"
launch 2024-11 "$DATA/wdl_2024_11.txt" "$REPO/extract_2024_11.log" "$REPO/extract_2024_11.pid"
launch 2024-12 "$DATA/wdl_2024_12.txt" "$REPO/extract_2024_12.log" "$REPO/extract_2024_12.pid"

echo ""
echo "17 jobs launched. Monitor:"
echo "  for p in $REPO/extract_202{4,6}_??.pid; do"
echo "    pid=\$(cat \$p 2>/dev/null); m=\$(basename \$p .pid);"
echo "    kill -0 \$pid 2>/dev/null && echo \"\$m: running\" || echo \"\$m: done\"; done"
