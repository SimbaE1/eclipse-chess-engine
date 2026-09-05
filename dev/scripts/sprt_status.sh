#!/usr/bin/env bash
# Progress report for the running SPRT isolation queue.
#   dev/scripts/sprt_status.sh          one-shot
#   dev/scripts/sprt_status.sh -w       refresh every 60s
set -uo pipefail
REPO="/Users/ezra/eclipse-chess-engine"
# Newest iso_queue*.pid, not a hardcoded one: each queue generation
# (iso_queue.pid, iso_queue3.pid, iso_queue5.pid, ...) writes its own,
# and pinning the first meant every later queue reported NOT RUNNING.
QUEUE_PID_FILE="$(ls -t "$REPO"/dev/sprt_runs/iso_queue*.pid 2>/dev/null | head -1)"

report(){
  local pid alive
  pid=$(cat "${QUEUE_PID_FILE:-/nonexistent}" 2>/dev/null || echo "")
  if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then alive="RUNNING (pid $pid)"; else alive="NOT RUNNING"; fi
  echo "=============================================================="
  echo "queue: $alive    $(date '+%Y-%m-%d %H:%M:%S')"
  echo "engines: $(pgrep -f 'build/src/eclipse' | wc -l | tr -d ' ') procs, load$(uptime | sed 's/.*averages*:/ /')"
  echo
  # Every leg directory, oldest first. The one with the newest log is current.
  for d in $(ls -dt "$REPO"/dev/sprt_runs/2026*_{ab,aspiration,tt,mcts,movehorizon,see}* 2>/dev/null | tail -r); do
    local name last score sprt started
    name=$(basename "$d")
    [[ -f "$d/log.txt" ]] || continue
    score=$(grep -E '^Score of ' "$d/log.txt" | tail -1)
    sprt=$(grep -E '^SPRT: ' "$d/log.txt" | tail -1)
    elo=$(grep -E '^Elo difference' "$d/log.txt" | tail -1)
    started=$(date -r "$d" '+%m-%d %H:%M')
    printf '%-34s  started %s\n' "$name" "$started"
    [[ -n "$score" ]] && echo "   $score"
    [[ -n "$elo"   ]] && echo "   $elo"
    [[ -n "$sprt"  ]] && echo "   $sprt"
    grep -q 'Finished match' "$d/log.txt" && echo "   >> FINISHED"
    echo
  done
}

if [[ "${1:-}" == "-w" ]]; then while :; do clear; report; sleep 60; done; else report; fi
