#!/usr/bin/env bash
# Persistent retry of the split-chunk upload (Kaggle GCS finalize is flaky over
# this link for files above a few hundred MB). Retries the whole `datasets
# version` (all-or-nothing), killing an attempt that stalls on finalize, until
# one run gets all 192 sub-files through. Then pushes the training kernel and
# sets the `launched` flag so the hourly babysit takes over.
export KAGGLE_API_TOKEN=KGAT_eb6669ac1f5d79234bb07341c23a7285
export PATH="/usr/local/bin:$PATH"
REPO=/Users/ezra/eclipse-chess-engine
STATE=/Users/ezra/.eclipse_train_babysit
LOG=$STATE/upload_launch.log
KERNEL=simbae11/tcec-chess-engine
mkdir -p "$STATE"
ts(){ date '+%F %T'; }
log(){ echo "[$(ts)] $*" >> "$LOG"; }

nfiles(){ kaggle datasets files simbae11/eclipse-chunks-a 2>/dev/null | grep -c eval_chunk; }

log "=== upload_and_launch started ==="
attempt=0
while :; do
  n=$(nfiles)
  if [ "${n:-0}" -ge 60 ]; then log "dataset already has $n chunk files -> upload done"; break; fi
  attempt=$((attempt+1))
  [ "$attempt" -gt 300 ] && { log "gave up after $attempt attempts"; exit 1; }
  ALOG=$STATE/ul_attempt.log; : > "$ALOG"
  log "attempt $attempt: kaggle datasets version -r skip (192 x 484MB)"
  ( cd "$REPO" && kaggle datasets version -p data/chunks_small \
      -m "dedup+in-check+IID 731.9M positions, 484MB sub-chunks" -r skip ) >> "$ALOG" 2>&1 &
  UP=$!
  last=0; stall=0
  while kill -0 $UP 2>/dev/null; do
    sleep 60
    sz=$(stat -f%z "$ALOG" 2>/dev/null || echo 0)
    if [ "$sz" -eq "$last" ]; then
      stall=$((stall+60))
      if [ "$stall" -ge 480 ]; then log "attempt $attempt stalled 8min (finalize hang) -> kill"; kill -9 $UP 2>/dev/null; pkill -9 -f 'kaggle datasets version' 2>/dev/null; break; fi
    else stall=0; last=$sz; fi
  done
  wait $UP 2>/dev/null
  n=$(nfiles)
  if [ "${n:-0}" -ge 60 ]; then log "attempt $attempt SUCCEEDED ($n files in dataset)"; break; fi
  log "attempt $attempt incomplete (dataset still $n files); retry in 120s"
  sleep 120
done

log "chunks uploaded -> pushing training kernel"
( cd "$REPO" && kaggle kernels push -p dev/notebooks/ ) >> "$LOG" 2>&1
echo 1 > "$STATE/pushes"
touch "$STATE/launched"
log "kernel pushed; launched flag set. Babysit now owns resume + bot + progress."
