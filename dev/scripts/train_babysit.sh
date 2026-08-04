#!/usr/bin/env bash
# Hourly babysitter (launchd, every 3600s) for the Kaggle training run + bot.
# Pre-launch: keep the upload+launch loop alive until it uploads the chunks and
# pushes the kernel (creates the `launched` flag). Post-launch: keep the bot
# alive; when the kernel stops, tell "All done!" (complete) from a 12h session
# cut-off, and for the latter carry the checkpoint from kernel output into
# eclipse-checkpoint (the in-notebook secret sync is unavailable this account)
# so the next session's mount-fallback resumes, then re-push.
set -uo pipefail
export KAGGLE_API_TOKEN=KGAT_eb6669ac1f5d79234bb07341c23a7285
export PATH="/usr/local/bin:$PATH"
REPO=/Users/ezra/eclipse-chess-engine
STATE=/Users/ezra/.eclipse_train_babysit
LOG="$STATE/babysit.log"
KERNEL=simbae11/tcec-chess-engine
BOT_PLIST=/Users/ezra/Library/LaunchAgents/com.eclipsebot.lichess.plist
MAX_PUSHES=6
mkdir -p "$STATE"
ts(){ date '+%F %T'; }
log(){ echo "[$(ts)] $*" >> "$LOG"; }
# portable timeout: run "$@" but kill it after $1 seconds
run_to(){ local t=$1; shift; "$@" & local p=$!; ( sleep "$t"; kill -9 $p 2>/dev/null ) & local k=$!; wait $p 2>/dev/null; local rc=$?; kill -9 $k 2>/dev/null; return $rc; }

log "---- tick ----"

# Pre-launch: keep the upload+launch retry loop alive.
if [ ! -f "$STATE/launched" ]; then
  if pgrep -f upload_and_launch.sh >/dev/null 2>&1; then
    log "pre-launch: upload_and_launch running; waiting."
  else
    log "pre-launch: upload_and_launch not running -> (re)starting it"
    nohup bash "$REPO/dev/scripts/upload_and_launch.sh" >/dev/null 2>&1 </dev/null &
  fi
  exit 0
fi

# 1) Keep the lichess bot alive.
if pgrep -f "lichess-bot.py.*config-eclipse" >/dev/null 2>&1; then
  log "bot: running"
else
  log "bot: not running -> bootstrapping agent"
  launchctl bootstrap "gui/$(id -u)" "$BOT_PLIST" 2>>"$LOG" \
    || launchctl kickstart -k "gui/$(id -u)/com.eclipsebot.lichess" 2>>"$LOG" || true
fi

# 2) Training run.
[ -f "$STATE/done" ] && { log "training DONE; nothing to do."; exit 0; }
status=$(cd "$REPO" && kaggle kernels status "$KERNEL" 2>&1 | tr -d '\n')
log "kernel status: $status"
echo "$status" | grep -qiE "running|queued" && { log "run active; nothing to do."; exit 0; }

# Not running: session complete or errored. Fetch output (bounded) to tell
# "training finished" from "12h session cut it off".
OUT="$STATE/kout"; rm -rf "$OUT"; mkdir -p "$OUT"
log "fetching kernel output (<=15min)..."
run_to 900 bash -c "cd '$REPO' && kaggle kernels output '$KERNEL' -p '$OUT'" >>"$LOG" 2>&1
KLOG=$(ls "$OUT"/*.log 2>/dev/null | head -1)
if [ -n "$KLOG" ] && grep -q "All done!" "$KLOG"; then
  log "TRAINING COMPLETE (All done!). Trained net at $OUT/halfkav2.pt — ready to convert+SPRT+deploy."
  touch "$STATE/done"
  exit 0
fi

pushes=$(cat "$STATE/pushes" 2>/dev/null || echo 0)
[ "$pushes" -ge "$MAX_PUSHES" ] && { log "push cap ($MAX_PUSHES) reached; manual review needed."; exit 0; }

if [ -f "$OUT/halfkav2.pt" ]; then
  log "mid-training session end -> carrying checkpoint into eclipse-checkpoint"
  CK="$STATE/ck_carry"; rm -rf "$CK"; mkdir -p "$CK"
  cp "$OUT/halfkav2.pt" "$CK/"
  [ -f "$OUT/resume_state.pt" ] && cp "$OUT/resume_state.pt" "$CK/"
  printf '{"title":"eclipse-checkpoint","id":"simbae11/eclipse-checkpoint","licenses":[{"name":"other"}]}\n' > "$CK/dataset-metadata.json"
  ( cd "$CK" && kaggle datasets version -p "$CK" -m "carry-over after session $pushes" -r skip ) >>"$LOG" 2>&1
else
  log "no halfkav2.pt in output; re-pushing (warm-start from original mounted net)"
fi
log "re-pushing to resume (push #$((pushes+1)))"
( cd "$REPO" && kaggle kernels push -p dev/notebooks/ ) >>"$LOG" 2>&1
echo $((pushes+1)) > "$STATE/pushes"
log "re-push issued."
