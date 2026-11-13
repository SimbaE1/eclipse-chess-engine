#!/usr/bin/env bash
# Hourly babysitter (launchd, every 3600s) for the Kaggle training run + bot.
# Pre-launch: keep the upload+launch loop alive until it uploads the chunks and
# pushes the kernel (creates the `launched` flag). Post-launch: keep the bot
# alive; when the kernel stops, tell "All done!" (complete) from a 12h session
# cut-off, and for the latter carry the checkpoint from kernel output into
# eclipse-checkpoint (the in-notebook secret sync is unavailable this account)
# so the next session's mount-fallback resumes, then re-push.
set -uo pipefail
# Token comes from the environment or a private file OUTSIDE the repo; never commit it.
KAGGLE_API_TOKEN=${KAGGLE_API_TOKEN:-$(cat "$HOME/.kaggle/eclipse_token" 2>/dev/null)}
[ -n "$KAGGLE_API_TOKEN" ] || { echo "train_babysit: no Kaggle token (set KAGGLE_API_TOKEN or ~/.kaggle/eclipse_token)" >&2; exit 1; }
export KAGGLE_API_TOKEN
export PATH="/usr/local/bin:$PATH"
REPO=/Users/ezra/eclipse-chess-engine
STATE=${BABYSIT_STATE:-/Users/ezra/.eclipse_train_babysit}
LOG="$STATE/babysit.log"
KERNEL=${BABYSIT_KERNEL:-simbae11/tcec-chess-engine}
# Checkpoint lineage for THIS run; must match CKPT_DATASET in the notebook.
CKPT_ID=${BABYSIT_CKPT_ID:-simbae11/eclipse-checkpoint-quiet-v2}
NB_DIR=${BABYSIT_NB_DIR:-$REPO/dev/notebooks}   # dir holding kernel-metadata.json + the notebook
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

# 1) Keep the lichess bot alive (skipped for test kernels).
if [ -n "${BABYSIT_NO_BOT:-}" ]; then :
elif pgrep -f "lichess-bot.py.*config-eclipse" >/dev/null 2>&1; then
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
# A cancel is a human decision -- never resume it automatically.
if echo "$status" | grep -qi "cancel"; then
  log "kernel was CANCELLED -> not resuming. Remove $STATE/stopped and re-run to override."
  touch "$STATE/stopped"; exit 0
fi
[ -f "$STATE/stopped" ] && { log "stopped flag present; nothing to do."; exit 0; }

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

# Resume ONLY a clean session-budget stop (or, defensively, a hard 12 h kill with
# no Python error). A traceback / TRAINING ABORTED is deterministic: re-pushing
# would just burn sessions, so stop and leave it for a human.
if [ -z "$KLOG" ]; then
  log "no kernel log in output -> manual review."; touch "$STATE/stopped"; exit 0
fi
if grep -qE "TRAINING ABORTED|Traceback|Error:" "$KLOG" && ! grep -q "SESSION STOP" "$KLOG"; then
  log "kernel ERRORED (see $KLOG) -> not resuming; manual review."; touch "$STATE/stopped"; exit 0
fi
grep -q "SESSION STOP" "$KLOG" && log "clean session stop: $(grep -o 'SESSION STOP[^=]*' "$KLOG" | tail -1)" \
  || log "no SESSION STOP marker and no error (hard kill?) -> resuming from last saved chunk"

pushes=$(cat "$STATE/pushes" 2>/dev/null || echo 0)
[ "$pushes" -ge "$MAX_PUSHES" ] && { log "push cap ($MAX_PUSHES) reached; manual review needed."; exit 0; }

if [ -f "$OUT/halfkav2.pt" ]; then
  log "mid-training session end -> carrying checkpoint into $CKPT_ID"
  CK="$STATE/ck_carry"; rm -rf "$CK"; mkdir -p "$CK"
  cp "$OUT/halfkav2.pt" "$CK/"
  [ -f "$OUT/resume_state.pt" ] && cp "$OUT/resume_state.pt" "$CK/"
  printf '{"title":"%s","id":"%s","licenses":[{"name":"other"}]}\n' "${CKPT_ID#*/}" "$CKPT_ID" > "$CK/dataset-metadata.json"
  t_up=$(date -u '+%Y-%m-%d %H:%M:%S')   # Kaggle reports file creationDate in UTC
  if kaggle datasets files "$CKPT_ID" >/dev/null 2>&1; then
    ( cd "$CK" && kaggle datasets version -p "$CK" -m "carry-over after session $pushes" -r skip ) >>"$LOG" 2>&1
  else
    ( cd "$CK" && kaggle datasets create -p "$CK" -r skip ${BABYSIT_CREATE_FLAGS---public} ) >>"$LOG" 2>&1
  fi
  # Mount the checkpoint dataset so the notebook's mount-fallback resumes from it.
  python3 - "$NB_DIR/kernel-metadata.json" "$CKPT_ID" <<'PYEOF' >>"$LOG" 2>&1
import json, sys
p, ck = sys.argv[1], sys.argv[2]
m = json.load(open(p))
if ck not in m["dataset_sources"]:
    m["dataset_sources"].append(ck)
    json.dump(m, open(p, "w"), indent=2)
    print("kernel-metadata: mounted", ck)
PYEOF
  # The push must not happen until the NEW version is processed: mounting a stale
  # version would silently redo a session, a missing one would make the notebook
  # refuse to start. File sizes are identical across versions, so require each
  # file's creationDate (UTC) to be AFTER this upload started.
  want_w=$(stat -f%z "$CK/halfkav2.pt"); want_r=$(stat -f%z "$CK/resume_state.pt" 2>/dev/null || echo 0)
  fresh(){ echo "$files" | awk -v n="$1" -v sz="$2" -v t="$t_up" '$1==n && $2==sz && ($3" "substr($4,1,8)) >= t {f=1} END{exit !f}'; }
  ok=0
  for i in $(seq 1 40); do
    sleep 30
    st=$(kaggle datasets status "$CKPT_ID" 2>/dev/null | tail -1)
    files=$(kaggle datasets files "$CKPT_ID" 2>/dev/null)
    if [ "$st" = "ready" ] && fresh halfkav2.pt "$want_w" && fresh resume_state.pt "$want_r"; then ok=1; break; fi
  done
  if [ $ok -ne 1 ]; then
    log "carried checkpoint not READY/fresh (uploaded >= $t_up UTC) after 20 min -> NOT pushing; retry next tick."
    log "  last status '$st'; files: $(echo "$files" | tr '\n' '|')"
    exit 0
  fi
  log "carried checkpoint is ready in $CKPT_ID (files created >= $t_up UTC)"
else
  log "session ended without halfkav2.pt in output -> NOT re-pushing (would silently restart from ep2c44); manual review."
  touch "$STATE/stopped"; exit 0
fi
log "re-pushing to resume (push #$((pushes+1)))"
( cd "$REPO" && kaggle kernels push -p "$NB_DIR" ) >>"$LOG" 2>&1
echo $((pushes+1)) > "$STATE/pushes"
log "re-push issued."
