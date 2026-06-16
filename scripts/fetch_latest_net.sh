#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Download the latest training checkpoint from Kaggle and pack it into a .nnue.
#
# Pulls halfkav2.pt (+ resume_state.pt) from the eclipse-checkpoint dataset,
# reports which epoch/chunk it is, and converts it to data/eclipse.nnue with
# output_cp_per_unit=300 -- the scale the current notebook trains at. Passing
# 300 explicitly matters: convert_halfkav2_nnue.py still defaults to 410, so a
# bare conversion would miscalibrate every win-prob/MCTS-Q value at load time.
#
# Auth: set KAGGLE_API_TOKEN in the environment (Bearer token, no kaggle.json
# needed). The token is intentionally NOT stored in this repo.
#
#   export KAGGLE_API_TOKEN=KGAT_xxxxxxxx
#   scripts/fetch_latest_net.sh                 # -> data/eclipse.nnue (cp=300)
#   scripts/fetch_latest_net.sh --out data/eclipse_v8.nnue --keep-pt
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="simbae11/eclipse-checkpoint"
OUT="$REPO/data/eclipse.nnue"
CP_PER_UNIT=300
KEEP_PT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)     DATASET=$2; shift 2;;
    --out)         OUT=$2; shift 2;;
    --cp-per-unit) CP_PER_UNIT=$2; shift 2;;
    --keep-pt)     KEEP_PT=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN in the environment (see script header)}"
command -v kaggle >/dev/null || { echo "kaggle CLI not found (pip install kaggle)" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading latest checkpoint from $DATASET ..."
kaggle datasets download -d "$DATASET" -p "$TMP" --unzip --force

PT="$TMP/halfkav2.pt"
[ -f "$PT" ] || { echo "halfkav2.pt not found in $DATASET" >&2; exit 1; }

# Best-effort: report the training position carried in resume_state.pt.
if [ -f "$TMP/resume_state.pt" ]; then
  python3 - "$TMP/resume_state.pt" <<'PY' || true
import sys, torch
rs = torch.load(sys.argv[1], map_location="cpu")
print(f"  checkpoint: epoch {rs.get('epoch','?')} chunk {rs.get('chunk_idx','?')} "
      f"global_step {rs.get('global_step','?'):,}")
PY
fi

mkdir -p "$(dirname "$OUT")"
echo "Packing -> $OUT  (output_cp_per_unit=$CP_PER_UNIT)"
python3 "$REPO/scripts/convert_halfkav2_nnue.py" from-torch \
  --state-dict "$PT" --out "$OUT" --output-cp-per-unit "$CP_PER_UNIT"

# Verify the scale was written correctly (7x uint32 header, then float32).
python3 - "$OUT" "$CP_PER_UNIT" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    f.seek(28)
    got = struct.unpack("<f", f.read(4))[0]
want = float(sys.argv[2])
status = "OK" if abs(got - want) < 1e-6 else "MISMATCH"
print(f"  verified output_cp_per_unit = {got}  [{status}]")
sys.exit(0 if status == "OK" else 1)
PY

if [ "$KEEP_PT" = 1 ]; then
  cp "$PT" "$(dirname "$OUT")/halfkav2.pt"
  echo "  kept raw checkpoint: $(dirname "$OUT")/halfkav2.pt"
fi
echo "Done."
