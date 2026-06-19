#!/usr/bin/env bash
# Deploy a trained HalfKAv2 checkpoint into the Eclipse engine.
#
# Usage:
#   ./scripts/deploy.sh                          # auto-find latest .pt in data/
#   ./scripts/deploy.sh data/halfkav2_epoch2.pt  # explicit checkpoint
#
# Steps:
#   1. Locate the .pt checkpoint
#   2. Convert it to data/eclipse.nnue  (quantized binary format the C++ reads)
#   3. Rebuild the engine in release mode
#   4. Print the UCI setoption line to use in your GUI / TCEC config
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$REPO/data"
BUILD="$REPO/build"

# --- Locate checkpoint -------------------------------------------------------
if [ $# -ge 1 ]; then
    PT="$1"
else
    # Auto-detect: pick the most-recently-modified .pt file in data/
    PT=$(ls -t "$DATA"/*.pt 2>/dev/null | head -1 || true)
    if [ -z "$PT" ]; then
        echo "error: no .pt file found in $DATA — pass the path explicitly"
        exit 1
    fi
    echo "Auto-detected checkpoint: $PT"
fi

if [ ! -f "$PT" ]; then
    echo "error: checkpoint not found: $PT"; exit 1
fi

NNUE="$DATA/eclipse.nnue"

# --- Convert -----------------------------------------------------------------
echo "Converting $PT -> $NNUE ..."
python3 "$REPO/scripts/convert_halfkav2_nnue.py" from-torch \
    --state-dict "$PT" \
    --out "$NNUE"

# --- Build -------------------------------------------------------------------
if [ -d "$BUILD" ]; then
    echo "Rebuilding engine ..."
    cmake --build "$BUILD" --config Release -j"$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)"
else
    echo "Build directory not found at $BUILD — run cmake first:"
    echo "  cmake -S $REPO -B $BUILD -DCMAKE_BUILD_TYPE=Release"
    exit 1
fi

BINARY="$BUILD/eclipse"
if [ ! -f "$BINARY" ]; then
    echo "error: binary not found at $BINARY after build"; exit 1
fi

echo ""
echo "=== Done ==="
echo "Binary : $BINARY"
echo "NNUE   : $NNUE"
echo ""
echo "UCI config line:"
echo "  setoption name EvalFile value $NNUE"
echo ""
echo "To play in terminal:"
echo "  $BINARY"
echo "  uci"
echo "  setoption name EvalFile value $NNUE"
echo "  isready"
echo "  position startpos"
echo "  go movetime 5000"
