#!/usr/bin/env bash
# Profile-guided optimization build for Eclipse.
#
# Three stages:
#   1. Build an instrumented binary (ECLIPSE_PGO=generate, LTO off for speed).
#   2. Run a representative workload (opening / middlegame / endgame / tactical,
#      single- and multi-threaded) so the .profraw counters cover the real hot
#      paths the engine takes in a game.
#   3. Merge the profiles and produce the final optimized binary into build/
#      (ECLIPSE_PGO=use + LTO on). PGO only affects speed, not move choice.
#
# After this finishes, build/src/eclipse is the PGO+LTO binary and `cmake
# --build build` keeps rebuilding it in PGO-use mode (the merged profile is
# cached at build/eclipse.profdata; if it is ever deleted the build silently
# falls back to a plain optimized binary).
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
NET="${ECLIPSE_NET:-data/eclipse.nnue}"
GEN_BUILD="build-pgo-gen"
RAW_DIR="$ROOT/$GEN_BUILD/pgo-raw"
PROFDATA="$ROOT/build/eclipse.profdata"
PROFDATA_TOOL=(xcrun llvm-profdata)
command -v llvm-profdata >/dev/null 2>&1 && PROFDATA_TOOL=(llvm-profdata)

if [[ ! -f "$NET" ]]; then
    echo "error: net '$NET' not found (set ECLIPSE_NET=path)." >&2
    exit 1
fi

echo "==> Stage 1/3: instrumented build ($GEN_BUILD)"
cmake -S . -B "$GEN_BUILD" -DECLIPSE_PGO=generate -DECLIPSE_LTO=OFF \
      -DECLIPSE_BUILD_TESTS=OFF >/dev/null
cmake --build "$GEN_BUILD" -j --target eclipse

echo "==> Stage 2/3: running representative training workload"
rm -rf "$RAW_DIR"; mkdir -p "$RAW_DIR"

# Positions: startpos (opening), an open middlegame, Kiwipete (tactical, castling
# rights -> exercises the castling accumulator path), and a KPK pawn endgame
# (low material -> exercises refresh_perspective / endgame eval).
FENS=(
  "startpos"
  "fen r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1"
  "fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
  "fen 8/2k5/8/4P3/3K4/8/8/8 w - - 0 1"
)

run_workload() {  # $1 = threads
    # Drive the engine so each `go` runs to completion: block reading its output
    # until `bestmove` before sending the next position. Dumping all commands at
    # once would let `quit` cut every search short, leaving the profile blind to
    # the deep-search hot paths. macOS ships bash 3.2 (no `coproc`), so use a
    # pair of FIFOs and blocking reads (no polling/sleep).
    local threads="$1" line
    local in out
    in="$(mktemp -u)"; out="$(mktemp -u)"
    mkfifo "$in" "$out"
    export LLVM_PROFILE_FILE="$RAW_DIR/eclipse-t${threads}-%p.profraw"
    "./$GEN_BUILD/src/eclipse" <"$in" >"$out" 2>/dev/null &
    local epid=$!
    # Open the input write-end first (unblocks the engine's read-open of $in),
    # then the output read-end (unblocks its write-open of $out) -> no deadlock.
    exec 9>"$in"
    exec 8<"$out"
    printf 'setoption name EvalFile value %s\nsetoption name Threads value %s\nisready\n' \
           "$NET" "$threads" >&9
    while IFS= read -r line <&8; do [[ "$line" == readyok ]] && break; done
    for f in "${FENS[@]}"; do
        printf 'position %s\ngo movetime 5000\n' "$f" >&9
        while IFS= read -r line <&8; do [[ "$line" == bestmove* ]] && break; done
    done
    printf 'quit\n' >&9
    exec 9>&- 8<&-
    wait "$epid" 2>/dev/null || true
    rm -f "$in" "$out"
}

run_workload 1
run_workload 4

shopt -s nullglob
raw=( "$RAW_DIR"/*.profraw )
if (( ${#raw[@]} == 0 )); then
    echo "error: no .profraw files emitted -- PGO instrumentation failed." >&2
    exit 1
fi
echo "    collected ${#raw[@]} raw profile(s)"

echo "==> Stage 3/3: merge profiles + final optimized build (build/)"
mkdir -p "$(dirname "$PROFDATA")"
"${PROFDATA_TOOL[@]}" merge -output="$PROFDATA" "${raw[@]}"
echo "    merged -> $PROFDATA"

cmake -S . -B build -DECLIPSE_PGO=use -DECLIPSE_PGO_DATA="$PROFDATA" >/dev/null
# --clean-first forces a FULL recompile against the freshly merged profdata.
# Without it, an incremental rebuild (when build/ was already PGO=use, so the
# reconfigure above is a no-op) only recompiles changed sources -- the rest keep
# the ProfileSummary embedded from the previous profdata, and ThinLTO aborts the
# link with "module flags 'ProfileSummary': IDs have conflicting values". A
# changed profile summary isn't a tracked dependency, so the recompile must be
# forced. (The merged profdata lives at build/eclipse.profdata, which the clean
# target does not touch -- it only removes CMake-generated build outputs.)
cmake --build build -j --clean-first

echo "==> PGO build complete: build/src/eclipse"
