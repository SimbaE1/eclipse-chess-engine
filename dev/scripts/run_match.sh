#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Reusable cutechess-cli launcher for Eclipse self-play / head-to-head matches.
#
# Plays net1 vs net2 with color-reversed opening pairs (-repeat). Pass --book to
# start from an imbalanced opening suite (TCEC/Stockfish-testing style) so two
# near-equal nets produce decisive games instead of a draw-fest; fairness is
# preserved because every opening is played once with each color.
#
# Get the standard imbalanced book once with:
#   curl -sL -o /tmp/uho.zip \
#     https://github.com/official-stockfish/books/raw/master/UHO_4060_v2.epd.zip
#   unzip -o /tmp/uho.zip -d data/books/
#
# Examples:
#   scripts/run_match.sh --net1 /tmp/eclipse_v7_ep2c5.nnue \
#                        --net2 /tmp/eclipse_v7_ep2c3.nnue \
#                        --name1 ep2c5 --name2 ep2c3 \
#                        --tc 5+3 --games 20 --pgn /tmp/c5_vs_c3.pgn \
#                        --book data/books/UHO_4060_v2.epd
#
#   # fixed-nodes (deterministic compute per move, immune to system load --
#   # best for comparing nets; forces ponder off):
#   scripts/run_match.sh --net1 a.nnue --net2 b.nnue --nodes 3000 --games 200 \
#                        --book data/books/UHO_4060_v2.epd
#
#   # show the command without running it:
#   scripts/run_match.sh --net1 a.nnue --net2 b.nnue --dry-run
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE="$REPO/build/src/eclipse"
CUTECHESS="${CUTECHESS:-/Users/ezra/cutechess/cutechess/build/cutechess-cli}"

# Defaults (mirror the standard Eclipse self-play setup).
NET1=""; NET2=""; NAME1="net1"; NAME2="net2"
TC="5+3"; NODES=""; GAMES=20; THREADS=4; ABTHREADS=1; HASH=256
SYZYGY="${SYZYGY:-/Users/ezra/syzygy}"
CONCURRENCY=1; PONDER=1; BOOK=""; PGN=""; DEBUG=0; DRY_RUN=0; TB_ADJUDICATE=0; TBPIECES=5

while [ $# -gt 0 ]; do
  case "$1" in
    --net1)        NET1=$2; shift 2;;
    --net2)        NET2=$2; shift 2;;
    --name1)       NAME1=$2; shift 2;;
    --name2)       NAME2=$2; shift 2;;
    --tc)          TC=$2; shift 2;;
    --nodes)       NODES=$2; shift 2;;
    --games)       GAMES=$2; shift 2;;
    --threads)     THREADS=$2; shift 2;;
    --ab-threads)  ABTHREADS=$2; shift 2;;
    --hash)        HASH=$2; shift 2;;
    --syzygy)      SYZYGY=$2; shift 2;;
    --concurrency) CONCURRENCY=$2; shift 2;;
    --book)        BOOK=$2; shift 2;;
    --pgn)         PGN=$2; shift 2;;
    --no-ponder)   PONDER=0; shift;;
    --tb)          TB_ADJUDICATE=1; shift;;
    --tbpieces)    TBPIECES=$2; shift 2;;
    --debug)       DEBUG=1; shift;;
    --dry-run)     DRY_RUN=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[ -n "$NET1" ] && [ -n "$NET2" ] || { echo "need --net1 and --net2" >&2; exit 2; }
[ -x "$ENGINE" ] || { echo "engine not built: $ENGINE" >&2; exit 1; }
[ -n "$PGN" ] || PGN="/tmp/${NAME1}_vs_${NAME2}.pgn"

# Fixed-nodes mode: deterministic compute per move (immune to system load),
# the right tool for comparing nets. Pondering makes no sense under a node cap
# and would break determinism, so force it off.
if [ -n "$NODES" ]; then
  PONDER=0
  echo "mode         : fixed nodes=$NODES (tc=inf, ponder off)"
fi

PONDER_OPT=""; [ "$PONDER" = 1 ] && PONDER_OPT="ponder"

# Per-engine option block, reused for both with a different EvalFile.
engine_block() {  # $1=name $2=net
  printf -- '-engine name=%s cmd=%s option.EvalFile=%s option.Threads=%s option.AbThreads=%s option.Hash=%s option.SyzygyPath=%s %s' \
    "$1" "$ENGINE" "$2" "$THREADS" "$ABTHREADS" "$HASH" "$SYZYGY" "$PONDER_OPT"
}

# Assemble the full argv.
ARGS=()
read -r -a E1 <<< "$(engine_block "$NAME1" "$NET1")"; ARGS+=("${E1[@]}")
read -r -a E2 <<< "$(engine_block "$NAME2" "$NET2")"; ARGS+=("${E2[@]}")
if [ -n "$NODES" ]; then
  ARGS+=(-each "tc=inf" "nodes=$NODES" proto=uci)
else
  ARGS+=(-each "tc=$TC" proto=uci)
fi
ARGS+=(-games "$GAMES" -repeat -concurrency "$CONCURRENCY" -pgnout "$PGN")
if [ -n "$BOOK" ]; then
  [ -f "$BOOK" ] || { echo "book not found: $BOOK" >&2; exit 1; }
  ARGS+=(-openings "file=$BOOK" format=epd order=random)
  echo "opening book : $BOOK (imbalanced; color-reversed via -repeat)"
else
  echo "opening book : none (startpos) -- expect many draws between near-equal nets"
fi
if [ "$TB_ADJUDICATE" = 1 ]; then
  [ -d "$SYZYGY" ] || { echo "syzygy path not found for -tb adjudication: $SYZYGY" >&2; exit 1; }
  ARGS+=(-tb "$SYZYGY" -tbpieces "$TBPIECES")
  echo "tb adjudicate: $SYZYGY (<= $TBPIECES pieces)"
fi
[ "$DEBUG" = 1 ] && ARGS+=(-debug)

LIMIT_DESC=$([ -n "$NODES" ] && echo "nodes=$NODES" || echo "tc=$TC")
echo "match        : $NAME1 vs $NAME2   $LIMIT_DESC games=$GAMES threads=$THREADS ponder=$PONDER"
echo "pgn          : $PGN"

if [ "$DRY_RUN" = 1 ]; then
  printf 'DRY RUN, would exec:\n  %s' "$CUTECHESS"
  printf ' %q' "${ARGS[@]}"; printf '\n'
  exit 0
fi

[ -x "$CUTECHESS" ] || { echo "cutechess-cli not found: $CUTECHESS" >&2; exit 1; }
if [ "$DEBUG" = 1 ]; then
  exec "$CUTECHESS" "${ARGS[@]}" 2>&1 | tee "${PGN%.pgn}_debug.log"
else
  exec "$CUTECHESS" "${ARGS[@]}"
fi
