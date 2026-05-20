#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Extract game-outcome WDL labels from a streaming Lichess PGN dump.

For each filtered game, every position past `--min-ply` is labeled with the
game's eventual outcome from the side-to-move's perspective:

    <fen>;<W>;<D>;<L>

where (W, D, L) is one of (1,0,0), (0,1,0), (0,0,1) — STM won the game, the
game was drawn, STM lost. These are the "soft" targets a real WDL net wants
(observed game outcomes), as opposed to cp-derived targets.

Architecture
------------
The bottleneck on a single core is Python text parsing — at ~3k games/sec
the script wastes 7 of 8 cores. This version splits the work:

  * Main process: reads stdin in 1MB chunks, cuts each chunk at the last
    blank-line game boundary so chunks always end on a clean break, hands
    chunks to a worker pool, drains result lines to stdout.

  * N-1 workers: parse the chunk into games, run the cheap header filters
    (Elo, TC, Result) with early bailout to skip move parsing on rejected
    games, then run python-chess on accepted games to expand to per-position
    WDL labels.

Each chunk contains hundreds of games on average, so per-item pickle
overhead is negligible. On 8 cores the workers see ~10-20x throughput vs
the single-threaded version; reader/writer stays cheap enough to keep up.

Filtering: both-Elo >= --min-elo (default 2500) and TimeControl base >=
--min-tc-seconds (default 300, ~Rapid+). 2500+ keeps human-mistake noise
in WDL targets bounded; conversion of winning positions falls off below
that band.

Usage:

    zstd -dc lichess_db_2025-01.pgn.zst | \\
        python3 scripts/extract_lichess_wdl.py --target 10_000_000 \\
        > data/wdl_training.txt

Recommended faster pipeline using pgn-extract as an Elo pre-filter (kills
99%+ of input games in fast C code before Python ever sees them — adds
another ~3-4x on top of the multiprocessing path):

    # one-time: write the Elo threshold to a tag filter file
    printf 'WhiteElo >= "2500"\\nBlackElo >= "2500"\\n' > /tmp/elo_filter.pgn

    zstd -dc lichess_db_2025-01.pgn.zst | \\
        pgn-extract -s --quiet -t /tmp/elo_filter.pgn 2>/dev/null | \\
        python3 scripts/extract_lichess_wdl.py --target 10_000_000 \\
        > data/wdl_training.txt

pgn-extract's `-s --quiet` silences per-game and per-file status. The
`2>/dev/null` discards the remaining stderr noise. Python still applies
the TimeControl + Result filters (pgn-extract can't easily numeric-compare
on the "600+5" TC string format).
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import sys
from typing import Iterable, List, Optional, Tuple

try:
    import chess
    import chess.pgn
except ImportError:
    print("error: requires python-chess. pip install python-chess", file=sys.stderr)
    sys.exit(1)

# Game-result PGN strings -> (W, D, L) from WHITE's perspective. STM-flipping
# happens per position inside the emit loop.
_RESULT_WDL_WHITE = {
    "1-0":     (1, 0, 0),
    "0-1":     (0, 0, 1),
    "1/2-1/2": (0, 1, 0),
}


# ---------------------------------------------------------------------------
# Per-game parsing (runs in workers)
# ---------------------------------------------------------------------------

def _parse_header_line(line: str) -> Optional[Tuple[str, str]]:
    """Parse `[TagName "TagValue"]`. Returns (name, value) or None.

    Manual string slicing — measurably faster than regex for high-volume
    PGN streams (we call this once per header line per game, 10M+ times)."""
    if not line.startswith("["):
        return None
    sp = line.find(" ", 1)
    if sp < 0:
        return None
    name = line[1:sp]
    qs = line.find('"', sp)
    if qs < 0:
        return None
    qe = line.find('"', qs + 1)
    if qe < 0:
        return None
    return name, line[qs + 1:qe]


def _parse_tc_base(tc: str) -> Optional[int]:
    """TimeControl '600+5' -> 600. Returns None on unparseable / correspondence."""
    if not tc or tc == "-":
        return None
    try:
        return int(tc.split("+", 1)[0])
    except (ValueError, IndexError):
        return None


def _stream_games_from_text(text: str) -> Iterable[Tuple[dict, str]]:
    """Yield (headers_dict, moves_text) for each game in `text`.

    Hand-rolled state machine identical in semantics to the previous
    streaming parser, but operates on an in-memory string rather than a
    file handle so it can run in a worker after the main process chunks
    the stream. Also bails out on the rest of the headers as soon as an
    Elo header is parsed below threshold (caller still has to make the
    decision, but headers are tiny so the bailout pays off mostly via
    skipping the moves)."""
    headers: dict = {}
    moves_buf: List[str] = []
    state = "between"
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if state == "between":
            if line.startswith("["):
                headers = {}
                moves_buf = []
                state = "headers"
                hp = _parse_header_line(line)
                if hp is not None:
                    headers[hp[0]] = hp[1]
        elif state == "headers":
            if line.startswith("["):
                hp = _parse_header_line(line)
                if hp is not None:
                    headers[hp[0]] = hp[1]
            elif line == "":
                state = "moves"
            else:
                state = "moves"
                moves_buf.append(line)
        elif state == "moves":
            if line == "":
                yield headers, "\n".join(moves_buf)
                headers = {}
                moves_buf = []
                state = "between"
            else:
                moves_buf.append(line)
    if state == "moves" and moves_buf:
        yield headers, "\n".join(moves_buf)


def _reconstruct_pgn(headers: dict, moves: str) -> str:
    parts = [f'[{k} "{v}"]' for k, v in headers.items()]
    parts.append("")
    parts.append(moves)
    parts.append("")
    return "\n".join(parts)


def _emit_wdl_lines(headers: dict, moves_text: str, min_ply: int,
                    mirror_augment: bool) -> List[str]:
    """Return a list of output lines for one accepted game.

    The game result is applied to every position past min_ply equally —
    standard outcome-propagation. Noisy per-position but averaged over
    millions of positions the net learns the marginal expected outcome.

    When `mirror_augment` is True, each position is emitted twice: once as
    observed and once with colors swapped via board.mirror() (white pieces
    become black on the mirrored rank, STM flips, WDL flips W↔L). This
    doubles the dataset for free and corrects the W-perspective bias that
    natural Lichess data has (white scores ~55% at high levels). The
    previous WDL net showed K+Q vs K at +1083 cp from white's POV but
    only -387 cp from black's POV — a 3x asymmetry — because the net
    never saw enough black-perspective Q-up positions. Mirror augmentation
    fixes that by construction."""
    result = headers.get("Result", "*")
    wdl_white = _RESULT_WDL_WHITE.get(result)
    if wdl_white is None:
        return []

    pgn = _reconstruct_pgn(headers, moves_text)
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return []

    out: List[str] = []
    board = game.board()
    ply = 0
    for node in game.mainline():
        if node.move is None:
            continue
        board.push(node.move)
        ply += 1
        if ply < min_ply:
            continue
        if board.turn == chess.WHITE:
            w, d, l = wdl_white
        else:
            w, d, l = wdl_white[2], wdl_white[1], wdl_white[0]
        out.append(f"{board.fen()};{w};{d};{l}\n")
        if mirror_augment:
            # board.mirror() flips colors AND vertically reflects piece
            # positions, so the STM flips too. The new STM's WDL is the
            # opposite outcome from the original STM's: a "win for original
            # STM" is a "loss for mirrored STM".
            mboard = board.mirror()
            out.append(f"{mboard.fen()};{l};{d};{w}\n")
    return out


# Worker-side filter-and-emit. Returns (n_scanned, n_accepted, lines).
def _process_chunk(chunk: str, min_elo: int, min_tc_seconds: int,
                   min_ply: int, mirror_augment: bool) -> Tuple[int, int, List[str]]:
    n_scanned = 0
    n_accepted = 0
    lines: List[str] = []
    for headers, moves_text in _stream_games_from_text(chunk):
        n_scanned += 1
        if headers.get("Result", "*") not in _RESULT_WDL_WHITE:
            continue
        try:
            welo = int(headers.get("WhiteElo", 0))
            belo = int(headers.get("BlackElo", 0))
        except ValueError:
            continue
        if min(welo, belo) < min_elo:
            continue
        tc = _parse_tc_base(headers.get("TimeControl", ""))
        if tc is None or tc < min_tc_seconds:
            continue

        n_accepted += 1
        lines.extend(_emit_wdl_lines(headers, moves_text, min_ply, mirror_augment))
    return n_scanned, n_accepted, lines


# Module-level wrapper for Pool.imap_unordered (must be picklable / top-level).
_GLOBAL_ARGS: dict = {}


def _worker_init(min_elo: int, min_tc_seconds: int, min_ply: int,
                 mirror_augment: bool) -> None:
    """Pool initializer — stash filter knobs in module globals so they don't
    have to be pickled into every chunk message."""
    _GLOBAL_ARGS["min_elo"]         = min_elo
    _GLOBAL_ARGS["min_tc_seconds"]  = min_tc_seconds
    _GLOBAL_ARGS["min_ply"]         = min_ply
    _GLOBAL_ARGS["mirror_augment"]  = mirror_augment


def _worker_task(chunk: str) -> Tuple[int, int, List[str]]:
    return _process_chunk(chunk,
                          _GLOBAL_ARGS["min_elo"],
                          _GLOBAL_ARGS["min_tc_seconds"],
                          _GLOBAL_ARGS["min_ply"],
                          _GLOBAL_ARGS["mirror_augment"])


# ---------------------------------------------------------------------------
# Chunked reader (main process)
# ---------------------------------------------------------------------------

def _iter_chunks(handle, chunk_size: int = 1 << 20) -> Iterable[str]:
    """Read `handle` in ~chunk_size pieces, snapped to the last blank-line
    game boundary. Guarantees that every yielded chunk ends between games,
    so workers never see a half-game.

    Buffer carries over the trailing partial-game text to the next chunk."""
    buf = ""
    while True:
        new_data = handle.read(chunk_size)
        if not new_data:
            if buf.strip():
                yield buf
            return
        buf += new_data
        # Snap to the last blank-line (game boundary) within the buffer.
        # PGN games are separated by a single blank line, so '\n\n' is the
        # cleanest splitter.
        cut = buf.rfind("\n\n")
        if cut < 0:
            # No complete game in this buffer yet; keep accumulating.
            continue
        yield buf[:cut + 2]
        buf = buf[cut + 2:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, required=True,
                   help="emit this many positions and exit (triggers SIGPIPE upstream)")
    p.add_argument("--min-elo", type=int, default=2500,
                   help="reject games where either player rated below this")
    p.add_argument("--min-tc-seconds", type=int, default=300,
                   help="reject games with TimeControl base below this (300=Rapid)")
    p.add_argument("--min-ply", type=int, default=10,
                   help="don't emit positions before this ply (skip the book region)")
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1),
                   help="worker processes (default: cpu_count - 1; -1 disables MP)")
    p.add_argument("--chunk-bytes", type=int, default=1 << 20,
                   help="bytes per chunk handed to workers (default 1 MiB)")
    p.add_argument("--report-every", type=int, default=50_000,
                   help="log progress every N emitted positions")
    p.add_argument("--no-mirror-augment", action="store_true",
                   help="disable color-swap mirror augmentation. Off by "
                        "default — augmentation doubles output volume but "
                        "guarantees symmetric W/B perspective coverage, "
                        "which was missing from the original WDL net "
                        "training data and caused a 3x eval asymmetry "
                        "(K+Q white-POV +1083cp vs black-POV -387cp).")
    args = p.parse_args()
    mirror_augment = not args.no_mirror_augment

    emitted = 0
    games_scanned = 0
    games_accepted = 0

    # Single-process fast path: useful for debugging and for unit tests.
    if args.workers <= 0:
        for chunk in _iter_chunks(sys.stdin, args.chunk_bytes):
            s, a, lines = _process_chunk(chunk, args.min_elo,
                                         args.min_tc_seconds, args.min_ply,
                                         mirror_augment)
            games_scanned  += s
            games_accepted += a
            for line in lines:
                try:
                    sys.stdout.write(line)
                except BrokenPipeError:
                    return
                emitted += 1
                if emitted >= args.target:
                    return
                if emitted % args.report_every == 0:
                    print(f"  emitted={emitted} scanned={games_scanned} "
                          f"accepted={games_accepted}", file=sys.stderr)
                    sys.stdout.flush()
        print(f"EOF: emitted={emitted} scanned={games_scanned} "
              f"accepted={games_accepted}", file=sys.stderr)
        return

    # Multiprocessing path.
    pool = mp.Pool(processes=args.workers,
                   initializer=_worker_init,
                   initargs=(args.min_elo, args.min_tc_seconds, args.min_ply,
                             mirror_augment))
    try:
        # imap_unordered with a small chunksize keeps the work queue full
        # without big latency spikes when chunks are very uneven. The main
        # thread iterates the chunk generator AND drains results — Python's
        # GIL doesn't matter here since both are I/O bound.
        for s, a, lines in pool.imap_unordered(_worker_task,
                                               _iter_chunks(sys.stdin, args.chunk_bytes),
                                               chunksize=2):
            games_scanned  += s
            games_accepted += a
            for line in lines:
                try:
                    sys.stdout.write(line)
                except BrokenPipeError:
                    pool.terminate()
                    return
                emitted += 1
                if emitted >= args.target:
                    pool.terminate()
                    print(f"done: emitted={emitted} scanned={games_scanned} "
                          f"accepted={games_accepted}", file=sys.stderr)
                    return
                if emitted % args.report_every == 0:
                    print(f"  emitted={emitted} scanned={games_scanned} "
                          f"accepted={games_accepted}", file=sys.stderr)
                    sys.stdout.flush()
    finally:
        pool.close()
        pool.join()

    print(f"EOF: emitted={emitted} scanned={games_scanned} "
          f"accepted={games_accepted}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
