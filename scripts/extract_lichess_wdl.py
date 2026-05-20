#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Extract game-outcome WDL labels from a streaming Lichess PGN dump.

For each filtered game, every position past `--min-ply` is labeled with the
game's eventual outcome from the side-to-move's perspective:

    <fen>;<W>;<D>;<L>

where (W, D, L) is one of (1,0,0), (0,1,0), (0,0,1) — STM won the game, the
game was drawn, STM lost. These are the "soft" targets a real WDL net wants
(observed game outcomes), as opposed to the cp-derived targets that
extract_lichess_evals.py emits.

Filtering convention matches extract_lichess_evals.py: both-Elo >= --min-elo
(default 2500) and TimeControl base >= --min-tc-seconds (default 300, ~Rapid
and above). Strong-player real-game noise is bounded enough at 2500+ that
WDL labels track theoretical truth closely — at lower ratings, conversion
of winning positions falls off and the targets become noisier.

The fast-filter path reads headers as raw lines and only invokes python-chess
for accepted games. At 2500+ Rapid+ we keep <0.5% of a typical month, so the
per-line reject is the bulk of runtime.

Usage:

    zstd -dc lichess_db_2025-01.pgn.zst | \\
        python scripts/extract_lichess_wdl.py --target 10_000_000 \\
        > data/wdl_training.txt
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from typing import Iterable, Optional

try:
    import chess
    import chess.pgn
except ImportError:
    print("error: requires python-chess. pip install python-chess", file=sys.stderr)
    sys.exit(1)

_HEADER_RE = re.compile(r'\[(\w+)\s+"(.*)"\]')

# Game-result PGN strings -> (W, D, L) from WHITE's perspective. STM-flipping
# happens per position inside the emit loop.
_RESULT_WDL_WHITE = {
    "1-0":     (1.0, 0.0, 0.0),
    "0-1":     (0.0, 0.0, 1.0),
    "1/2-1/2": (0.0, 1.0, 0.0),
}


def parse_tc_base(tc: str) -> Optional[int]:
    """TimeControl '600+5' -> 600. Returns None on unparseable / correspondence."""
    if not tc or tc == "-":
        return None
    try:
        return int(tc.split("+", 1)[0])
    except (ValueError, IndexError):
        return None


def stream_games(handle) -> Iterable[tuple[dict, str]]:
    """Yield (headers_dict, moves_text) for each game in the stream.

    Hand-rolled because python-chess's read_game() eagerly parses moves, which
    we want to avoid for the 99%+ of games we'll reject on Elo alone. State
    machine: between -> headers -> moves -> between."""
    headers: dict = {}
    moves_buf: list[str] = []
    state = "between"
    for raw in handle:
        line = raw.rstrip("\r\n")
        if state == "between":
            if line.startswith("["):
                headers = {}
                moves_buf = []
                state = "headers"
                m = _HEADER_RE.match(line)
                if m:
                    headers[m.group(1)] = m.group(2)
        elif state == "headers":
            if line.startswith("["):
                m = _HEADER_RE.match(line)
                if m:
                    headers[m.group(1)] = m.group(2)
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


def emit_wdl_positions(headers: dict, moves_text: str, min_ply: int):
    """Yield (fen, w_stm, d_stm, l_stm) for every position past min_ply.

    The game result is applied to every position in the game equally — the
    standard "outcome propagation" used in AlphaZero-style WDL training. This
    is noisy per-position (a winning side may blunder in some positions) but
    averaged over millions of positions the net learns the marginal expected
    outcome for the position type."""
    result = headers.get("Result", "*")
    wdl_white = _RESULT_WDL_WHITE.get(result)
    if wdl_white is None:
        return  # decisive only; "*" (unfinished) gets dropped

    pgn = _reconstruct_pgn(headers, moves_text)
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return

    board = game.board()
    ply = 0
    for node in game.mainline():
        if node.move is None:
            continue
        board.push(node.move)
        ply += 1
        if ply < min_ply:
            continue
        # board.turn is the side to move IN this position. Flip the WDL tuple
        # if STM is Black so the label is always "STM's W/D/L".
        if board.turn == chess.WHITE:
            yield board.fen(), wdl_white[0], wdl_white[1], wdl_white[2]
        else:
            yield board.fen(), wdl_white[2], wdl_white[1], wdl_white[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, required=True,
                   help="emit this many positions and exit (triggers SIGPIPE upstream)")
    p.add_argument("--min-elo", type=int, default=2500,
                   help="reject games where either player rated below this. "
                        "2500+ keeps a small slice of titled-player rapid/classical.")
    p.add_argument("--min-tc-seconds", type=int, default=300,
                   help="reject games with TimeControl base below this. 300=5min, "
                        "the floor of Rapid. Going lower (Blitz) adds noise as "
                        "winning positions are blown more often.")
    p.add_argument("--min-ply", type=int, default=10,
                   help="don't emit positions before this ply (skip the book region; "
                        "early-opening positions are useless WDL targets — every game "
                        "starts the same way and outcomes are uncorrelated with the "
                        "first 5 moves at high levels)")
    p.add_argument("--report-every", type=int, default=50000,
                   help="log progress to stderr every N emitted positions")
    args = p.parse_args()

    emitted = 0
    games_scanned = 0
    games_accepted = 0

    try:
        for headers, moves_text in stream_games(sys.stdin):
            games_scanned += 1
            if games_scanned % 200000 == 0:
                print(f"  scanned={games_scanned} accepted={games_accepted} "
                      f"emitted={emitted}", file=sys.stderr)

            # Cheap filters first.
            if headers.get("Result", "*") not in _RESULT_WDL_WHITE:
                continue
            try:
                welo = int(headers.get("WhiteElo", 0))
                belo = int(headers.get("BlackElo", 0))
            except ValueError:
                continue
            if min(welo, belo) < args.min_elo:
                continue
            tc = parse_tc_base(headers.get("TimeControl", ""))
            if tc is None or tc < args.min_tc_seconds:
                continue

            games_accepted += 1
            for fen, w, d, l in emit_wdl_positions(headers, moves_text, args.min_ply):
                # Print integer 0/1 to keep file small; the trainer parses as float.
                print(f"{fen};{int(w)};{int(d)};{int(l)}")
                emitted += 1
                if emitted % args.report_every == 0:
                    print(f"  emitted={emitted} scanned={games_scanned} "
                          f"accepted={games_accepted}", file=sys.stderr)
                    sys.stdout.flush()
                if emitted >= args.target:
                    print(f"done: emitted={emitted} scanned={games_scanned} "
                          f"accepted={games_accepted}", file=sys.stderr)
                    return
    except BrokenPipeError:
        return

    print(f"EOF: emitted={emitted} scanned={games_scanned} "
          f"accepted={games_accepted}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
