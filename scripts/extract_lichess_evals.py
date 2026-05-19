#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Extract Lichess-provided %eval annotations from a streaming PGN dump.

Lichess runs Stockfish at depth ~22 server-side on a large fraction of rated
games and embeds the score directly in the PGN as comments like:

    1. e4 { [%eval 0.27] [%clk 0:03:00] } e5 { [%eval 0.30] } ...

That eval is *much* better than anything we'd label locally at depth 8, and
it's already in the file. This script harvests every %eval-annotated position
from games matching our quality bar (Elo + time control) and outputs them in
the same `<fen>;<score_cp>` format train_halfkav2.py expects.

Convention conversions:
  - Lichess %eval is in pawns, from WHITE's perspective.
  - Lichess %eval mate scores are "#N" (positive = White mates) / "#-N".
  - Training expects centipawns from SIDE-TO-MOVE's perspective.
  - So: stm_cp = sign_flip(white_cp) if side_to_move == Black else white_cp.

The fast-filter loop reads PGNs as raw lines and only hands accepted games to
python-chess for proper SAN parsing. This matters at scale: at 2500+ Elo we
keep maybe 0.5% of games in a Lichess monthly dump, so the per-line filter is
~200x faster than full parsing.
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

# Match one of: "[%eval 0.34]", "[%eval -1.2]", "[%eval #3]", "[%eval #-2]"
_EVAL_RE   = re.compile(r"\[%eval (#?-?\d+(?:\.\d+)?)\]")
_HEADER_RE = re.compile(r'\[(\w+)\s+"(.*)"\]')


def parse_eval_to_cp(eval_str: str, mate_cp: int) -> int:
    """Lichess eval string -> centipawns (White's perspective)."""
    if eval_str.startswith("#"):
        n = int(eval_str[1:])
        # Lichess mate sign convention matches White's POV: #N positive = White
        # mates, negative = Black mates. Saturate to fixed cp so the sigmoid
        # loss doesn't see infinity.
        return mate_cp if n >= 0 else -mate_cp
    return int(round(float(eval_str) * 100))


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
            # blank/junk lines outside any game just get dropped
        elif state == "headers":
            if line.startswith("["):
                m = _HEADER_RE.match(line)
                if m:
                    headers[m.group(1)] = m.group(2)
            elif line == "":
                state = "moves"
            else:
                # Moves can also start without a separating blank line on some
                # malformed dumps - tolerate it.
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
    """Glue headers + moves back into a valid mini-PGN for python-chess."""
    parts = [f'[{k} "{v}"]' for k, v in headers.items()]
    parts.append("")
    parts.append(moves)
    parts.append("")
    return "\n".join(parts)


def emit_eval_positions(headers: dict, moves_text: str,
                        min_ply: int, mate_cp: int):
    """Yield (fen, stm_cp) for every %eval-annotated move in the game."""
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
        comment = node.comment or ""
        m = _EVAL_RE.search(comment)
        if not m:
            continue
        white_cp = parse_eval_to_cp(m.group(1), mate_cp)
        # board.turn is the side TO MOVE in this position (opposite of who
        # just moved). The eval was reported for this position, by convention
        # from White's perspective, so flip when Black is to move.
        stm_cp = white_cp if board.turn == chess.WHITE else -white_cp
        yield board.fen(), stm_cp


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, required=True,
                   help="emit this many positions and exit (triggers SIGPIPE upstream)")
    p.add_argument("--min-elo", type=int, default=2500,
                   help="reject games where either player rated below this. "
                        "2500+ gets you a small slice of titled-player rapid/classical.")
    p.add_argument("--min-tc-seconds", type=int, default=300,
                   help="reject games with TimeControl base time below this (in seconds).")
    p.add_argument("--min-ply", type=int, default=10,
                   help="don't emit positions before this ply (skip the book region)")
    p.add_argument("--mate-cp", type=int, default=1500,
                   help="centipawn value used to saturate mate-in-N scores")
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

            # Cheap filters first: Elo + time control header reads, no
            # tokenization of moves.
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
            # Skip games with no eval annotations entirely - very fast reject.
            if "%eval" not in moves_text:
                continue

            games_accepted += 1
            for fen, cp in emit_eval_positions(headers, moves_text,
                                                args.min_ply, args.mate_cp):
                print(f"{fen};{cp}")
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
        # Downstream closed (Ctrl-C upstream curl etc.). Quiet exit.
        return

    print(f"EOF: emitted={emitted} scanned={games_scanned} "
          f"accepted={games_accepted}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
