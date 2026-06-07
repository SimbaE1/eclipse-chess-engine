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
import json
import re
import sys
from pathlib import Path
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


class _RawStdinCounter:
    """Wraps sys.stdin.buffer, yields decoded text lines, tracks raw bytes read."""

    def __init__(self):
        self.bytes_read = 0
        self._buf = b""

    def __iter__(self):
        while True:
            chunk = sys.stdin.buffer.read(1 << 16)
            if not chunk:
                if self._buf:
                    yield self._buf.decode("utf-8", errors="replace")
                    self._buf = b""
                break
            self.bytes_read += len(chunk)
            combined = self._buf + chunk
            lines = combined.split(b"\n")
            self._buf = lines[-1]
            for line in lines[:-1]:
                yield line.decode("utf-8", errors="replace") + "\n"


def _fast_forward(n_bytes: int) -> None:
    """Read and discard n_bytes from sys.stdin.buffer."""
    remaining = n_bytes
    while remaining > 0:
        chunk = sys.stdin.buffer.read(min(remaining, 1 << 20))
        if not chunk:
            break
        remaining -= len(chunk)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, default=None,
                   help="stop after this many positions (triggers SIGPIPE upstream). "
                        "Omit to run to EOF and extract everything.")
    p.add_argument("--output", type=Path, default=None,
                   help="write positions here instead of stdout; enables checkpointing")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore any existing checkpoint and restart from scratch")
    p.add_argument("--min-elo", type=int, default=2500,
                   help="reject games where either player rated below this")
    p.add_argument("--min-tc-seconds", type=int, default=300,
                   help="reject games with TimeControl base time below this (in seconds)")
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
    out = sys.stdout
    ckpt_path: Optional[Path] = None

    if args.output:
        ckpt_path = Path(str(args.output) + ".ckpt")
        skip_bytes = 0
        if not args.no_resume and ckpt_path.exists():
            try:
                ckpt = json.loads(ckpt_path.read_text())
                skip_bytes = int(ckpt.get("bytes_consumed", 0))
                emitted    = int(ckpt.get("emitted", 0))
                print(f"  resuming: skip={skip_bytes}b emitted={emitted}", file=sys.stderr)
            except Exception:
                skip_bytes = 0
                emitted = 0
        if skip_bytes > 0:
            _fast_forward(skip_bytes)
        mode = "a" if emitted > 0 and not args.no_resume else "w"
        out = open(args.output, mode)  # noqa: SIM115

    reader = _RawStdinCounter()
    if args.output and emitted > 0:
        reader.bytes_read = skip_bytes  # type: ignore[possibly-undefined]

    try:
        for headers, moves_text in stream_games(reader):
            games_scanned += 1
            if games_scanned % 200000 == 0:
                print(f"  scanned={games_scanned} accepted={games_accepted} "
                      f"emitted={emitted}", file=sys.stderr)

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
            if "%eval" not in moves_text:
                continue

            games_accepted += 1
            for fen, cp in emit_eval_positions(headers, moves_text,
                                                args.min_ply, args.mate_cp):
                print(f"{fen};{cp}", file=out)
                emitted += 1
                if emitted % args.report_every == 0:
                    if ckpt_path is not None:
                        ckpt_path.write_text(json.dumps({
                            "bytes_consumed": reader.bytes_read,
                            "emitted": emitted,
                        }))
                    print(f"  emitted={emitted} scanned={games_scanned} "
                          f"accepted={games_accepted}", file=sys.stderr)
                    out.flush()
                if args.target is not None and emitted >= args.target:
                    if ckpt_path is not None:
                        ckpt_path.write_text(json.dumps({
                            "bytes_consumed": reader.bytes_read,
                            "emitted": emitted,
                        }))
                    print(f"done: emitted={emitted} scanned={games_scanned} "
                          f"accepted={games_accepted}", file=sys.stderr)
                    return
    except BrokenPipeError:
        return
    finally:
        if out is not sys.stdout:
            out.close()

    print(f"EOF: emitted={emitted} scanned={games_scanned} "
          f"accepted={games_accepted}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
