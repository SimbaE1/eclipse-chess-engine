#!/usr/bin/env python3
"""
Convert Eclipse self-play PGN → training format.

Each PGN comment has the form "123cp n45678" or "123cp n45678 | AB ...".
This script replays every game, extracts the FEN before each move, and writes
one training line per position:

    <fen>;<score_cp>

Positions with |cp| > max_cp (default 4000) are filtered out, matching the
training pipeline.

Usage:
    python3 scripts/pgn_to_training.py input.pgn [input2.pgn ...] --out training_selfplay.txt
    python3 scripts/pgn_to_training.py *.pgn --out training_selfplay.txt [--max-cp 4000]
"""
import argparse
import re
import sys
from pathlib import Path

try:
    import chess
    import chess.pgn
except ImportError:
    print("chess package not installed: pip install chess")
    sys.exit(1)

_CP_RE = re.compile(r'^(-?\d+)cp\b')


def extract_positions(pgn_path: Path, max_cp: int) -> list[tuple[str, int]]:
    """Return list of (fen, score_cp) from all games in pgn_path."""
    positions = []
    with pgn_path.open() as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for node in game.mainline():
                comment = node.comment.strip()
                m = _CP_RE.match(comment)
                if m:
                    cp = int(m.group(1))
                    if abs(cp) <= max_cp:
                        fen = board.fen()
                        positions.append((fen, cp))
                board.push(node.move)
    return positions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pgn', nargs='+', type=Path, help='input PGN file(s)')
    ap.add_argument('--out', type=Path, required=True, help='output txt file')
    ap.add_argument('--max-cp', type=int, default=4000,
                    help='filter positions with |cp| > max_cp (default 4000)')
    args = ap.parse_args()

    total = 0
    with args.out.open('w') as fout:
        for pgn_path in args.pgn:
            positions = extract_positions(pgn_path, args.max_cp)
            for fen, cp in positions:
                fout.write(f'{fen};{cp}\n')
            total += len(positions)
            print(f'  {pgn_path.name}: {len(positions)} positions')

    print(f'Total: {total} positions → {args.out}')


if __name__ == '__main__':
    main()
