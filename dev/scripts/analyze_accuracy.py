#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Per-game move accuracy report using Stockfish as the reference engine.

Evaluates every position in each PGN game with Stockfish at a fixed depth,
converts centipawn scores to win% (lichess formula), and reports each side's
average move accuracy and average centipawn loss (ACPL).

Uses the same raw-fd UCI driver as bench_vs_stockfish.py: python-chess's
asyncio chess.engine driver hangs on macOS (a select()/TextIOWrapper buffering
mismatch when an engine flushes many lines at once). Runs single-threaded by
default so it can share a spare core without stealing search threads from a
concurrently-running engine match.

    python3 scripts/analyze_accuracy.py data/v3_vs_v2.pgn --depth 14
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile

import chess
import chess.pgn

from bench_vs_stockfish import UciEngine

# Default on-disk cache of computed per-game accuracies. Keyed by
# (abs PGN path, game identity, depth, cap), so re-running on the same file
# (e.g. a match cutechess is still appending to) only evaluates new games.
_DEFAULT_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, ".accuracy_cache.json")


def _game_key(pgn_path: str, game: chess.pgn.Game, depth: int, cap: int) -> str:
    """Stable identity for a game under a given analysis config.

    The move sequence plus both player names uniquely identifies the game;
    depth and cap are folded in because they change the computed numbers.
    """
    moves = ";".join(m.uci() for m in game.mainline_moves())
    ident = "|".join((
        game.headers.get("White", ""), game.headers.get("Black", ""), moves))
    h = hashlib.sha1(ident.encode()).hexdigest()
    return f"{os.path.abspath(pgn_path)}|d{depth}|c{cap}|{h}"


def _load_cache(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(path: str, cache: dict) -> None:
    # Atomic write so an interrupted run can't corrupt the cache file.
    d = os.path.dirname(path) or "."
    try:
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    except OSError:
        pass  # caching is best-effort; never fail the analysis over it


class Analyzer(UciEngine):
    def eval_cp(self, fen: str, depth: int) -> int:
        """Stockfish eval in centipawns from the side-to-move's POV.

        Mate scores are mapped to a large bounded value so downstream win% /
        accuracy math stays finite.
        """
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        score = 0
        while True:
            line = self._readline()
            if line.startswith("info") and " score " in line:
                t = line.split()
                try:
                    si = t.index("score")
                    kind, val = t[si + 1], int(t[si + 2])
                    if kind == "cp":
                        score = val
                    elif kind == "mate":
                        score = (30000 - abs(val)) if val > 0 else -(30000 - abs(val))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("bestmove"):
                return score


def win_percent(cp: int) -> float:
    """Lichess win% formula, from White's perspective."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(loss_win: float) -> float:
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * loss_win) - 3.1669))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--cap-cp", type=int, default=1000,
                    help="clamp |eval| to this before ACPL/win%% so a decided "
                         "game (and mate scores) don't dominate the averages")
    ap.add_argument("--cache", default=_DEFAULT_CACHE,
                    help="JSON cache of computed per-game accuracies "
                         "(default: <repo>/.accuracy_cache.json)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore and do not update the cache")
    args = ap.parse_args()
    cap = args.cap_cp

    def clamp(cp: int) -> int:
        return max(-cap, min(cap, cp))

    cache = {} if args.no_cache else _load_cache(args.cache)

    # name -> [sum_acc, sum_acpl, n_moves] aggregated across all games
    agg: dict[str, list[float]] = {}

    def record(name: str, side_stats: dict) -> None:
        a = agg.setdefault(name, [0.0, 0.0, 0])
        a[0] += side_stats["sum_acc"]
        a[1] += side_stats["sum_acpl"]
        a[2] += side_stats["n"]

    def render(game_idx: int, entry: dict, cached: bool) -> None:
        tag = "  (cached)" if cached else ""
        print(f"\n=== Game {game_idx}: {entry['white']} (White) vs "
              f"{entry['black']} (Black) [{entry['result']}] ==={tag}", flush=True)
        for key, label, name in (("W", "White", entry["white"]),
                                  ("B", "Black", entry["black"])):
            s = entry["sides"].get(key)
            if s and s["n"]:
                print(f"  {name:>12} ({label}): "
                      f"accuracy={s['sum_acc']/s['n']:5.1f}%  "
                      f"ACPL={s['sum_acpl']/s['n']:4.0f}cp  ({s['n']} moves)",
                      flush=True)
                record(name, s)

    sf = None  # lazily started only if there's an uncached game to evaluate

    def analyze_game(game: chess.pgn.Game) -> dict:
        nonlocal sf
        if sf is None:
            sf = Analyzer(args.stockfish, "sf-analyze")
            sf.configure({"Threads": args.threads, "Hash": args.hash})

        evals: dict[str, int] = {}  # White-POV cp, cached by FEN within the game

        def white_cp(board: chess.Board) -> int:
            fen = board.fen()
            if fen not in evals:
                cp = sf.eval_cp(fen, args.depth)               # side-to-move POV
                evals[fen] = cp if board.turn == chess.WHITE else -cp
            return evals[fen]

        board = game.board()
        cp_before = clamp(white_cp(board))
        win_before = win_percent(cp_before)
        acpl = {chess.WHITE: [], chess.BLACK: []}
        acc = {chess.WHITE: [], chess.BLACK: []}

        for move in game.mainline_moves():
            mover = board.turn
            board.push(move)
            cp_after = clamp(white_cp(board))
            win_after = win_percent(cp_after)
            if mover == chess.WHITE:
                loss_cp = max(0, cp_before - cp_after)
                loss_win = max(0.0, win_before - win_after)
            else:
                loss_cp = max(0, cp_after - cp_before)
                loss_win = max(0.0, win_after - win_before)
            acpl[mover].append(loss_cp)
            acc[mover].append(move_accuracy(loss_win))
            cp_before, win_before = cp_after, win_after

        sides = {}
        for key, side in (("W", chess.WHITE), ("B", chess.BLACK)):
            sides[key] = {
                "n": len(acc[side]),
                "sum_acc": sum(acc[side]),
                "sum_acpl": sum(acpl[side]),
            }
        return {
            "white": game.headers.get("White", "White"),
            "black": game.headers.get("Black", "Black"),
            "result": game.headers.get("Result", "?"),
            "sides": sides,
        }

    with open(args.pgn) as f:
        game_idx = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if not list(game.mainline_moves()):
                continue  # unfinished / empty game
            game_idx += 1
            key = _game_key(args.pgn, game, args.depth, cap)
            if not args.no_cache and key in cache:
                render(game_idx, cache[key], cached=True)
                continue
            entry = analyze_game(game)
            if not args.no_cache:
                cache[key] = entry
                _save_cache(args.cache, cache)  # persist after each game
            render(game_idx, entry, cached=False)

    if sf is not None:
        sf.quit()

    print(f"\n=== Aggregate over {game_idx} game(s), Stockfish depth {args.depth} ===")
    for name in sorted(agg):
        sum_acc, sum_acpl, n = agg[name]
        print(f"  {name:>12}: accuracy={sum_acc/n:5.1f}%  ACPL={sum_acpl/n:4.0f}cp  "
              f"({int(n)} moves)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
