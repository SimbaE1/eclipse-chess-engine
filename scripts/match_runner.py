#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
UCI match runner for Eclipse engine benchmarking.
Plays games between two UCI engines and reports W/L/D + Elo estimates.

Usage examples:
  # Head-to-head: halfkav2 vs 3.26M (200 games, 100ms/move)
  python3 scripts/match_runner.py head-to-head \
      --engine1 build/src/eclipse --eval1 data/eclipse_halfkav2_v2.nnue \
      --engine2 build/src/eclipse --eval2 data/eclipse_v3_3.26M.nnue \
      --movetime 100 --games 200 --out results_hkav2_vs_326.pgn

  # Calibrate vs Stockfish at a known Elo level
  python3 scripts/match_runner.py vs-stockfish \
      --engine build/src/eclipse --eval data/eclipse_halfkav2_v2.nnue \
      --sf-elo 1200 --games 100 --movetime 100
"""

import argparse
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chess
import chess.pgn


# ---------------------------------------------------------------------------
# Opening book — 20 short, balanced openings to reduce first-move bias
# ---------------------------------------------------------------------------
OPENINGS = [
    "e2e4",
    "d2d4",
    "e2e4 e7e5",
    "d2d4 d7d5",
    "e2e4 c7c5",
    "e2e4 e7e6",
    "d2d4 g8f6",
    "e2e4 e7e5 g1f3 b8c6",
    "d2d4 d7d5 c2c4",
    "e2e4 e7e5 g1f3 b8c6 f1b5",
    "e2e4 e7e5 g1f3 b8c6 f1c4",
    "d2d4 g8f6 c2c4 e7e6",
    "d2d4 g8f6 c2c4 g7g6",
    "e2e4 c7c6",
    "e2e4 d7d5",
    "e2e4 g7g6",
    "c2c4",
    "g1f3",
    "e2e4 e7e5 g1f3 g8f6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
]


# ---------------------------------------------------------------------------
# UCI engine wrapper
# ---------------------------------------------------------------------------
class UCIEngine:
    def __init__(self, cmd: str, name_hint: str = "engine"):
        self.cmd = cmd
        self.name_hint = name_hint
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._q: queue.Queue = queue.Queue()

    def start(self):
        self._proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

    def _reader(self):
        for line in self._proc.stdout:
            self._q.put(line.rstrip())

    def send(self, cmd: str):
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def read_until(self, prefix: str, timeout: float = 60.0) -> Optional[str]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._q.get(timeout=min(remaining, 1.0))
                if line.startswith(prefix):
                    return line
            except queue.Empty:
                pass
        return None

    def uci_init(self, eval_file: Optional[str] = None,
                 policy_file: Optional[str] = None,
                 hash_mb: int = 32,
                 threads: int = 1,
                 ab_threads: int = 1,
                 extra_options: Optional[dict] = None) -> str:
        self.send("uci")
        resp = self.read_until("uciok", timeout=10)
        if resp is None:
            raise RuntimeError(f"{self.name_hint}: no uciok")
        if eval_file:
            self.send(f"setoption name EvalFile value {eval_file}")
        if policy_file:
            self.send(f"setoption name PolicyFile value {policy_file}")
        self.send(f"setoption name Hash value {hash_mb}")
        self.send(f"setoption name Threads value {threads}")
        self.send(f"setoption name AbThreads value {ab_threads}")
        if extra_options:
            for k, v in extra_options.items():
                self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.read_until("readyok", timeout=30)
        return resp

    def new_game(self):
        self.send("ucinewgame")
        self.send("isready")
        self.read_until("readyok", timeout=10)

    def get_move(self, fen: str, moves_so_far: list[str], movetime_ms: int) -> Optional[str]:
        if moves_so_far:
            self.send(f"position fen {fen} moves {' '.join(moves_so_far)}")
        else:
            self.send(f"position fen {fen}")
        self.send(f"go movetime {movetime_ms}")
        resp = self.read_until("bestmove", timeout=movetime_ms / 1000.0 + 15)
        if resp is None:
            return None
        parts = resp.split()
        if len(parts) < 2:
            return None
        return parts[1] if parts[1] != "(none)" else None

    def quit(self):
        try:
            self.send("quit")
        except Exception:
            pass
        if self._proc:
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# ---------------------------------------------------------------------------
# Stockfish wrapper with known-Elo skill settings
# ---------------------------------------------------------------------------
class StockfishEngine(UCIEngine):
    SF_ELO_TABLE = {
        800: ("Skill Level", 0),
        1000: ("Skill Level", 3),
        1200: ("Skill Level", 6),
        1400: ("Skill Level", 9),
        1600: ("Skill Level", 12),
        1800: ("Skill Level", 15),
        2000: ("Skill Level", 17),
        2200: ("Skill Level", 19),
        2400: ("Skill Level", 20),
    }

    def uci_init_sf(self, target_elo: int, hash_mb: int = 32, threads: int = 1):
        self.send("uci")
        resp = self.read_until("uciok", timeout=10)
        if resp is None:
            raise RuntimeError("Stockfish: no uciok")

        if target_elo in self.SF_ELO_TABLE:
            option, val = self.SF_ELO_TABLE[target_elo]
            self.send(f"setoption name {option} value {val}")
        else:
            # Use UCI_LimitStrength if available for finer-grained control
            self.send("setoption name UCI_LimitStrength value true")
            self.send(f"setoption name UCI_Elo value {target_elo}")

        self.send(f"setoption name Hash value {hash_mb}")
        self.send(f"setoption name Threads value {threads}")
        self.send("isready")
        self.read_until("readyok", timeout=30)


# ---------------------------------------------------------------------------
# ELO calculation
# ---------------------------------------------------------------------------
def elo_diff(wins: int, losses: int, draws: int) -> tuple[float, float]:
    """Return (elo_diff, error_margin_95pct). Returns (0, inf) on degenerate input."""
    n = wins + losses + draws
    if n == 0:
        return 0.0, float("inf")
    score = wins + 0.5 * draws
    score_rate = score / n
    score_rate = max(0.001, min(0.999, score_rate))
    diff = -400.0 * math.log10(1.0 / score_rate - 1.0)
    # 95% confidence interval via normal approximation of score variance
    variance = score_rate * (1.0 - score_rate) / n
    std_err = math.sqrt(variance)
    elo_err = 400.0 * std_err / (math.log(10) * score_rate * (1.0 - score_rate))
    return diff, elo_err * 1.96


def elo_from_score_rate(rate: float, anchor_elo: float) -> float:
    rate = max(0.001, min(0.999, rate))
    return anchor_elo - 400.0 * math.log10(1.0 / rate - 1.0)


# ---------------------------------------------------------------------------
# Game logic
# ---------------------------------------------------------------------------
@dataclass
class GameResult:
    winner: Optional[str]  # "e1", "e2", or None (draw)
    termination: str
    moves_played: int
    pgn: str = ""


def play_game(
    engine1: UCIEngine,
    engine2: UCIEngine,
    opening_moves: list[str],
    movetime_ms: int,
    max_moves: int = 250,
) -> GameResult:
    """Play one game. Engine1=White for this call; caller flips sides."""
    board = chess.Board()
    # Apply opening moves
    for uci in opening_moves:
        try:
            board.push_uci(uci)
        except ValueError:
            break

    engines = [engine1, engine2]  # index 0=White if even ply, else Black
    engine1.new_game()
    engine2.new_game()

    start_fen = chess.STARTING_FEN   # engines receive full move list from start
    moves_so_far: list[str] = list(opening_moves[:board.ply()])

    game = chess.pgn.Game()
    game.headers["White"] = engine1.name_hint
    game.headers["Black"] = engine2.name_hint
    node = game

    for uci in opening_moves[:board.ply()]:
        try:
            node = node.add_variation(chess.Move.from_uci(uci))
        except Exception:
            pass

    move_count = 0
    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        # Whose turn: white=0, black=1
        engine_idx = 0 if board.turn == chess.WHITE else 1
        eng = engines[engine_idx]

        mv_uci = eng.get_move(start_fen, moves_so_far, movetime_ms)
        if mv_uci is None or mv_uci == "0000":
            # Null move: check if the board state explains it (no legal moves).
            if board.is_checkmate():
                winner = "e2" if engine_idx == 0 else "e1"  # side to move lost
                return GameResult(winner=winner, termination="CHECKMATE",
                                  moves_played=move_count)
            if board.is_stalemate():
                return GameResult(winner=None, termination="STALEMATE",
                                  moves_played=move_count)
            # Genuine engine failure / resignation → other side wins
            winner = "e2" if engine_idx == 0 else "e1"
            return GameResult(winner=winner, termination="engine-error",
                              moves_played=move_count)

        try:
            move = board.parse_uci(mv_uci)
            if move not in board.legal_moves:
                winner = "e2" if engine_idx == 0 else "e1"
                return GameResult(winner=winner, termination="illegal-move",
                                  moves_played=move_count)
        except ValueError:
            winner = "e2" if engine_idx == 0 else "e1"
            return GameResult(winner=winner, termination="illegal-move",
                              moves_played=move_count)

        node = node.add_variation(move)
        board.push(move)
        moves_so_far.append(mv_uci)
        move_count += 1

    # Determine result
    outcome = board.outcome(claim_draw=True)
    termination = "normal"
    winner = None

    if outcome is None:
        termination = "max-moves"
    elif outcome.winner == chess.WHITE:
        winner = "e1"
        termination = outcome.termination.name
    elif outcome.winner == chess.BLACK:
        winner = "e2"
        termination = outcome.termination.name
    else:
        termination = outcome.termination.name

    result_str = "*"
    if outcome:
        result_str = outcome.result()
    game.headers["Result"] = result_str

    pgn_str = str(game)
    return GameResult(winner=winner, termination=termination,
                      moves_played=move_count, pgn=pgn_str)


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------
@dataclass
class MatchStats:
    wins_e1: int = 0
    wins_e2: int = 0
    draws: int = 0
    errors: int = 0
    games_played: int = 0

    def score_e1(self) -> float:
        return self.wins_e1 + 0.5 * self.draws

    def score_rate_e1(self) -> float:
        n = self.wins_e1 + self.wins_e2 + self.draws
        return self.score_e1() / n if n > 0 else 0.5

    def print_interim(self, label: str = ""):
        n = self.wins_e1 + self.wins_e2 + self.draws
        if n == 0:
            return
        diff, err = elo_diff(self.wins_e1, self.wins_e2, self.draws)
        rate = self.score_rate_e1()
        tag = f"[{label}] " if label else ""
        print(f"  {tag}+{self.wins_e1} ={self.draws} -{self.wins_e2}"
              f"  ({n} games, {rate*100:.1f}%)"
              f"  Elo diff: {diff:+.0f} ± {err:.0f}")
        sys.stdout.flush()


def run_match(
    engine1: UCIEngine,
    engine2: UCIEngine,
    n_games: int,
    movetime_ms: int,
    out_pgn: Optional[str] = None,
    label_e1: str = "E1",
    label_e2: str = "E2",
) -> MatchStats:
    stats = MatchStats()
    pgn_lines = []

    print(f"\n=== Match: {label_e1} vs {label_e2} ===")
    print(f"    {n_games} games, {movetime_ms}ms/move")

    for g in range(n_games):
        opening_idx = g % len(OPENINGS)
        opening_str = OPENINGS[opening_idx]
        opening_moves = opening_str.split()

        # Alternate colours every game
        if g % 2 == 0:
            white_eng, black_eng = engine1, engine2
            e1_is_white = True
        else:
            white_eng, black_eng = engine2, engine1
            e1_is_white = False

        try:
            result = play_game(white_eng, black_eng, opening_moves, movetime_ms)
        except Exception as exc:
            print(f"  Game {g+1}: ERROR — {exc}")
            stats.errors += 1
            continue

        # Map engine-side winner to E1/E2
        if result.winner == "e1":  # white won
            if e1_is_white:
                stats.wins_e1 += 1
            else:
                stats.wins_e2 += 1
        elif result.winner == "e2":  # black won
            if e1_is_white:
                stats.wins_e2 += 1
            else:
                stats.wins_e1 += 1
        else:
            stats.draws += 1

        stats.games_played += 1

        if result.pgn:
            pgn_lines.append(result.pgn)

        # Progress every 10 games
        if (g + 1) % 10 == 0 or (g + 1) == n_games:
            stats.print_interim(f"game {g+1}/{n_games}")

    if out_pgn and pgn_lines:
        with open(out_pgn, "w") as f:
            f.write("\n\n".join(pgn_lines))
        print(f"  PGN saved → {out_pgn}")

    return stats


# ---------------------------------------------------------------------------
# Calibrated Elo from Stockfish matches
# ---------------------------------------------------------------------------
def calibrate_elo(
    eclipse_bin: str,
    eval_file: str,
    stockfish_bin: str,
    sf_elos: list[int],
    games_per_level: int,
    movetime_ms: int,
    engine_label: str = "Eclipse",
    threads: int = 1,
    hash_mb: int = 32,
) -> list[tuple[int, MatchStats]]:
    results = []
    for sf_elo in sf_elos:
        print(f"\n--- Calibration vs Stockfish @ ~{sf_elo} Elo ---")
        eng = UCIEngine(eclipse_bin.split(), name_hint=engine_label)
        sf  = StockfishEngine(stockfish_bin.split(), name_hint=f"SF{sf_elo}")
        eng.start()
        sf.start()
        eng.uci_init(eval_file=eval_file, hash_mb=hash_mb, threads=threads, ab_threads=1)
        sf.uci_init_sf(target_elo=sf_elo, hash_mb=32, threads=1)

        stats = run_match(eng, sf, games_per_level, movetime_ms,
                          label_e1=engine_label, label_e2=f"SF~{sf_elo}")
        results.append((sf_elo, stats))
        eng.quit()
        sf.quit()
    return results


def estimate_elo_from_calibration(results: list[tuple[int, MatchStats]]) -> Optional[float]:
    """Simple linear interpolation: find the Sf Elo where score rate ≈ 50%."""
    if not results:
        return None
    # sort by sf_elo ascending
    results = sorted(results, key=lambda x: x[0])
    rates = [(sf_elo, s.score_rate_e1()) for sf_elo, s in results]

    # Find bracket where rate crosses 0.5
    for i in range(len(rates) - 1):
        elo_lo, r_lo = rates[i]
        elo_hi, r_hi = rates[i + 1]
        if (r_lo >= 0.5) != (r_hi >= 0.5):
            # Interpolate
            t = (0.5 - r_lo) / (r_hi - r_lo)
            return elo_lo + t * (elo_hi - elo_lo)

    # Extrapolate from best bracket
    if rates[0][1] < 0.5:
        elo, r = rates[0]
        diff, _ = elo_diff(results[0][1].wins_e1, results[0][1].wins_e2, results[0][1].draws)
        return elo + diff
    else:
        elo, r = rates[-1]
        diff, _ = elo_diff(results[-1][1].wins_e1, results[-1][1].wins_e2, results[-1][1].draws)
        return elo + diff


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_head_to_head(args):
    e1_cmd = args.engine1.split()
    e2_cmd = args.engine2.split()
    e1 = UCIEngine(e1_cmd, name_hint=args.label1 or "Engine1")
    e2 = UCIEngine(e2_cmd, name_hint=args.label2 or "Engine2")
    e1.start()
    e2.start()
    e1.uci_init(eval_file=args.eval1, hash_mb=args.hash,
                threads=args.threads, ab_threads=args.ab_threads)
    e2.uci_init(eval_file=args.eval2, hash_mb=args.hash,
                threads=args.threads, ab_threads=args.ab_threads)

    stats = run_match(e1, e2, args.games, args.movetime,
                      out_pgn=args.out,
                      label_e1=e1.name_hint,
                      label_e2=e2.name_hint)

    e1.quit()
    e2.quit()

    diff, err = elo_diff(stats.wins_e1, stats.wins_e2, stats.draws)
    rate = stats.score_rate_e1()
    print(f"\n{'='*60}")
    print(f"FINAL  {e1.name_hint} vs {e2.name_hint}")
    print(f"  +{stats.wins_e1}  ={stats.draws}  -{stats.wins_e2}   ({stats.games_played} games)")
    print(f"  Score: {rate*100:.1f}%   Elo diff: {diff:+.0f} ± {err:.0f}")
    print(f"{'='*60}")
    return stats


def cmd_vs_stockfish(args):
    sf_elos = [int(x) for x in args.sf_elos.split(",")]
    results = calibrate_elo(
        eclipse_bin=args.engine,
        eval_file=args.eval,
        stockfish_bin=args.stockfish,
        sf_elos=sf_elos,
        games_per_level=args.games,
        movetime_ms=args.movetime,
        engine_label=args.label or "Eclipse",
    )

    estimated_elo = estimate_elo_from_calibration(results)

    print(f"\n{'='*60}")
    print(f"ELO CALIBRATION SUMMARY  ({args.label or 'Eclipse'})")
    for sf_elo, stats in results:
        diff, err = elo_diff(stats.wins_e1, stats.wins_e2, stats.draws)
        rate = stats.score_rate_e1()
        print(f"  vs SF~{sf_elo:4d}:  +{stats.wins_e1} ={stats.draws} -{stats.wins_e2}"
              f"  {rate*100:.1f}%  diff {diff:+.0f}±{err:.0f}")
    if estimated_elo is not None:
        print(f"\n  Estimated Elo: ~{estimated_elo:.0f}")
    print(f"{'='*60}")


def cmd_full_bench(args):
    """Run head-to-head AND calibration, then summarize everything."""
    results_dir = Path(args.outdir)
    results_dir.mkdir(parents=True, exist_ok=True)

    engine_bin  = args.engine
    eval_new    = args.eval_new
    eval_old    = args.eval_old
    sf_bin      = args.stockfish
    movetime    = args.movetime
    n_h2h       = args.games_h2h
    n_cal       = args.games_cal
    sf_elos     = [int(x) for x in args.sf_elos.split(",")]

    # --- Head-to-head ---
    e1_cmd = engine_bin.split()
    e2_cmd = engine_bin.split()
    e1 = UCIEngine(e1_cmd, name_hint="HalfKAv2")
    e2 = UCIEngine(e2_cmd, name_hint="3.26M")
    e1.start(); e2.start()
    e1.uci_init(eval_file=eval_new, hash_mb=32, threads=1, ab_threads=1)
    e2.uci_init(eval_file=eval_old, hash_mb=32, threads=1, ab_threads=1)

    h2h_stats = run_match(e1, e2, n_h2h, movetime,
                          out_pgn=str(results_dir / "halfkav2_vs_326M.pgn"),
                          label_e1="HalfKAv2", label_e2="3.26M")
    e1.quit(); e2.quit()

    # --- Calibration: new model ---
    cal_new = calibrate_elo(engine_bin, eval_new, sf_bin, sf_elos,
                            n_cal, movetime, "HalfKAv2")

    # --- Calibration: old model ---
    cal_old = calibrate_elo(engine_bin, eval_old, sf_bin, sf_elos,
                            n_cal, movetime, "3.26M")

    # --- Summary ---
    elo_new = estimate_elo_from_calibration(cal_new)
    elo_old = estimate_elo_from_calibration(cal_old)
    h2h_diff, h2h_err = elo_diff(h2h_stats.wins_e1, h2h_stats.wins_e2, h2h_stats.draws)

    summary_path = results_dir / "benchmark_summary.txt"
    lines = [
        "=" * 60,
        "ECLIPSE BENCHMARK SUMMARY",
        f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "HEAD-TO-HEAD: HalfKAv2 vs 3.26M",
        f"  +{h2h_stats.wins_e1} ={h2h_stats.draws} -{h2h_stats.wins_e2}"
        f"  ({h2h_stats.games_played} games)",
        f"  Score: {h2h_stats.score_rate_e1()*100:.1f}%",
        f"  Elo diff (HalfKAv2 - 3.26M): {h2h_diff:+.0f} ± {h2h_err:.0f}",
        "",
        "CALIBRATION: HalfKAv2 vs Stockfish",
    ]
    for sf_elo, stats in cal_new:
        diff, err = elo_diff(stats.wins_e1, stats.wins_e2, stats.draws)
        rate = stats.score_rate_e1()
        lines.append(f"  vs SF~{sf_elo:4d}: {rate*100:.1f}%  diff {diff:+.0f}±{err:.0f}")
    if elo_new:
        lines.append(f"  => Estimated HalfKAv2 Elo: ~{elo_new:.0f}")

    lines += ["", "CALIBRATION: 3.26M vs Stockfish"]
    for sf_elo, stats in cal_old:
        diff, err = elo_diff(stats.wins_e1, stats.wins_e2, stats.draws)
        rate = stats.score_rate_e1()
        lines.append(f"  vs SF~{sf_elo:4d}: {rate*100:.1f}%  diff {diff:+.0f}±{err:.0f}")
    if elo_old:
        lines.append(f"  => Estimated 3.26M Elo: ~{elo_old:.0f}")

    lines += ["", "=" * 60]
    print("\n" + "\n".join(lines))

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary written → {summary_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    # head-to-head
    p_h2h = sub.add_parser("head-to-head", aliases=["h2h"])
    p_h2h.add_argument("--engine1", required=True)
    p_h2h.add_argument("--engine2", required=True)
    p_h2h.add_argument("--eval1", default=None)
    p_h2h.add_argument("--eval2", default=None)
    p_h2h.add_argument("--label1", default=None)
    p_h2h.add_argument("--label2", default=None)
    p_h2h.add_argument("--movetime", type=int, default=100)
    p_h2h.add_argument("--games", type=int, default=200)
    p_h2h.add_argument("--out", default=None)
    p_h2h.add_argument("--hash", type=int, default=32)
    p_h2h.add_argument("--threads", type=int, default=1)
    p_h2h.add_argument("--ab-threads", type=int, default=1)

    # vs-stockfish
    p_sf = sub.add_parser("vs-stockfish")
    p_sf.add_argument("--engine", required=True)
    p_sf.add_argument("--eval", default=None)
    p_sf.add_argument("--stockfish", default="stockfish")
    p_sf.add_argument("--sf-elos", default="800,1000,1200,1400,1600,1800")
    p_sf.add_argument("--label", default=None)
    p_sf.add_argument("--movetime", type=int, default=100)
    p_sf.add_argument("--games", type=int, default=50)

    # full-bench (head-to-head + calibration in one go)
    p_fb = sub.add_parser("full-bench")
    p_fb.add_argument("--engine", required=True)
    p_fb.add_argument("--eval-new", required=True)
    p_fb.add_argument("--eval-old", required=True)
    p_fb.add_argument("--stockfish", default="stockfish")
    p_fb.add_argument("--sf-elos", default="800,1000,1200,1400,1600,1800,2000")
    p_fb.add_argument("--movetime", type=int, default=100)
    p_fb.add_argument("--games-h2h", type=int, default=200)
    p_fb.add_argument("--games-cal", type=int, default=50)
    p_fb.add_argument("--outdir", default="results/benchmark")

    args = p.parse_args()
    if args.command in ("head-to-head", "h2h"):
        cmd_head_to_head(args)
    elif args.command == "vs-stockfish":
        cmd_vs_stockfish(args)
    elif args.command == "full-bench":
        cmd_full_bench(args)


if __name__ == "__main__":
    main()
