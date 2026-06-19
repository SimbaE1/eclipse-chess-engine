#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Engine-vs-engine benchmark using a minimal synchronous UCI driver. Stands in
for cutechess-cli on machines where cutechess is not installed.

Usage (from the eclipse/ repo root):

    python3 scripts/bench_vs_stockfish.py \
        --games 20 --tc 60+1 --opponent-elo 1800 \
        --pgn data/eclipse_vs_sf1800.pgn

Plays an even number of games, alternating colours each pair. Sets Stockfish's
UCI_LimitStrength + UCI_Elo to the requested rating. Prints per-game results
and a final Elo estimate with a 95% Wilson CI.

Drives engines via raw subprocess pipes rather than python-chess's asyncio
transport: on macOS, chess.engine's kqueue-based pipe transport has been
observed to silently drop a queued stdin write mid-game, hanging the bench
forever with both engines idle in their UCI read loop. A blocking
readline()/write() loop has no such failure mode.
"""

from __future__ import annotations

import argparse
import math
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import chess
import chess.pgn


def parse_tc(s: str) -> tuple[float, float]:
    """`'60+1'` -> (60.0, 1.0). Both numbers in seconds."""
    base, _, inc = s.partition("+")
    return float(base), float(inc or 0.0)


def elo_from_score(score: float, n: int) -> tuple[float, float]:
    """Bayeselo-style Elo estimate + 95% half-width from a fraction in [0,1].

    Saturates at +/- 800 Elo for sweeps (0 or 1).
    """
    if score <= 0.0:
        return -800.0, 0.0
    if score >= 1.0:
        return 800.0, 0.0
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    # Wilson 95% interval on the score, mapped through the same logistic.
    z = 1.96
    p = score
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    lo = max(1e-6, centre - margin)
    hi = min(1 - 1e-6, centre + margin)
    elo_lo = -400.0 * math.log10(1.0 / lo - 1.0)
    elo_hi = -400.0 * math.log10(1.0 / hi - 1.0)
    return elo, (elo_hi - elo_lo) / 2.0


class UciEngine:
    """Minimal blocking UCI driver over subprocess pipes."""

    def __init__(self, cmd: str, name: str, transcript=None):
        self.name = name
        self.transcript = transcript
        self.proc = subprocess.Popen(
            [cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        self._out_fd = self.proc.stdout.fileno()
        self._out_buf = b""
        # Drain stderr continuously so a chatty engine can't fill the pipe
        # buffer and block.
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._send("uci")
        self._read_until("uciok")

    def _drain_stderr(self) -> None:
        for line in self.proc.stderr:
            if self.transcript:
                self.transcript.write(
                    f"{time.time():.3f} {self.name} !!! "
                    f"{line.decode(errors='replace').rstrip()}\n")
                self.transcript.flush()

    def _send(self, line: str) -> None:
        if self.transcript:
            self.transcript.write(f"{time.time():.3f} {self.name} >>> {line}\n")
            self.transcript.flush()
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()

    def _readline(self, timeout: float = 180.0) -> str:
        # Read raw bytes ourselves and maintain our own line buffer: select()
        # only reflects the raw fd, so if a prior read already pulled
        # multiple lines into a TextIOWrapper's internal buffer, a later
        # select() can falsely report "not ready" while a buffered readline()
        # would return instantly. Engines like to flush several lines (id
        # name/id author/options/uciok) in one write, so this isn't rare.
        while b"\n" not in self._out_buf:
            ready, _, _ = select.select([self._out_fd], [], [], timeout)
            if not ready:
                if self.transcript:
                    self.transcript.write(
                        f"{time.time():.3f} {self.name} *** TIMEOUT after "
                        f"{timeout:.0f}s waiting for output\n")
                    self.transcript.flush()
                raise TimeoutError(
                    f"{self.name} produced no output for {timeout:.0f}s "
                    f"(cmd={self.proc.args})")
            chunk = os.read(self._out_fd, 65536)
            if not chunk:
                raise EOFError(f"engine exited unexpectedly (cmd={self.proc.args})")
            self._out_buf += chunk
        line, self._out_buf = self._out_buf.split(b"\n", 1)
        line = line.decode(errors="replace").rstrip("\r")
        if self.transcript:
            self.transcript.write(f"{time.time():.3f} {self.name} <<< {line}\n")
            self.transcript.flush()
        return line

    def _read_until(self, token: str) -> None:
        while not self._readline().startswith(token):
            pass

    def configure(self, options: dict) -> None:
        for name, value in options.items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._read_until("readyok")

    def newgame(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def go(self, moves: list[str], wtime: float, btime: float,
            winc: float, binc: float) -> str | None:
        pos = "position startpos"
        if moves:
            pos += " moves " + " ".join(moves)
        self._send(pos)
        self._send(
            f"go wtime {int(wtime * 1000)} btime {int(btime * 1000)} "
            f"winc {int(winc * 1000)} binc {int(binc * 1000)}"
        )
        while True:
            line = self._readline()
            if line.startswith("bestmove"):
                parts = line.split()
                move = parts[1] if len(parts) > 1 else None
                return None if move == "(none)" else move

    def quit(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--tc", default="60+1", help="base+inc seconds, e.g. 60+1")
    ap.add_argument("--opponent-elo", type=int, default=1800)
    ap.add_argument("--eclipse", default=str(repo / "build/src/eclipse"))
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--nnue", default=str(repo / "data/eclipse.nnue"))
    ap.add_argument("--policy", default=str(repo / "data/policy.onnx"))
    ap.add_argument("--pgn", default=str(repo / "data/eclipse_vs_sf.pgn"))
    ap.add_argument("--transcript", default="",
                    help="optional path to write a timestamped UCI transcript")
    ap.add_argument("--threads", type=int, default=1,
                    help="total Eclipse search threads (includes AB threads)")
    ap.add_argument("--ab-threads", type=int, default=1,
                    help="Eclipse threads dedicated to the AB verifier")
    ap.add_argument("--sf-threads", type=int, default=1,
                    help="Stockfish search threads")
    args = ap.parse_args()

    base, inc = parse_tc(args.tc)

    transcript = open(args.transcript, "w") if args.transcript else None

    eclipse = UciEngine(args.eclipse, "eclipse", transcript)
    eclipse.configure({"EvalFile": args.nnue, "PolicyFile": args.policy,
                        "Threads": args.threads, "AbThreads": args.ab_threads})

    sf = UciEngine(args.stockfish, "stockfish", transcript)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.opponent_elo,
                   "Threads": args.sf_threads})

    wins = draws = losses = 0  # from Eclipse's perspective
    pgn_path = Path(args.pgn)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    pgn_fh = pgn_path.open("w")
    t_start = time.time()

    try:
        for g in range(args.games):
            eclipse_is_white = (g % 2 == 0)
            white = eclipse if eclipse_is_white else sf
            black = sf if eclipse_is_white else eclipse
            white_name = "Eclipse" if eclipse_is_white else f"SF-{args.opponent_elo}"
            black_name = f"SF-{args.opponent_elo}" if eclipse_is_white else "Eclipse"

            eclipse.newgame()
            sf.newgame()

            board = chess.Board()
            moves: list[str] = []
            wclock, bclock = base, base
            outcome: str | None = None
            while not board.is_game_over(claim_draw=True):
                engine = white if board.turn == chess.WHITE else black
                t0 = time.time()
                move = engine.go(moves, wclock, bclock, inc, inc)
                elapsed = time.time() - t0
                white_to_move = (board.turn == chess.WHITE)
                if white_to_move:
                    wclock = wclock - elapsed + inc
                else:
                    bclock = bclock - elapsed + inc
                if wclock <= 0.0 or bclock <= 0.0:
                    # Whoever just moved blew their clock.
                    outcome = "0-1" if white_to_move else "1-0"
                    if move is not None:
                        board.push_uci(move)
                        moves.append(move)
                    break
                if move is None:
                    # Engine resigned or has no legal move - let board.result
                    # decide (it will handle checkmate/stalemate correctly).
                    break
                board.push_uci(move)
                moves.append(move)
            if outcome is None:
                outcome = board.result(claim_draw=True)

            # Score from Eclipse's side
            if outcome == "1-0":
                if eclipse_is_white: wins += 1
                else: losses += 1
            elif outcome == "0-1":
                if eclipse_is_white: losses += 1
                else: wins += 1
            else:
                draws += 1

            # PGN
            game = chess.pgn.Game.from_board(board)
            game.headers["White"] = white_name
            game.headers["Black"] = black_name
            game.headers["Result"] = outcome
            game.headers["TimeControl"] = f"{int(base)}+{inc:g}"
            pgn_fh.write(str(game) + "\n\n")
            pgn_fh.flush()

            n = wins + draws + losses
            score = (wins + 0.5 * draws) / n
            elo, ci = elo_from_score(score, n)
            elapsed_min = (time.time() - t_start) / 60.0
            print(f"[{g+1:>3}/{args.games}] {outcome}  "
                  f"W:{wins} D:{draws} L:{losses}  "
                  f"score={score:.3f}  Elo={elo:+.0f} ±{ci:.0f}  "
                  f"t={elapsed_min:.1f}min", flush=True)
    finally:
        pgn_fh.close()
        eclipse.quit()
        sf.quit()
        if transcript:
            transcript.close()

    n = wins + draws + losses
    if n == 0:
        print("no games played", file=sys.stderr)
        return 1
    score = (wins + 0.5 * draws) / n
    elo, ci = elo_from_score(score, n)
    print()
    print(f"Final: {wins}W {draws}D {losses}L  "
          f"score={score:.3f}  Elo diff vs SF-{args.opponent_elo} = "
          f"{elo:+.0f} ± {ci:.0f}  (95% CI, n={n})")
    print(f"PGN: {pgn_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
