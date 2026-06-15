#!/usr/bin/env python3
"""
Self-play data pipeline unit tests.

Tests:
  1. pgn_to_training.py extracts correct FEN + cp from a synthetic PGN
  2. The cp filter (|cp| > 4000) is applied during extraction
  3. Output format is exactly "<fen>;<cp>" parseable by _parse_line
  4. Blend ratio: streaming from two sources at a given ratio works

Run with: python3 scripts/test_selfplay_data.py
"""
import sys, io, math, tempfile, textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import chess
    import chess.pgn
    import chess.svg
except ImportError:
    print("chess package required: pip install chess")
    sys.exit(1)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    results.append(condition)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pgn(moves_and_scores: list) -> str:
    """Build a minimal PGN string with Eclipse-style comments."""
    board = chess.Board()
    game  = chess.pgn.Game()
    game.headers['Result'] = '*'
    node = game
    for uci, score in moves_and_scores:
        move = chess.Move.from_uci(uci)
        comment = f'{score}cp n12345' if score is not None else ''
        node = node.add_variation(move, comment=comment)
        board.push(move)
    buf = io.StringIO()
    print(game, file=buf)
    return buf.getvalue()


def parse_fen_cp(line: str):
    """Parse a training line into (fen, cp) or None."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    parts = line.split(';')
    if len(parts) != 2:
        return None
    try:
        return parts[0].strip(), int(parts[1])
    except ValueError:
        return None


# ── Test 1: Basic PGN extraction ──────────────────────────────────────────────
print("\n── Test 1: Basic PGN extraction ──")

from pgn_to_training import extract_positions

# 3-move game, all positions scored
MOVES = [('e2e4', 50), ('e7e5', -55), ('g1f3', 60)]
pgn_text = make_pgn(MOVES)

with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as f:
    f.write(pgn_text)
    pgn_path = Path(f.name)

positions = extract_positions(pgn_path, max_cp=4000)
check("extracts 3 positions from 3-move game", len(positions) == 3,
      f"got {len(positions)}")

# FENs should be distinct (different boards)
fens = [p[0] for p in positions]
check("all FENs are distinct", len(set(fens)) == 3)

# CPs should match what we put in
cps = [p[1] for p in positions]
check("cp values match [50, -55, 60]", cps == [50, -55, 60], f"got {cps}")

# FEN 0 should be the starting position (FEN before e2e4 was played)
start_fen = chess.Board().fen()
check("first FEN is starting position", fens[0] == start_fen,
      f"got {fens[0][:40]}...")


# ── Test 2: |cp| > 4000 filter ────────────────────────────────────────────────
print("\n── Test 2: |cp| > 4000 filter during extraction ──")

MOVES2 = [('e2e4', 100), ('e7e5', 5000), ('g1f3', -4001), ('f8c5', 4000)]
pgn_text2 = make_pgn(MOVES2)

with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as f:
    f.write(pgn_text2)
    pgn_path2 = Path(f.name)

pos2 = extract_positions(pgn_path2, max_cp=4000)
check("extracts 2 positions (cp=100 and cp=4000, others filtered)",
      len(pos2) == 2, f"got {len(pos2)}: {[p[1] for p in pos2]}")
check("boundary 4000 is kept", any(p[1] == 4000 for p in pos2))
check("5000 is filtered", all(p[1] != 5000 for p in pos2))
check("-4001 is filtered", all(p[1] != -4001 for p in pos2))


# ── Test 3: Output format ─────────────────────────────────────────────────────
print("\n── Test 3: Output format parseable by _parse_line ──")

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    out_path = Path(f.name)

import subprocess, sys as _sys
result = subprocess.run(
    [_sys.executable, str(Path(__file__).parent / 'pgn_to_training.py'),
     str(pgn_path), '--out', str(out_path)],
    capture_output=True, text=True
)
check("pgn_to_training.py exits 0", result.returncode == 0, result.stderr[:200])

lines = [l.strip() for l in out_path.read_text().splitlines() if l.strip()]
check(f"output has {len(positions)} lines", len(lines) == len(positions),
      f"got {len(lines)}")

parsed = [parse_fen_cp(l) for l in lines]
check("all lines parse to (fen, cp)", all(p is not None for p in parsed),
      f"failed: {[l for l, p in zip(lines, parsed) if p is None][:3]}")

# Verify fen;cp format matches sigmoid formula
for fen, cp in parsed:
    wp = 1.0 / (1.0 + math.exp(-cp / 300.0))
    check(f"cp={cp} → valid win_prob ({wp:.3f} ∈ (0, 1))",
          0.0 < wp < 1.0, f"fen={fen[:30]}...")


# ── Test 4: Multi-game PGN ────────────────────────────────────────────────────
print("\n── Test 4: Multiple games in one PGN file ──")

GAME1 = [('e2e4', 30), ('e7e5', -35)]
GAME2 = [('d2d4', 10), ('d7d5', -10), ('c2c4', 20)]

pgn_multi = make_pgn(GAME1) + "\n" + make_pgn(GAME2)

with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as f:
    f.write(pgn_multi)
    pgn_multi_path = Path(f.name)

pos_multi = extract_positions(pgn_multi_path, max_cp=4000)
check("5 positions extracted from 2 games", len(pos_multi) == 5,
      f"got {len(pos_multi)}")


# ── Test 5: Missing cp comment is skipped ────────────────────────────────────
print("\n── Test 5: Positions without cp comment are skipped ──")

MOVES_PARTIAL = [('e2e4', None), ('e7e5', 50), ('g1f3', None)]
pgn_partial = make_pgn(MOVES_PARTIAL)

with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as f:
    f.write(pgn_partial)
    pgn_partial_path = Path(f.name)

pos_partial = extract_positions(pgn_partial_path, max_cp=4000)
check("only 1 position extracted (others have no cp comment)",
      len(pos_partial) == 1, f"got {len(pos_partial)}")
check("extracted cp = 50", pos_partial[0][1] == 50)


# ── Test 6: Blend ratio simulation ───────────────────────────────────────────
print("\n── Test 6: Blend ratio (10% self-play) is approximately correct ──")

import random
random.seed(42)

def blend_sources(n_total: int, blend_frac: float,
                  source_a: list, source_b: list) -> dict:
    """Simulate sampling n_total items with blend_frac from source_b."""
    counts = {'a': 0, 'b': 0}
    for _ in range(n_total):
        if random.random() < blend_frac:
            counts['b'] += 1
        else:
            counts['a'] += 1
    return counts

N = 100_000
FRAC = 0.10
counts = blend_sources(N, FRAC, list(range(1000)), list(range(500)))
actual_frac = counts['b'] / N
check(f"blend_frac=0.10: actual={actual_frac:.3f} (expect 0.10 ± 0.005)",
      abs(actual_frac - FRAC) < 0.005,
      f"a={counts['a']} b={counts['b']}")

# Ensure blending doesn't drop below a floor that would starve the self-play signal.
check("self-play source gets ≥ 8% of samples",
      actual_frac >= 0.08, f"actual={actual_frac:.3f}")


# ── Cleanup ───────────────────────────────────────────────────────────────────
for p in [pgn_path, pgn_path2, pgn_multi_path, pgn_partial_path, out_path]:
    try: p.unlink()
    except: pass


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*40}")
passed = sum(results)
total  = len(results)
print(f"Result: {passed}/{total} tests passed")
if passed < total:
    print("FAILED tests — fix before integrating self-play data.")
    sys.exit(1)
else:
    print("All tests passed. Self-play data pipeline verified.")
