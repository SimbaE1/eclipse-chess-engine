#!/usr/bin/env python3
"""
CP scale and position-filtering unit tests.

Improvement 2:
  - Lower cp_scale: 410 → 300 (sharper sigmoid gradient in the interesting range).
  - Position filtering: skip positions with |cp| > 4000 (near-mate noise).

Run with: python3 scripts/test_cp_scale.py
"""
import sys, math
import numpy as np

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


def sigmoid(cp, scale):
    return 1.0 / (1.0 + math.exp(-cp / scale))

def sigmoid_gradient(cp, scale):
    """d(sigmoid)/d(cp) — how much the label changes per centipawn."""
    s = sigmoid(cp, scale)
    return s * (1 - s) / scale


# ── Test 1: Sigmoid sharpness comparison ─────────────────────────────────────
print("\n── Test 1: cp_scale=300 has steeper gradients than 410 ──")

for cp in [50, 100, 200, 400]:
    g300 = sigmoid_gradient(cp, 300)
    g410 = sigmoid_gradient(cp, 410)
    check(f"gradient at cp={cp:3d}: 300 > 410",
          g300 > g410,
          f"300-scale={g300:.5f}  410-scale={g410:.5f}  ratio={g300/g410:.2f}x")


# ── Test 2: Label range sanity ────────────────────────────────────────────────
print("\n── Test 2: Label values at key centipawn points ──")

EXPECTED = {
    410: dict(adv=(0.55, 0.68), win=(0.70, 0.85), deci=(0.88, 0.96)),
    300: dict(adv=(0.55, 0.70), win=(0.80, 0.95), deci=(0.95, 0.999)),
}
for scale, name in [(410, 'old'), (300, 'new')]:
    eq   = sigmoid(0,    scale)
    adv  = sigmoid(100,  scale)
    win  = sigmoid(500,  scale)
    deci = sigmoid(1000, scale)
    E = EXPECTED[scale]
    check(f"[{name}] equal position = 0.50",
          abs(eq - 0.5) < 1e-10, f"{eq:.4f}")
    check(f"[{name}] +100cp ∈ {E['adv']}",
          E['adv'][0] < adv < E['adv'][1], f"{adv:.4f}")
    check(f"[{name}] +500cp ∈ {E['win']}",
          E['win'][0] < win < E['win'][1], f"{win:.4f}")
    check(f"[{name}] +1000cp ∈ {E['deci']}",
          E['deci'][0] < deci < E['deci'][1], f"{deci:.4f}")


# ── Test 3: Near-zero gradient zone ──────────────────────────────────────────
print("\n── Test 3: Position filtering removes near-zero gradient positions ──")

# At |cp| > 4000, sigmoid is effectively 1 or 0 — gradient ≈ 0 for both scales.
# These positions add noise (label is near 0 or 1 regardless of eval accuracy)
# and waste gradient budget.
for cp in [3000, 4000, 5000, 8000]:
    s = sigmoid(cp, 300)
    g = sigmoid_gradient(cp, 300)
    check(f"|cp|={cp}: label near 1.0 (sigmoid={s:.5f})",
          s > 0.9999 if cp >= 4000 else s > 0.999,
          f"gradient={g:.2e}")

# Verify filter threshold: |cp| <= 4000 keeps the meaningful range.
CUT = 4000
kept = [cp for cp in range(0, 8001, 50) if cp <= CUT]
cut  = [cp for cp in range(0, 8001, 50) if cp >  CUT]
check(f"Filter keeps {len(kept)} cp values in [0,{CUT}]", len(kept) > 0)
check(f"Filter removes {len(cut)} cp values > {CUT}", len(cut) > 0)


# ── Test 4: Gradient mass comparison ─────────────────────────────────────────
print("\n── Test 4: Gradient mass in 0–500cp range (typical game positions) ──")

cps = np.arange(-500, 501, 1)
mass_300 = sum(sigmoid_gradient(cp, 300) for cp in cps)
mass_410 = sum(sigmoid_gradient(cp, 410) for cp in cps)
check("cp_scale=300 concentrates more gradient mass in [-500, 500]",
      mass_300 > mass_410,
      f"300={mass_300:.4f}  410={mass_410:.4f}  ratio={mass_300/mass_410:.3f}x")

# Sanity: total mass of sigmoid derivative integrates to 1.0 (it's a density)
# so mass in a window tells us fraction of gradient "budget" in that window.
cps_all = np.arange(-20000, 20001, 1)
total_300 = sum(sigmoid_gradient(cp, 300) for cp in cps_all)
total_410 = sum(sigmoid_gradient(cp, 410) for cp in cps_all)
frac_300 = mass_300 / total_300
frac_410 = mass_410 / total_410
check(f"cp_scale=300: {frac_300:.1%} of gradient in [-500,+500] (vs {frac_410:.1%} at 410)",
      frac_300 > frac_410,
      f"300={frac_300:.1%}  410={frac_410:.1%}")


# ── Test 5: _parse_line position filter ──────────────────────────────────────
print("\n── Test 5: _parse_line with cp_scale=300 and |cp|>4000 filter ──")

# Simulate the _parse_line filter logic without importing the full notebook deps.
def parse_line_new(line, cp_scale=300.0, max_cp=4000):
    """Mirrors the updated _parse_line with filtering."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    parts = line.split(';')
    if len(parts) == 2:
        try:
            fen = parts[0].strip()
            cp  = float(parts[1])
        except ValueError:
            return None
        if abs(cp) > max_cp:
            return None   # filter near-mate positions
        return 1.0 / (1.0 + math.exp(-cp / cp_scale))
    return None

# Normal positions should pass through.
check("cp=+100 passes filter", parse_line_new("fen;100") is not None)
check("cp=-200 passes filter", parse_line_new("fen;-200") is not None)
check("cp=0 passes filter",    parse_line_new("fen;0") is not None)
check("cp=+4000 passes filter (boundary)", parse_line_new("fen;4000") is not None)
check("cp=-4000 passes filter (boundary)", parse_line_new("fen;-4000") is not None)

# Near-mate positions should be filtered out.
check("cp=+4001 filtered", parse_line_new("fen;4001") is None)
check("cp=-4001 filtered", parse_line_new("fen;-4001") is None)
check("cp=+8000 filtered (deep mate search)", parse_line_new("fen;8000") is None)
check("cp=-9999 filtered", parse_line_new("fen;-9999") is None)

# Label values at boundary with new scale.
label_100  = parse_line_new("fen;100")
label_4000 = parse_line_new("fen;4000")
check(f"label at +100cp ≈ {sigmoid(100,300):.4f}",
      abs(label_100 - sigmoid(100, 300)) < 1e-10)
check(f"label at +4000cp ≈ {sigmoid(4000,300):.6f} (very near 1)",
      label_4000 > 0.9999)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*40}")
passed = sum(results)
total  = len(results)
print(f"Result: {passed}/{total} tests passed")
if passed < total:
    print("FAILED tests — fix before committing CP scale changes.")
    sys.exit(1)
else:
    print("All tests passed. CP scale=300 + position filter verified.")
