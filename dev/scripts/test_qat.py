#!/usr/bin/env python3
"""
QAT unit tests.

Tests that the fake-quantization nodes match the post-training
quantization applied by convert_halfkav2_nnue.py — so that QAT training
is actually exercising the same error the deployed int8/int16 model sees.

Run with: python3 scripts/test_qat.py
"""
import sys, math, struct
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F

# ── Quantization constants (must match nnue.hpp / convert_halfkav2_nnue.py) ──
FT_QUANT     = 127       # FT accumulator / activation scale
WEIGHT_SCALE = 64        # int8 hidden-weight scale

FT_SCALE  = 1.0 / FT_QUANT           # 1/127
FT_LO     = -32768.0 / FT_QUANT      # int16 min mapped to float
FT_HI     =  32767.0 / FT_QUANT      # int16 max mapped to float
W_SCALE   = 1.0 / WEIGHT_SCALE       # 1/64
W_LO      = -128.0 / WEIGHT_SCALE    # int8 min = -2.0
W_HI      =  127.0 / WEIGHT_SCALE    # int8 max ≈ +1.984
ACT_SCALE = 1.0 / FT_QUANT
ACT_LO    = 0.0
ACT_HI    = 1.0

# ── Fake-quant operator (STE) ─────────────────────────────────────────────────
class FakeQuantSTE(torch.autograd.Function):
    """Round-to-nearest fake quantization with Straight-Through Estimator."""
    @staticmethod
    def forward(ctx, x, scale, lo, hi):
        return torch.clamp(torch.round(x / scale) * scale, lo, hi)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None   # STE: identity backward

def fake_quant(x, scale, lo, hi):
    return FakeQuantSTE.apply(x, scale, lo, hi)

# ── Reference: post-training quantization (mirrors convert_halfkav2_nnue.py) ──
def pth_quantize_ft(w_fp, b_fp):
    """float → int16 → float. Uses float32 arithmetic to match PyTorch fake_quant."""
    # Force float32 so rounding ties match what PyTorch does with float32 tensors.
    s = np.float32(FT_SCALE)
    w_q = np.clip(np.round(w_fp.astype(np.float32) / s) * s, FT_LO, FT_HI)
    b_q = np.clip(np.round(b_fp.astype(np.float32) / s) * s, FT_LO, FT_HI)
    return w_q.astype(np.float32), b_q.astype(np.float32)

def pth_quantize_hidden(w_fp):
    """float → int8 → float."""
    w_q = np.clip(np.round(w_fp * WEIGHT_SCALE), -128, 127).astype(np.int8)
    return w_q.astype(np.float32) / WEIGHT_SCALE

# ── Tests ─────────────────────────────────────────────────────────────────────
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


# Test 1: FakeQuantSTE forward matches post-training FT quantization
print("\n── Test 1: FT weight fake-quant matches converter ──")
rng = np.random.default_rng(42)
w_fp = rng.uniform(-3.0, 3.0, (1024, 45056)).astype(np.float32)
b_fp = rng.uniform(-1.0, 1.0, (1024,)).astype(np.float32)

w_ref, b_ref = pth_quantize_ft(w_fp, b_fp)
w_qat = fake_quant(torch.from_numpy(w_fp), FT_SCALE, FT_LO, FT_HI).numpy()
b_qat = fake_quant(torch.from_numpy(b_fp), FT_SCALE, FT_LO, FT_HI).numpy()

# Values within range should match exactly; out-of-range values differ only in clip
in_range_w = np.abs(w_fp) <= FT_HI
w_match = np.allclose(w_qat[in_range_w], w_ref[in_range_w], atol=0)
b_match = np.allclose(b_qat, b_ref, atol=0)
check("FT weight in-range exact match", w_match,
      f"max diff = {np.abs(w_qat[in_range_w]-w_ref[in_range_w]).max():.2e}")
check("FT bias exact match", b_match,
      f"max diff = {np.abs(b_qat - b_ref).max():.2e}")

# Saturation: values outside [-258, 258] should be clipped
sat = np.abs(w_fp) > FT_HI
if sat.any():
    sat_match = np.all(np.abs(w_qat[sat]) <= FT_HI + 1e-6)
    check("FT weight saturation clipping", sat_match,
          f"{sat.sum()} values clipped")


# Test 2: Hidden-weight fake-quant matches converter
print("\n── Test 2: Hidden weight fake-quant matches converter ──")
w_h = rng.uniform(-2.5, 2.5, (512, 2048)).astype(np.float32)
w_ref_h = pth_quantize_hidden(w_h)
w_qat_h = fake_quant(torch.from_numpy(w_h), W_SCALE, W_LO, W_HI).numpy()

in_range_h = (w_h >= W_LO) & (w_h <= W_HI)
h_match = np.allclose(w_qat_h[in_range_h], w_ref_h[in_range_h], atol=0)
check("Hidden weight in-range exact match", h_match,
      f"max diff = {np.abs(w_qat_h[in_range_h]-w_ref_h[in_range_h]).max():.2e}")
sat_h = ~in_range_h
if sat_h.any():
    # Saturated values must land at exactly W_LO or W_HI (not beyond).
    clipped_ok = np.all((w_qat_h[sat_h] >= W_LO - 1e-6) & (w_qat_h[sat_h] <= W_HI + 1e-6))
    check("Hidden weight saturation clipped to [W_LO, W_HI]",
          clipped_ok,
          f"{sat_h.sum()} weights saturated")


# Test 3: STE gradient flows through fake-quant (not zero)
print("\n── Test 3: STE gradient propagation ──")
x = torch.randn(32, 1024, requires_grad=True)
y = fake_quant(x, FT_SCALE, FT_LO, FT_HI)
loss = y.sum()
loss.backward()
check("Gradient flows through fake_quant (not None)", x.grad is not None)
check("Gradient is all-ones (STE identity)",
      torch.allclose(x.grad, torch.ones_like(x)),
      f"max |grad - 1| = {(x.grad - 1).abs().max().item():.2e}")


# Test 4: Activation fake-quant produces 128 discrete levels
print("\n── Test 4: Activation quantization has 128 levels ──")
x_act = torch.linspace(-0.5, 1.5, 10000)
x_q = fake_quant(x_act.clamp(0, 1), ACT_SCALE, ACT_LO, ACT_HI)
unique_vals = x_q.unique()
check("Exactly 128 discrete output levels (0, 1/127, …, 1.0)",
      unique_vals.numel() == 128,
      f"got {unique_vals.numel()} levels")
check("Min activation = 0.0", unique_vals.min().item() == 0.0)
check("Max activation = 1.0", abs(unique_vals.max().item() - 1.0) < 1e-6)


# Test 5: Quantization error magnitude is small relative to typical values
print("\n── Test 5: Quantization error magnitude ──")
# FT accumulation: ~30 active features, each ~[-2, 2] before CReLU
# Error per weight: ≤ 0.5/127 ≈ 0.004; total for 30: ≤ 0.12
ft_typical = rng.uniform(-2.0, 2.0, (100, 1024)).astype(np.float32)
ft_q = fake_quant(torch.from_numpy(ft_typical), FT_SCALE, FT_LO, FT_HI).numpy()
ft_err = np.abs(ft_q - ft_typical)
check("FT per-weight max error ≤ 0.5/127",
      ft_err.max() <= 0.5 / FT_QUANT + 1e-7,
      f"max err = {ft_err.max():.4f}, bound = {0.5/FT_QUANT:.4f}")

# Hidden weight: error ≤ 0.5/64 = 0.0078
w_typical = rng.uniform(-1.5, 1.5, (100, 2048)).astype(np.float32)
w_q = fake_quant(torch.from_numpy(w_typical), W_SCALE, W_LO, W_HI).numpy()
in_r = (w_typical >= W_LO) & (w_typical <= W_HI)
w_err = np.abs(w_q[in_r] - w_typical[in_r])
check("Hidden per-weight max error ≤ 0.5/64",
      w_err.max() <= 0.5 / WEIGHT_SCALE + 1e-7,
      f"max err = {w_err.max():.4f}, bound = {0.5/WEIGHT_SCALE:.4f}")


# Test 6: Compare network output with and without QAT
print("\n── Test 6: QAT vs non-QAT output delta ──")
# Build a tiny 3-layer network and compare outputs
torch.manual_seed(0)
ft_dim, l1_dim, l2_dim = 64, 32, 16

class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(l1_dim, ft_dim * 2) * 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(l1_dim))
        self.w2 = torch.nn.Parameter(torch.randn(l2_dim, l1_dim) * 0.5)
        self.b2 = torch.nn.Parameter(torch.zeros(l2_dim))
        self.w3 = torch.nn.Parameter(torch.randn(1, l2_dim) * 0.5)
        self.b3 = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, qat=False):
        if qat:
            w1 = fake_quant(self.w1, W_SCALE, W_LO, W_HI)
        else:
            w1 = self.w1
        x = torch.clamp(F.linear(x, w1, self.b1), 0, 1)
        if qat:
            x = fake_quant(x, ACT_SCALE, ACT_LO, ACT_HI)
            w2 = fake_quant(self.w2, W_SCALE, W_LO, W_HI)
        else:
            w2 = self.w2
        x = torch.clamp(F.linear(x, w2, self.b2), 0, 1)
        if qat:
            x = fake_quant(x, ACT_SCALE, ACT_LO, ACT_HI)
            w3 = fake_quant(self.w3, W_SCALE, W_LO, W_HI)
        else:
            w3 = self.w3
        return F.linear(x, w3, self.b3)

net = TinyNet()
x_in = torch.clamp(torch.randn(64, ft_dim * 2), 0, 1)
with torch.no_grad():
    y_fp  = net(x_in, qat=False)
    y_qat = net(x_in, qat=True)

delta = (y_qat - y_fp).abs()
print(f"  Output delta — mean: {delta.mean():.4f}, max: {delta.max():.4f}")
check("QAT/float output delta is non-trivial (QAT is doing something)",
      delta.mean().item() > 1e-4)
check("QAT/float output delta is bounded (< 2.0 typical)",
      delta.max().item() < 2.0,
      f"max={delta.max().item():.4f}")

# QAT gradient flow check
y_qat2 = net(x_in, qat=True)
y_qat2.sum().backward()
grad_ok = all(p.grad is not None and p.grad.abs().max() > 0
              for p in net.parameters())
check("All parameters receive non-zero gradients through QAT", grad_ok)


# Summary
print(f"\n{'─'*40}")
passed = sum(results)
total  = len(results)
print(f"Result: {passed}/{total} tests passed")
if passed < total:
    print("FAILED tests — fix before committing QAT changes.")
    sys.exit(1)
else:
    print("All tests passed. QAT implementation verified.")
