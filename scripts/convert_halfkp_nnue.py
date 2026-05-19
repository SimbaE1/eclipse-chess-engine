#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
HalfKAv2-1024x2-128-32 weight packer for Eclipse's .nnue binary format.

This script is the bridge between PyTorch training and the C++ inference path.
It has two modes:

  init        - emit a small-magnitude initialized net for testing the loader
                and forward-pass infrastructure end-to-end without needing real
                trained weights yet. Output is statistically zero-mean so eval
                values will hover near 0cp.

  from-torch  - convert a trained PyTorch state_dict (.pt) into Eclipse's
                quantized binary format. Use this once you've trained a
                HalfKAv2 model with a layout matching the C++ side:
                    ft.weight        shape [1024, 45056]  (PyTorch convention)
                    ft.bias          shape [1024]
                    l1.weight        shape [128, 2048]
                    l1.bias          shape [128]
                    l2.weight        shape [32, 128]
                    l2.bias          shape [32]
                    l3.weight        shape [1, 32]
                    l3.bias          shape [1]
                Real-valued floats, no quantization applied - this script
                handles all quantization.

The binary file format is documented at the top of src/nnue.cpp. Anything that
differs here is a bug; keep them in sync.
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# These must mirror the constants in src/nnue.hpp. If you bump the C++ side,
# bump here too or the loader will reject the file. Magic was bumped when the
# feature set switched from HalfKP (40960 features, 5 piece types per cell)
# to HalfKAv2 (45056 features, 11 piece-type slots including opp king).
MAGIC               = 0xECCC0003
VERSION             = 1
FT_IN_FEATURES      = 45056          # HalfKAv2: 64 * 64 * 11
FT_OUT              = 1024           # keep in sync with kFtOutSize in src/nnue.hpp
L1_IN               = 2 * FT_OUT     # 2048 (concat of both perspectives)
L1_OUT              = 128
L2_OUT              = 32
L3_OUT              = 1

# Quantization scales - must match the C++ side exactly.
FT_QUANT            = 127            # FT activation scale (kFtQuant)
WEIGHT_SCALE        = 64             # int8 weight scale (kWeightScale)
WEIGHT_SHIFT        = 6              # log2(WEIGHT_SCALE)


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _saturating_int(arr: np.ndarray, lo: int, hi: int, dtype: np.dtype, name: str) -> np.ndarray:
    """Round, clip into [lo, hi], cast. Warns on saturation so silent
    information-loss can't hide during a long training run."""
    rounded = np.round(arr).astype(np.int64)
    saturated = ((rounded < lo) | (rounded > hi)).sum()
    if saturated > 0:
        pct = 100.0 * saturated / rounded.size
        print(f"  WARNING: {name}: {saturated}/{rounded.size} weights saturated "
              f"({pct:.2f}%). Loss of precision; consider reducing weight magnitudes "
              f"or tightening regularization in training.", file=sys.stderr)
    return np.clip(rounded, lo, hi).astype(dtype)


def quantize_ft(weight_fp: np.ndarray, bias_fp: np.ndarray):
    """Feature transformer: weight [FT_OUT, FT_IN_FEATURES] / bias [FT_OUT].

    The accumulator stores values that, after clipped-ReLU at FT_QUANT, become
    "fixed-point with denominator FT_QUANT". So we encode `real_value * FT_QUANT`
    directly into the int16 accumulator.

    Returns:
        ft_w_packed: int16 [FT_IN_FEATURES * FT_OUT], row-major [feature, hidden]
        ft_b_packed: int16 [FT_OUT]
    """
    assert weight_fp.shape == (FT_OUT, FT_IN_FEATURES), \
        f"FT weight shape {weight_fp.shape} != ({FT_OUT}, {FT_IN_FEATURES})"
    assert bias_fp.shape == (FT_OUT,), f"FT bias shape {bias_fp.shape} != ({FT_OUT},)"

    # Transpose to [feature, hidden] for SIMD-friendly column reads on the C++ side.
    ft_w_real = weight_fp.T  # [FT_IN_FEATURES, FT_OUT]
    ft_w_q = _saturating_int(ft_w_real * FT_QUANT, -32768, 32767, np.int16, "ft_w")
    ft_b_q = _saturating_int(bias_fp     * FT_QUANT, -32768, 32767, np.int16, "ft_b")
    return ft_w_q.reshape(-1), ft_b_q


def quantize_int8_layer(weight_fp: np.ndarray, bias_fp: np.ndarray, name: str):
    """Generic int8-weight / int32-bias quantization for L1/L2/L3.

    Each layer takes uint8 input (range [0, FT_QUANT]) and produces int32 sums
    that are post-shifted by WEIGHT_SHIFT inside the C++ inference path:

        out = (bias + sum(w_int8 * in_uint8)) >> WEIGHT_SHIFT

    To keep this consistent with the real arithmetic
        out_real = (bias_real + sum(w_real * in_real))
    we need:
        w_int8     = round(w_real * WEIGHT_SCALE)
        in_uint8   = round(in_real * FT_QUANT)              (already enforced upstream)
        bias_int32 = round(bias_real * WEIGHT_SCALE * FT_QUANT)

    so the multiplied product carries `WEIGHT_SCALE * FT_QUANT` of total scale,
    matching the bias, and the post-shift by log2(WEIGHT_SCALE) leaves
    `FT_QUANT * out_real` - i.e. the same fixed-point form expected by the next
    layer.
    """
    w_q = _saturating_int(weight_fp * WEIGHT_SCALE, -128, 127, np.int8, f"{name} w")
    b_q = _saturating_int(bias_fp   * WEIGHT_SCALE * FT_QUANT, -(2**31), 2**31 - 1,
                          np.int32, f"{name} b")
    return w_q.reshape(-1), b_q


# ---------------------------------------------------------------------------
# Binary writer
# ---------------------------------------------------------------------------

def write_nnue_file(path: Path, *,
                    ft_w_fp: np.ndarray, ft_b_fp: np.ndarray,
                    l1_w_fp: np.ndarray, l1_b_fp: np.ndarray,
                    l2_w_fp: np.ndarray, l2_b_fp: np.ndarray,
                    l3_w_fp: np.ndarray, l3_b_fp: np.ndarray,
                    output_cp_per_unit: float):
    """Quantize and serialize. Shapes are validated up-front so we never
    write a partial file."""
    assert l1_w_fp.shape == (L1_OUT, L1_IN), f"l1_w {l1_w_fp.shape}"
    assert l1_b_fp.shape == (L1_OUT,),       f"l1_b {l1_b_fp.shape}"
    assert l2_w_fp.shape == (L2_OUT, L1_OUT), f"l2_w {l2_w_fp.shape}"
    assert l2_b_fp.shape == (L2_OUT,),        f"l2_b {l2_b_fp.shape}"
    assert l3_w_fp.shape == (L3_OUT, L2_OUT), f"l3_w {l3_w_fp.shape}"
    assert l3_b_fp.shape == (L3_OUT,),        f"l3_b {l3_b_fp.shape}"

    ft_w_q, ft_b_q = quantize_ft(ft_w_fp, ft_b_fp)
    l1_w_q, l1_b_q = quantize_int8_layer(l1_w_fp, l1_b_fp, "l1")
    l2_w_q, l2_b_q = quantize_int8_layer(l2_w_fp, l2_b_fp, "l2")
    l3_w_q, l3_b_q = quantize_int8_layer(l3_w_fp, l3_b_fp, "l3")

    with path.open("wb") as f:
        # Header (7 uint32 + 1 float).
        f.write(struct.pack("<IIIIIIIf",
                            MAGIC, VERSION,
                            FT_IN_FEATURES, FT_OUT,
                            L1_OUT, L2_OUT, L3_OUT,
                            float(output_cp_per_unit)))
        # Weights, in the exact order the C++ loader reads.
        ft_b_q.tofile(f)
        ft_w_q.tofile(f)
        l1_b_q.tofile(f)
        l1_w_q.tofile(f)
        l2_b_q.tofile(f)
        l2_w_q.tofile(f)
        l3_b_q.tofile(f)
        l3_w_q.tofile(f)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({size_mb:.2f} MB, output_cp/unit={output_cp_per_unit})")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def make_init_net(seed: int = 0):
    """Tiny-magnitude random init. All values are statistically zero-mean so
    the resulting eval hovers near 0 cp regardless of position. Useful for
    smoke-testing the loader + forward pass without any training."""
    rng = np.random.default_rng(seed)
    # std=1/sqrt(fan_in) is the usual Xavier-ish choice; scaled down so the
    # accumulator never saturates int16 even with 32 active features.
    ft_w = rng.normal(0.0, 0.01, size=(FT_OUT, FT_IN_FEATURES)).astype(np.float32)
    ft_b = np.zeros(FT_OUT, dtype=np.float32)
    l1_w = rng.normal(0.0, 0.05, size=(L1_OUT, L1_IN)).astype(np.float32)
    l1_b = np.zeros(L1_OUT, dtype=np.float32)
    l2_w = rng.normal(0.0, 0.1, size=(L2_OUT, L1_OUT)).astype(np.float32)
    l2_b = np.zeros(L2_OUT, dtype=np.float32)
    l3_w = rng.normal(0.0, 0.1, size=(L3_OUT, L2_OUT)).astype(np.float32)
    l3_b = np.zeros(L3_OUT, dtype=np.float32)
    return dict(ft_w_fp=ft_w, ft_b_fp=ft_b,
                l1_w_fp=l1_w, l1_b_fp=l1_b,
                l2_w_fp=l2_w, l2_b_fp=l2_b,
                l3_w_fp=l3_w, l3_b_fp=l3_b)


def make_from_torch(state_dict_path: Path):
    """Load PyTorch state_dict and remap keys. Expects float32 tensors with
    the layer names listed in the module docstring."""
    try:
        import torch  # local import so `init` mode works without torch installed
    except ImportError:
        print("error: from-torch requires PyTorch. pip install torch", file=sys.stderr)
        sys.exit(1)

    sd = torch.load(state_dict_path, map_location="cpu")
    # Allow nested keys like "model.ft.weight" by stripping a single leading
    # prefix - matches how PyTorch Lightning / nn.Sequential typically save.
    if any(k.startswith("model.") for k in sd):
        sd = {k.removeprefix("model."): v for k, v in sd.items()}

    def get(name):
        if name not in sd:
            raise KeyError(f"missing tensor '{name}' in state_dict. "
                           f"available keys: {sorted(sd.keys())}")
        return sd[name].detach().cpu().numpy().astype(np.float32)

    return dict(
        ft_w_fp=get("ft.weight"), ft_b_fp=get("ft.bias"),
        l1_w_fp=get("l1.weight"), l1_b_fp=get("l1.bias"),
        l2_w_fp=get("l2.weight"), l2_b_fp=get("l2.bias"),
        l3_w_fp=get("l3.weight"), l3_b_fp=get("l3.bias"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    p_init = sub.add_parser("init", help="emit a small-magnitude scaffold net")
    p_init.add_argument("--out", type=Path, required=True)
    p_init.add_argument("--seed", type=int, default=0)
    p_init.add_argument("--output-cp-per-unit", type=float, default=410.0,
                        help="centipawns per real-unit of L3 output. 410 matches "
                             "the tiny NNUE convention (sigmoid scale).")

    p_torch = sub.add_parser("from-torch", help="convert a PyTorch state_dict (.pt)")
    p_torch.add_argument("--state-dict", type=Path, required=True)
    p_torch.add_argument("--out", type=Path, required=True)
    p_torch.add_argument("--output-cp-per-unit", type=float, default=410.0)

    args = p.parse_args()

    if args.mode == "init":
        params = make_init_net(seed=args.seed)
    elif args.mode == "from-torch":
        params = make_from_torch(args.state_dict)
    else:
        p.error("unknown mode")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_nnue_file(args.out, output_cp_per_unit=args.output_cp_per_unit, **params)


if __name__ == "__main__":
    main()
