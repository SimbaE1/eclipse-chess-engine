#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
HalfKAv2-1024x2-512-128 training loop for Eclipse.

The model here is the float32 mirror of the inference path in src/nnue.cpp.
After training, save state_dict (.pt) and convert with:

    python scripts/convert_halfkav2_nnue.py from-torch \
        --state-dict path/to/weights.pt \
        --out data/eclipse.nnue

Training data format (line-oriented, one position per line). Two formats
are autodetected per line (mixed files are fine):

    <fen>;<W>;<D>;<L>       # observed game outcome (preferred for WDL head)
    <fen>;<score_cp>        # cp-derived legacy format (single-cp head only)

WDL labels come from real game results (use extract_lichess_wdl.py — labels
each position with the game's eventual W/D/L from STM's perspective). cp
labels are Stockfish-evaluated centipawns from STM's perspective.

The WDL→cp derivation that used to live in the cp path was broken in a
subtle way: `sigmoid(x) + sigmoid(-x) = 1` algebraically, so
`draw_prob = 1 - sigmoid(x) - sigmoid(-x)` was always 0. The trainer was
silently producing a W/L binary classifier with the D channel zeroed out.
The WDL-labels path uses cross-entropy on the true distribution and is
the right thing for the WDL head; the cp path remains for the legacy
single-cp head and emits a deprecation warning.

Lines starting with '#' are ignored.

Example label generation (bash):

    for fen in $(cat positions.txt); do
        score=$(echo -e "position fen $fen\\ngo depth 8" | stockfish | \\
                awk '/score cp/ {print $10; exit}')
        echo "$fen;$score"
    done > training.txt

This is a deliberately minimal trainer - the right tool for a PoC. For serious
training you'd want:
  - Stockfish binpack reader (see github.com/glinscott/nnue-pytorch)
  - WDL-only training (no cp labels)
  - Curriculum + king-bucket sampling
  - GPU support with proper sparse FT input

Hit those once the engine is generating wins.
"""

from __future__ import annotations  # keeps `int | None` syntax working on 3.9

import argparse
import gzip
import math
import random
import struct as _struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, IterableDataset
except ImportError:
    print("error: training requires PyTorch. pip install torch", file=sys.stderr)
    sys.exit(1)

# Binary format produced by preprocess_halfkav2.py (must stay in sync).
_BIN_MAGIC = b'HKAV2BIN'
_BIN_HSIZE = 16   # 8 magic + 8 N (int64 LE)

# Mirror src/nnue.hpp - any drift here = silently wrong inference.
#
# HalfKAv2 feature set (Stockfish-style):
#   For each (own_king_sq, piece_sq) there are 11 piece-type slots:
#     0..4 : own pieces (P, N, B, R, Q) -- own king is NOT encoded since it
#            always sits at the indexing king square (the feature would be
#            constant, redundant).
#     5..10: opponent pieces (P, N, B, R, Q, K) -- opp king IS a feature.
#   Total per perspective: 64 * 64 * 11 = 45056.
FT_IN_FEATURES = 45056
FT_OUT         = 1024                # keep in sync with kFtOutSize in src/nnue.hpp
L1_IN          = 2 * FT_OUT          # 2048 (concat of both perspectives)
L1_OUT         = 512
L2_OUT         = 128
L3_OUT         = 1                   # Value head: single logit (sigmoid = win prob)

# Piece-type slots within a (king_sq, piece_sq) cell. See FT_IN_FEATURES.
N_PIECE_SLOTS  = 11


# ---------------------------------------------------------------------------
# Position -> HalfKAv2 feature indices
# ---------------------------------------------------------------------------

# Standard chess piece-square layout matching src/types.hpp:
# A1=0..H1=7, A2=8..H8=63.

_PIECE_LETTER_TO_TYPE = {
    "p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6,
}
_FLIP_RANK = lambda s: s ^ 56  # noqa: E731


def parse_fen_board(fen_board: str):
    """Returns a list of (square_index, piece_color_0w_1b, piece_type_1..6)."""
    pieces = []
    sq = 56  # rank 8, file a
    for ch in fen_board:
        if ch == "/":
            sq -= 16   # back to file a, one rank down
            continue
        if ch.isdigit():
            sq += int(ch)
            continue
        is_white = ch.isupper()
        pt = _PIECE_LETTER_TO_TYPE[ch.lower()]
        pieces.append((sq, 0 if is_white else 1, pt))
        sq += 1
    return pieces


def feature_index(king_sq: int, piece_sq: int, pt: int, piece_is_ours: bool) -> int:
    """HalfKAv2 feature index. Must match feature_index() in src/nnue.cpp.

    Slot layout within each (king_sq, piece_sq) cell:
        ours:    0..4   (P=0, N=1, B=2, R=3, Q=4; own king is invalid)
        theirs:  5..10  (P=5, N=6, B=7, R=8, Q=9, K=10)

    Returns a value in [0, FT_IN_FEATURES). Raises ValueError if asked to
    encode the own king (caller must skip it -- own king is implicit at the
    indexing king square).
    """
    if piece_is_ours and pt == 6:
        raise ValueError("own king is not a HalfKAv2 feature; skip it in the caller")
    if piece_is_ours:
        slot = pt - 1                  # P=0 .. Q=4
    else:
        slot = 5 + (pt - 1)             # P=5 .. K=10
    return king_sq * (64 * N_PIECE_SLOTS) + piece_sq * N_PIECE_SLOTS + slot


def fen_to_features(fen: str):
    """Returns (us_indices, them_indices), each a list of HalfKAv2 feature
    indices for the corresponding perspective.

    The C++ uses 0=White / 1=Black perspectives. "us" here is the side to move.

    HalfKAv2 vs HalfKP: kings ARE features (specifically, the opponent's king
    from each perspective; the own king is skipped because it's pinned at the
    indexing king square).
    """
    parts = fen.split()
    board, stm = parts[0], parts[1]
    pieces = parse_fen_board(board)

    stm_color = 0 if stm == "w" else 1

    # Find kings.
    king_sq = [None, None]
    for sq, color, pt in pieces:
        if pt == 6:
            king_sq[color] = sq

    if king_sq[0] is None or king_sq[1] is None:
        raise ValueError(f"FEN missing a king: {fen}")

    indices = [[], []]  # [perspective] -> list of feature indices
    for persp in (0, 1):
        persp_color = persp
        ksq = king_sq[persp_color]
        if persp_color == 1:  # Black perspective: mirror
            ksq = _FLIP_RANK(ksq)
        for sq, color, pt in pieces:
            is_ours = (color == persp_color)
            # Skip own king: HalfKAv2 doesn't encode it (would be a constant).
            if is_ours and pt == 6:
                continue
            psq = sq if persp_color == 0 else _FLIP_RANK(sq)
            indices[persp].append(feature_index(ksq, psq, pt, is_ours))

    # Order [us, them] from side-to-move's perspective.
    if stm_color == 0:
        return indices[0], indices[1]
    return indices[1], indices[0]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# HalfKAv2 has at most 31 active features per perspective (32 starting pieces
# minus the perspective side's own king, which is implicit at the indexing
# king square; the opponent's king IS a feature). 32 is the round padding cap.
MAX_ACTIVE = 32


def _open_data(path: Path):
    """Open a training file, transparently decompressing .gz if needed."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _parse_line(line: str, cp_scale: float):
    """Parse one data line. Returns (us_arr, them_arr, win_prob float32) or None."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(";")
    if len(parts) == 4:
        try:
            fen     = parts[0].strip()
            win_prob = np.float32(float(parts[1]) + 0.5 * float(parts[2]))
        except ValueError:
            return None
    elif len(parts) == 2:
        try:
            fen      = parts[0].strip()
            win_prob = np.float32(1.0 / (1.0 + math.exp(-float(parts[1]) / cp_scale)))
        except ValueError:
            return None
    else:
        return None
    try:
        us, them = fen_to_features(fen)
    except (KeyError, IndexError, ValueError):
        return None
    us_arr   = np.full(MAX_ACTIVE, FT_IN_FEATURES, dtype=np.int32)
    them_arr = np.full(MAX_ACTIVE, FT_IN_FEATURES, dtype=np.int32)
    us_arr[:min(len(us),   MAX_ACTIVE)] = us[:MAX_ACTIVE]
    them_arr[:min(len(them), MAX_ACTIVE)] = them[:MAX_ACTIVE]
    return us_arr, them_arr, win_prob


class HalfKAv2Dataset(Dataset):
    """In-memory dataset. Use for validation sets (≤ a few million positions).
    Supports plain text and .gz files. Single-pass when max_lines is set."""

    def __init__(self, path: Path, max_lines: int | None = None, cp_scale: float = 410.0):
        n_alloc = max_lines if max_lines is not None else self._count_lines(path)
        if n_alloc == 0:
            raise RuntimeError(f"no training rows in {path}")

        self.us_idx   = np.full((n_alloc, MAX_ACTIVE), FT_IN_FEATURES, dtype=np.int32)
        self.them_idx = np.full((n_alloc, MAX_ACTIVE), FT_IN_FEATURES, dtype=np.int32)
        self.targets  = np.zeros((n_alloc, 1), dtype=np.float32)

        i = 0
        with _open_data(path) as f:
            for line in f:
                if i >= n_alloc:
                    break
                parsed = _parse_line(line, cp_scale)
                if parsed is None:
                    continue
                self.us_idx[i], self.them_idx[i], self.targets[i, 0] = parsed
                i += 1
                if i % 500_000 == 0:
                    print(f"  loaded {i} positions", file=sys.stderr)

        if i < n_alloc:
            self.us_idx   = self.us_idx[:i]
            self.them_idx = self.them_idx[:i]
            self.targets  = self.targets[:i]
        print(f"loaded {i} positions", file=sys.stderr)

    @staticmethod
    def _count_lines(path: Path) -> int:
        n = 0
        with _open_data(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    n += 1
        return n

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.us_idx[idx], self.them_idx[idx], self.targets[idx]


class HalfKAv2StreamDataset(IterableDataset):
    """Memory-bounded streaming dataset for files too large to fit in RAM.

    Reads the file sequentially and maintains a reservoir shuffle buffer so
    samples within each buffer-sized window are randomly shuffled.

    Memory per worker: shuffle_buffer * (32+32)*4 + shuffle_buffer*4 bytes.
    At shuffle_buffer=500_000 that is ~133 MB per worker.

    Multi-worker: each DataLoader worker takes every Nth data line
    (worker i reads lines i, i+N, i+2N, ...) so workers see non-overlapping,
    complementary subsets without inter-process coordination.
    """

    def __init__(self, path: Path, skip_lines: int = 0,
                 max_lines: int | None = None, cp_scale: float = 410.0,
                 shuffle_buffer: int = 500_000):
        self.path           = path
        self.skip_lines     = skip_lines
        self.max_lines      = max_lines
        self.cp_scale       = cp_scale
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id          if worker_info else 0
        n_workers   = worker_info.num_workers if worker_info else 1

        # Each worker owns 1/n_workers of max_lines so the total across all
        # workers equals max_lines, not max_lines * n_workers.
        if self.max_lines is not None:
            per_worker_max = (self.max_lines + n_workers - 1) // n_workers
        else:
            per_worker_max = None

        buf_us   = np.full((self.shuffle_buffer, MAX_ACTIVE), FT_IN_FEATURES, dtype=np.int32)
        buf_them = np.full((self.shuffle_buffer, MAX_ACTIVE), FT_IN_FEATURES, dtype=np.int32)
        buf_tgt  = np.zeros(self.shuffle_buffer, dtype=np.float32)
        buf_fill = 0
        emitted  = 0
        data_idx = 0  # counts parseable data lines seen after the skip

        with _open_data(self.path) as f:
            # Skip the val lines that HalfKAv2Dataset already holds in memory.
            if self.skip_lines > 0:
                skipped = 0
                for raw in f:
                    if raw.strip() and not raw.strip().startswith("#"):
                        skipped += 1
                        if skipped >= self.skip_lines:
                            break

            for raw in f:
                if per_worker_max is not None and emitted >= per_worker_max:
                    break
                stripped = raw.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                # Stride check BEFORE parsing: each worker only parses its own
                # lines (previously all workers parsed every line — 8x wasted CPU).
                if data_idx % n_workers != worker_id:
                    data_idx += 1
                    continue
                data_idx += 1
                parsed = _parse_line(raw, self.cp_scale)
                if parsed is None:
                    continue

                us_arr, them_arr, win_prob = parsed

                if buf_fill < self.shuffle_buffer:
                    buf_us[buf_fill]   = us_arr
                    buf_them[buf_fill] = them_arr
                    buf_tgt[buf_fill]  = win_prob
                    buf_fill += 1
                else:
                    # Reservoir replacement: evict a random slot and yield it.
                    idx = random.randrange(buf_fill)
                    yield (buf_us[idx].copy(),
                           buf_them[idx].copy(),
                           buf_tgt[idx:idx + 1].copy())
                    buf_us[idx]   = us_arr
                    buf_them[idx] = them_arr
                    buf_tgt[idx]  = win_prob
                    emitted += 1

        # Drain buffer in shuffled order.
        perm = list(range(buf_fill))
        random.shuffle(perm)
        for idx in perm:
            if per_worker_max is not None and emitted >= per_worker_max:
                break
            yield (buf_us[idx].copy(),
                   buf_them[idx].copy(),
                   buf_tgt[idx:idx + 1].copy())
            emitted += 1


def collate(batch):
    """Stack pre-padded rows into batch tensors. Cast int32 -> int64 here
    (PyTorch's nn.Embedding requires LongTensor); doing the cast at batch
    time keeps storage at int32 in the dataset."""
    us   = np.stack([b[0] for b in batch])
    them = np.stack([b[1] for b in batch])
    tgt  = np.stack([b[2] for b in batch])  # [B, 3]
    return (torch.from_numpy(us).long(),
            torch.from_numpy(them).long(),
            torch.from_numpy(tgt))


_BIN_RECORD_DTYPE = np.dtype([
    ('us',   np.uint16, (MAX_ACTIVE,)),
    ('them', np.uint16, (MAX_ACTIVE,)),
    ('tgt',  np.float16),
])


class HalfKAv2BinaryDataset(Dataset):
    """Memory-mapped dataset produced by preprocess_halfkav2.py.

    Supports random access with zero FEN parsing — workers do array indexing
    only, so DataLoader(shuffle=True) replaces the reservoir buffer entirely.
    """

    def __init__(self, path: Path, start: int = 0, end: int | None = None):
        with open(path, 'rb') as f:
            if f.read(8) != _BIN_MAGIC:
                raise ValueError(f'{path} is not a HKAV2BIN file')
            n_total = _struct.unpack('<q', f.read(8))[0]
        self._mm  = np.memmap(path, dtype=_BIN_RECORD_DTYPE, mode='r',
                              offset=_BIN_HSIZE, shape=(n_total,))
        self.start = start
        self.end   = n_total if end is None else min(end, n_total)

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        r = self._mm[self.start + idx]
        return (r['us'].astype(np.int64),
                r['them'].astype(np.int64),
                np.array([float(r['tgt'])], dtype=np.float32))


# ---------------------------------------------------------------------------
# Model - float mirror of the C++ inference path
# ---------------------------------------------------------------------------

class ClippedReLU(nn.Module):
    """Activations are clamped into [0, 1] in float training and [0, FT_QUANT]
    after quantization - same shape, different denominator."""
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class HalfKAv2Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Padded feature index FT_IN_FEATURES sits at the end of the embedding
        # table with a zero row (we'll zero it after init and never train it)
        # so padding contributes nothing.
        self.ft = nn.Embedding(FT_IN_FEATURES + 1, FT_OUT, padding_idx=FT_IN_FEATURES)
        nn.init.uniform_(self.ft.weight, -0.01, 0.01)
        with torch.no_grad():
            self.ft.weight[FT_IN_FEATURES].zero_()
        self.ft_bias = nn.Parameter(torch.zeros(FT_OUT))

        self.l1 = nn.Linear(L1_IN, L1_OUT)
        self.l2 = nn.Linear(L1_OUT, L2_OUT)
        self.l3 = nn.Linear(L2_OUT, L3_OUT)
        self.act = ClippedReLU()

    def forward(self, us_idx, them_idx):
        # Sum-pool feature embeddings -> per-perspective accumulator.
        # mean=False to match the linear-add accumulator on the C++ side.
        us   = self.ft(us_idx).sum(dim=1) + self.ft_bias   # [B, FT_OUT]
        them = self.ft(them_idx).sum(dim=1) + self.ft_bias # [B, FT_OUT]

        x = torch.cat([self.act(us), self.act(them)], dim=1)  # [B, L1_IN]
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.l3(x)
        return x  # [B, 1]

    def export_state_dict_for_quantization(self):
        """Repack the embedding into a plain weight matrix so convert_halfkav2_nnue.py
        can read it as a regular Linear layer.

        Embedding stores weights as [num_embeddings, embedding_dim] and our
        convention is `ft.weight` shape [FT_OUT, FT_IN_FEATURES] (PyTorch
        Linear-style: out, in). So transpose and drop the padding row."""
        sd = {}
        ft_weight_full = self.ft.weight.detach()          # [FT_IN+1, FT_OUT]
        sd["ft.weight"] = ft_weight_full[:FT_IN_FEATURES].T.contiguous()  # [FT_OUT, FT_IN]
        sd["ft.bias"]   = self.ft_bias.detach()
        sd["l1.weight"] = self.l1.weight.detach()
        sd["l1.bias"]   = self.l1.bias.detach()
        sd["l2.weight"] = self.l2.weight.detach()
        sd["l2.bias"]   = self.l2.bias.detach()
        sd["l3.weight"] = self.l3.weight.detach()
        sd["l3.bias"]   = self.l3.bias.detach()
        return sd

    def load_exported_state_dict(self, sd: dict) -> None:
        """Inverse of export_state_dict_for_quantization: restores a checkpoint
        saved in the exported (Linear-style, padding row dropped) layout into
        this module's native nn.Embedding layout, for resuming training."""
        with torch.no_grad():
            self.ft.weight[:FT_IN_FEATURES].copy_(sd["ft.weight"].T)
            self.ft.weight[FT_IN_FEATURES].zero_()
            self.ft_bias.copy_(sd["ft.bias"])
            self.l1.weight.copy_(sd["l1.weight"]); self.l1.bias.copy_(sd["l1.bias"])
            self.l2.weight.copy_(sd["l2.weight"]); self.l2.bias.copy_(sd["l2.bias"])
            self.l3.weight.copy_(sd["l3.weight"]); self.l3.bias.copy_(sd["l3.bias"])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def pick_device(requested: str) -> torch.device:
    """Resolve --device. "auto" prefers cuda > mps > cpu. Explicit choices are
    honored but we validate the backend is actually available so the user gets
    a clear message instead of a cryptic crash deep in a kernel."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None \
                and torch.backends.mps.is_available() \
                and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("error: --device cuda but CUDA is not available")
    if requested == "mps":
        mps_mod = getattr(torch.backends, "mps", None)
        if mps_mod is None or not mps_mod.is_available() or not mps_mod.is_built():
            raise SystemExit("error: --device mps but MPS backend is not built/available. "
                             "Need PyTorch with MPS (Apple Silicon Mac, torch >= 1.12).")
    return torch.device(requested)


def _is_binary(path: Path) -> bool:
    try:
        with open(path, 'rb') as f:
            return f.read(8) == _BIN_MAGIC
    except OSError:
        return False


def train(args):
    device = pick_device(args.device)
    print(f"training on {device}")
    if device.type == "cpu" and args.threads:
        torch.set_num_threads(args.threads)
        print(f"cpu intra-op threads: {args.threads}")

    out_pt = Path(args.out)
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    use_binary = _is_binary(args.data)
    if use_binary:
        print(f"binary dataset detected: {args.data}")
        val_ds   = HalfKAv2BinaryDataset(args.data, end=args.val_size)
        train_ds = HalfKAv2BinaryDataset(args.data, start=len(val_ds))
        n_val    = len(val_ds)
        print(f"val: {n_val:,}  train: {len(train_ds):,} positions (random-access)")
        shuffle_train = True
    else:
        # Val: load the first val_size lines into memory once (deterministic).
        print(f"loading {args.val_size:,} val positions into memory ...")
        val_ds = HalfKAv2Dataset(args.data, max_lines=args.val_size, cp_scale=args.cp_scale)
        n_val  = len(val_ds)

        # Train: stream remaining positions with a reservoir shuffle buffer.
        # Per-epoch dataset is (re)built in make_epoch_train_loader below so
        # each epoch advances to a new slice of the file when --max-lines is set.
        suffix = f" (max {args.max_lines:,}/epoch)" if args.max_lines else ""
        print(f"val: {n_val:,} positions (in memory)  train: streaming{suffix}")
        shuffle_train = False

    nw = args.num_workers

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            collate_fn=collate,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(nw > 0),
            prefetch_factor=(4 if nw > 0 else None),
        )

    # For the streaming path with --max-lines set, each epoch must read a
    # NEW slice of the file, not the same skip_lines..skip_lines+max_lines
    # window every time (StreamDataset.__iter__ re-opens the file from
    # scratch on every DataLoader iteration, so without advancing skip_lines
    # per epoch, every "epoch" would silently retrain on the same chunk).
    def make_epoch_train_loader(epoch: int):
        if use_binary:
            return make_loader(train_ds, shuffle_train)
        chunk = args.max_lines or 0
        ds = HalfKAv2StreamDataset(
            args.data,
            skip_lines=n_val + epoch * chunk,
            max_lines=args.max_lines,
            cp_scale=args.cp_scale,
            shuffle_buffer=args.shuffle_buffer,
        )
        return make_loader(ds, shuffle_train)

    loader = make_epoch_train_loader(0)
    val_nw = nw // 2 if nw > 0 else 0
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=val_nw,
        collate_fn=collate,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(val_nw > 0),
        prefetch_factor=(4 if val_nw > 0 else None),
    )

    net = HalfKAv2Net().to(device)
    if args.resume:
        print(f"resuming weights from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if "ft_bias" in ckpt:
            net.load_state_dict(ckpt)
        else:
            net.load_exported_state_dict(ckpt)
    # Weight decay (L2 regularization) helps prevent overfitting in the larger network.
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine annealing schedule with a linear warmup. Warmup helps when
    # using larger batch sizes (16k+) on a fresh init — Adam's first
    # moment estimates are very noisy in the first ~500 steps, and a
    # full LR there can push the embedding into a degenerate region.
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    warmup_steps = args.warmup_steps

    # Random baseline: BCE(sigmoid(0), 0.5) = log(2) ≈ 0.693
    print(f"random baseline BCE loss: {0.6931:.4f}")

    step = 0
    for epoch in range(args.epochs):
        if epoch > 0:
            loader = make_epoch_train_loader(epoch)
        net.train()
        epoch_loss = 0.0
        nbatch = 0
        for us, them, target in loader:
            us, them, target = us.to(device), them.to(device), target.to(device)

            # Linear warmup for the first warmup_steps. After warmup the
            # cosine schedule takes over (post-epoch step).
            if step < warmup_steps:
                lr_scale = (step + 1) / max(1, warmup_steps)
                for g in opt.param_groups:
                    g["lr"] = args.lr * lr_scale

            pred_logits = net(us, them)  # [B, 1]
            loss = F.binary_cross_entropy_with_logits(pred_logits.squeeze(1), target.squeeze(1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Keep the padding row pinned to zero so it never moves under us.
            with torch.no_grad():
                net.ft.weight[FT_IN_FEATURES].zero_()

            epoch_loss += loss.item()
            nbatch += 1
            step += 1
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step} loss {loss.item():.5f}")

        # Validation: BCE loss + MAE in centipawns (logit * cp_scale ≈ centipawns).
        net.eval()
        val_loss_sum = 0.0
        val_mae_sum  = 0.0
        val_count    = 0
        with torch.no_grad():
            for us, them, target in val_loader:
                us, them, target = us.to(device), them.to(device), target.to(device)
                logits = net(us, them)
                val_loss_sum += F.binary_cross_entropy_with_logits(
                    logits.squeeze(1), target.squeeze(1)).item() * target.shape[0]
                pred_cp = logits.squeeze(1) * args.cp_scale
                tgt_cp  = torch.special.logit(target.squeeze(1).clamp(1e-6, 1.0 - 1e-6)) * args.cp_scale
                val_mae_sum  += (pred_cp - tgt_cp).abs().sum().item()
                val_count    += target.shape[0]
        val_loss = val_loss_sum / max(1, val_count)
        val_mae  = val_mae_sum  / max(1, val_count)

        cosine.step()
        print(f"epoch {epoch} train_loss {epoch_loss / max(nbatch, 1):.5f} "
              f"val_loss {val_loss:.5f} val_mae_cp {val_mae:.1f} "
              f"(lr next: {cosine.get_last_lr()[0]:.6f})")

        ckpt = out_pt.with_name(out_pt.stem + f"_epoch{epoch}" + out_pt.suffix)
        torch.save(net.export_state_dict_for_quantization(), ckpt)
        print(f"saved {ckpt}")

    torch.save(net.export_state_dict_for_quantization(), out_pt)
    print(f"saved {out_pt}")
    print("now run: python scripts/convert_halfkav2_nnue.py from-torch "
          f"--state-dict {out_pt} --out data/eclipse.nnue "
          f"--output-cp-per-unit {args.cp_scale}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True,
                   help="text file with '<fen>;<W>;<D>;<L>' (preferred) or "
                        "'<fen>;<score_cp>' (legacy) lines")
    p.add_argument("--out", type=Path, default=Path("data/halfkav2.pt"),
                   help="PyTorch state_dict output")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-size", type=int, default=200_000,
                   help="number of lines read into memory for the validation set. "
                        "These are the first N lines of the file; the rest streams.")
    p.add_argument("--shuffle-buffer", type=int, default=500_000,
                   help="reservoir shuffle buffer size per DataLoader worker. "
                        "Larger = better shuffle, more RAM. 500k ≈ 133 MB/worker.")
    p.add_argument("--warmup-steps", type=int, default=500,
                   help="linear LR warmup for the first N steps (mitigates "
                        "Adam's noisy first-moment estimates on large batches)")
    p.add_argument("--max-lines", type=int, default=None,
                   help="cap on training positions per epoch. Without this the "
                        "full file is streamed each epoch.")
    p.add_argument("--cp-scale", type=float, default=410.0,
                   help="sigmoid scale; also written as output_cp_per_unit "
                        "in the .nnue header. Keep the two consistent.")
    p.add_argument("--resume", type=Path, default=None,
                   help="path to a raw float32 state_dict (.pt) to load weights from "
                        "before training (e.g. a checkpoint saved by a previous run). "
                        "Optimizer/scheduler state is not restored — Adam moments "
                        "restart fresh.")
    p.add_argument("--threads", type=int, default=None,
                   help="CPU intra-op thread count (torch.set_num_threads); ignored "
                        "on mps/cuda. Leave unset to use torch's default.")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                   help="compute device. auto = prefer cuda > mps > cpu. "
                        "On Apple Silicon, 'mps' uses Metal; if you see NaNs "
                        "or hangs in embedding ops, fall back to cpu.")
    train(p.parse_args())


if __name__ == "__main__":
    main()
