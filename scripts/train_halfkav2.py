#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
HalfKAv2-1024x2-128-32 training loop for Eclipse.

The model here is the float32 mirror of the inference path in src/nnue.cpp.
After training, save state_dict (.pt) and convert with:

    python scripts/convert_halfkav2_nnue.py from-torch \
        --state-dict path/to/weights.pt \
        --out data/eclipse.nnue

Training data format (line-oriented, one position per line):

    <fen>;<score_cp>

where score_cp is the centipawn evaluation from the side-to-move's perspective,
e.g. produced by `stockfish go depth 8`. Lines starting with '#' are ignored.

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
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("error: training requires PyTorch. pip install torch", file=sys.stderr)
    sys.exit(1)

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
L1_OUT         = 256                 # Increased for more capacity
L2_OUT         = 64                  # Increased for more capacity
L3_OUT         = 3                   # WDL Head: [Win, Draw, Loss]

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


class HalfKAv2Dataset(Dataset):
    """Reads `<fen>;<score_cp>` lines into flat numpy arrays.

    Memory model: three contiguous arrays
        us_idx   : int32 [N, MAX_ACTIVE]   (padded with FT_IN_FEATURES sentinel)
        them_idx : int32 [N, MAX_ACTIVE]
        scores   : float32 [N]

    For 10M positions this is 10M * (32+32) * 4 + 10M * 4 = ~2.6 GB - flat,
    cache-friendly, and pickling-cheap if DataLoader workers are ever used.

    Versus the original per-row tuple/list storage, which spent ~80 bytes of
    Python object overhead per row, that's a 5-10x memory saving at scale.
    """

    def __init__(self, path: Path, max_lines: int | None = None):
        # Two-pass: count usable lines, then allocate flat arrays of the right
        # size and fill them. Avoids Python list growth + per-row object
        # overhead during construction.
        n = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                n += 1
                if max_lines is not None and n >= max_lines:
                    break
        if n == 0:
            raise RuntimeError(f"no training rows in {path}")

        self.us_idx   = np.full((n, MAX_ACTIVE), FT_IN_FEATURES, dtype=np.int32)
        self.them_idx = np.full((n, MAX_ACTIVE), FT_IN_FEATURES, dtype=np.int32)
        self.scores   = np.zeros(n, dtype=np.float32)

        i = 0
        skipped = 0
        with path.open() as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if i >= n:
                    break
                try:
                    fen, score_str = line.split(";")
                    score = float(score_str.strip())
                except ValueError:
                    skipped += 1
                    continue
                try:
                    us, them = fen_to_features(fen.strip())
                except (KeyError, IndexError, ValueError) as e:
                    if skipped < 5:
                        print(f"  skip line {ln}: feature error: {e}", file=sys.stderr)
                    skipped += 1
                    continue
                k_us   = min(len(us),   MAX_ACTIVE)
                k_them = min(len(them), MAX_ACTIVE)
                self.us_idx[i,   :k_us]   = us[:k_us]
                self.them_idx[i, :k_them] = them[:k_them]
                self.scores[i] = score
                i += 1
                if i % 500_000 == 0:
                    print(f"  loaded {i}/{n} positions", file=sys.stderr)

        # If we skipped some, the tail of the arrays is unused. Truncate so
        # __len__ is accurate and we don't train on zero-rows that mean
        # "completely empty board" (which would actually be a valid HalfKAv2
        # encoding - all padding).
        if i < n:
            self.us_idx   = self.us_idx[:i]
            self.them_idx = self.them_idx[:i]
            self.scores   = self.scores[:i]

        print(f"loaded {i} positions ({skipped} skipped)")

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        # Return numpy slices - cheap, no allocation. collate() stacks them.
        return self.us_idx[idx], self.them_idx[idx], self.scores[idx]


def collate(batch):
    """Stack pre-padded rows into batch tensors. Cast int32 -> int64 here
    (PyTorch's nn.Embedding requires LongTensor); doing the cast at batch
    time keeps storage at int32 in the dataset."""
    us   = np.stack([b[0] for b in batch])
    them = np.stack([b[1] for b in batch])
    scr  = np.array([b[2] for b in batch], dtype=np.float32)
    return (torch.from_numpy(us).long(),
            torch.from_numpy(them).long(),
            torch.from_numpy(scr))


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
        return x  # [B, 3]

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


def train(args):
    device = pick_device(args.device)
    print(f"training on {device}")

    ds = HalfKAv2Dataset(args.data, max_lines=args.max_lines)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate,
                        drop_last=True)

    net = HalfKAv2Net().to(device)
    # Weight decay (L2 regularization) helps prevent overfitting in the larger network.
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # Scheduler: Cosine Annealing to help the model converge more precisely.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Sigmoid scaling: matches the tiny NNUE trainer's convention.
    cp_scale = args.cp_scale

    step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        nbatch = 0
        for us, them, scores in loader:
            us, them, scores = us.to(device), them.to(device), scores.to(device)
            
            # WDL target derivation: Convert CP to soft Win/Draw/Loss probabilities.
            # This is a simplified sigmoid mapping.
            win_prob  = torch.sigmoid(scores / cp_scale)
            loss_prob = torch.sigmoid(-scores / cp_scale)
            draw_prob = 1.0 - win_prob - loss_prob
            draw_prob = torch.clamp(draw_prob, 0.0, 1.0)
            
            # Normalize to ensure sum=1
            s = win_prob + draw_prob + loss_prob
            target = torch.stack([win_prob/s, draw_prob/s, loss_prob/s], dim=1)

            # Prediction
            pred_logits = net(us, them)
            pred_probs  = F.softmax(pred_logits, dim=1)
            
            # Loss: KL Divergence or MSE on probabilities
            loss = F.mse_loss(pred_probs, target)

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
        
        scheduler.step()
        print(f"epoch {epoch} avg loss {epoch_loss / max(nbatch, 1):.5f} (lr: {scheduler.get_last_lr()[0]:.6f})")

    out_pt = Path(args.out)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.export_state_dict_for_quantization(), out_pt)
    print(f"saved {out_pt}")
    print("now run: python scripts/convert_halfkav2_nnue.py from-torch "
          f"--state-dict {out_pt} --out data/eclipse.nnue "
          f"--output-cp-per-unit {cp_scale}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True,
                   help="text file with '<fen>;<score_cp>' lines")
    p.add_argument("--out", type=Path, default=Path("data/halfkav2.pt"),
                   help="PyTorch state_dict output")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-lines", type=int, default=None,
                   help="cap on training rows (useful for quick sanity runs)")
    p.add_argument("--cp-scale", type=float, default=410.0,
                   help="sigmoid scale; also written as output_cp_per_unit "
                        "in the .nnue header. Keep the two consistent.")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                   help="compute device. auto = prefer cuda > mps > cpu. "
                        "On Apple Silicon, 'mps' uses Metal; if you see NaNs "
                        "or hangs in embedding ops, fall back to cpu.")
    train(p.parse_args())


if __name__ == "__main__":
    main()
