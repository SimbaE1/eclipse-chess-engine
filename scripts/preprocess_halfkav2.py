#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert a HalfKAv2 text training file to a fast binary format.

Run once before training. Training then uses memmap random-access instead of
FEN parsing, eliminating the CPU bottleneck that makes text-mode training slow.

Binary layout
  [0:8]    magic  b'HKAV2BIN'
  [8:16]   N      int64 little-endian  (number of valid positions)
  [16:]    N records × 130 bytes each:
             us_idx[32]   uint16  (HalfKAv2 feature indices, 0-45056)
             them_idx[32] uint16
             target       float16 (win probability)

Usage:
    python scripts/preprocess_halfkav2.py \\
        --data /path/to/training.txt \\
        --out  /path/to/training.bin \\
        [--max-records 140_000_000]   # 140M ≈ 18 GB, fits in Kaggle /working
        [--workers 4]
"""
from __future__ import annotations

import argparse
import gzip
import math
import multiprocessing as mp
import struct
import time
from pathlib import Path

import numpy as np

# ── must stay in sync with train_halfkav2.py / src/nnue.cpp ─────────────────
FT_IN_FEATURES = 45056
MAX_ACTIVE     = 32
N_PIECE_SLOTS  = 11
_PT            = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6}


def _parse_board(board: str):
    pieces, sq = [], 56
    for ch in board:
        if ch == '/':
            sq -= 16
            continue
        if ch.isdigit():
            sq += int(ch)
            continue
        pieces.append((sq, 0 if ch.isupper() else 1, _PT[ch.lower()]))
        sq += 1
    return pieces


def _feat(ks: int, ps: int, pt: int, ours: bool) -> int:
    if ours and pt == 6:
        raise ValueError
    slot = (pt - 1) if ours else (5 + pt - 1)
    return ks * (64 * N_PIECE_SLOTS) + ps * N_PIECE_SLOTS + slot


def _fen_to_features(fen: str):
    parts = fen.split()
    board, stm = parts[0], parts[1]
    pieces = _parse_board(board)
    sc = 0 if stm == 'w' else 1
    ks: list = [None, None]
    for sq, c, pt in pieces:
        if pt == 6:
            ks[c] = sq
    if ks[0] is None or ks[1] is None:
        raise ValueError('missing king')
    idx: list = [[], []]
    for p in (0, 1):
        ksq = ks[p] if p == 0 else ks[p] ^ 56
        for sq, c, pt in pieces:
            if c == p and pt == 6:
                continue
            idx[p].append(_feat(ksq, sq if p == 0 else sq ^ 56, pt, c == p))
    return (idx[0], idx[1]) if sc == 0 else (idx[1], idx[0])


def _parse_line(line: str, cp_scale: float, max_cp: float | None = None):
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    parts = line.split(';')
    if len(parts) == 4:
        try:
            fen = parts[0].strip()
            wp  = float(parts[1]) + 0.5 * float(parts[2])
        except ValueError:
            return None
    elif len(parts) == 2:
        try:
            fen = parts[0].strip()
            cp  = float(parts[1])
            if max_cp is not None and abs(cp) > max_cp:
                return None  # near-mate noise, ~0 gradient -- match notebook filter
            wp  = 1.0 / (1.0 + math.exp(-cp / cp_scale))
        except ValueError:
            return None
    else:
        return None
    try:
        us, them = _fen_to_features(fen)
    except (KeyError, IndexError, ValueError):
        return None
    ua = np.full(MAX_ACTIVE, FT_IN_FEATURES, dtype=np.uint16)
    ta = np.full(MAX_ACTIVE, FT_IN_FEATURES, dtype=np.uint16)
    ua[:min(len(us),   MAX_ACTIVE)] = us[:MAX_ACTIVE]
    ta[:min(len(them), MAX_ACTIVE)] = them[:MAX_ACTIVE]
    return ua, ta, np.float16(wp)
# ─────────────────────────────────────────────────────────────────────────────

MAGIC       = b'HKAV2BIN'
RECORD_SIZE = MAX_ACTIVE * 2 + MAX_ACTIVE * 2 + 2   # 130 bytes
HEADER_SIZE = 16                                      # 8 magic + 8 N (int64 LE)
RECORD_DTYPE = np.dtype([
    ('us',   np.uint16, (MAX_ACTIVE,)),
    ('them', np.uint16, (MAX_ACTIVE,)),
    ('tgt',  np.float16),
])
_BATCH = 50_000   # lines per worker task


def _process_batch(args: tuple) -> bytes:
    """Worker: parse a list of raw text lines, return packed binary bytes."""
    lines, cp_scale, max_cp = args
    recs = []
    for line in lines:
        r = _parse_line(line, cp_scale, max_cp)
        if r:
            recs.append(r)
    if not recs:
        return b''
    buf = np.empty(len(recs), dtype=RECORD_DTYPE)
    for i, (ua, ta, wp) in enumerate(recs):
        buf[i]['us']  = ua
        buf[i]['them'] = ta
        buf[i]['tgt'] = wp
    return buf.tobytes()


def _line_batches(path: Path, cp_scale: float, skip_records: int = 0,
                   max_cp: float | None = None, max_lines: int | None = None):
    """Yield (lines, cp_scale, max_cp) batches.

    Both `skip_records` and `max_lines` operate in INPUT-LINE space (data lines
    after stripping blanks/comments), NOT in written-record space. This makes
    chunks a clean, non-overlapping partition of the file even when the max_cp
    filter drops lines: chunk i = data lines [i*max_lines, (i+1)*max_lines).
    It also matches notebooks/eclipse_wdl_train.ipynb's _batches2 exactly, so a
    premade chunk equals what the notebook would have produced live for that
    chunk_idx. The cap is checked on full batches only, so max_lines should be a
    multiple of _BATCH for an exact boundary (CHUNK_SIZE=100M is)."""
    opener = gzip.open if str(path).endswith('.gz') else open
    batch: list = []
    skipped = 0
    taken = 0
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if skipped < skip_records:
                skipped += 1
                continue
            batch.append(line)
            if len(batch) == _BATCH:
                yield (batch, cp_scale, max_cp)
                taken += len(batch)
                batch = []
                if max_lines and taken >= max_lines:
                    return
    if batch:
        yield (batch, cp_scale, max_cp)


def preprocess(args) -> None:
    inp    = Path(args.data)
    out    = Path(args.out)
    limit  = args.max_records          # cap in INPUT-LINE space (see _line_batches)
    cp     = args.cp_scale

    out.parent.mkdir(parents=True, exist_ok=True)

    size_gb_in = inp.stat().st_size / 1e9
    est_out    = f'≤ {limit * RECORD_SIZE / 1e9:.1f} GB' if limit else '?'
    print(f'input  : {inp}  ({size_gb_in:.2f} GB)')
    print(f'output : {out}  ({est_out})')
    print(f'workers: {args.workers}   cp_scale: {cp}   max_cp: {args.max_cp}')
    if args.skip_records:
        print(f'skip   : {args.skip_records:,} lines')
    if limit:
        print(f'max    : {limit:,} lines')
    print()

    total = 0          # records written (after max_cp filtering)
    lines_consumed = 0  # data lines read post-skip (for EOF detection)
    t0    = time.time()

    def _counting(gen):
        nonlocal lines_consumed
        for batch_tuple in gen:
            lines_consumed += len(batch_tuple[0])
            yield batch_tuple

    with open(out, 'wb') as fout:
        fout.write(MAGIC)
        fout.write(struct.pack('<q', 0))   # placeholder N

        with mp.Pool(args.workers) as pool:
            for data in pool.imap(_process_batch,
                                  _counting(_line_batches(inp, cp, args.skip_records,
                                                           args.max_cp, limit)),
                                  chunksize=args.workers):
                if not data:
                    continue
                n_rec = len(data) // RECORD_SIZE
                fout.write(data)
                total += n_rec
                elapsed = time.time() - t0
                rate    = total / elapsed / 1_000 if elapsed else 0
                gb_out  = total * RECORD_SIZE / 1e9
                print(f'  {total:>12,}  {gb_out:.2f} GB  {rate:.0f}k pos/s', end='\r')

    # Write actual N to header
    with open(out, 'r+b') as f:
        f.seek(8)
        f.write(struct.pack('<q', total))

    elapsed = time.time() - t0
    gb_out  = out.stat().st_size / 1e9
    print(f'\ndone : {total:,} positions in {elapsed:.0f}s  '
          f'({total / elapsed / 1_000:.0f}k pos/s)')
    print(f'file : {out}  ({gb_out:.2f} GB)')
    # Machine-readable summary for orchestrators: lines_consumed < requested
    # max-records means EOF was reached (this is the final partial chunk).
    print(f'SUMMARY records={total} lines_consumed={lines_consumed} '
          f'requested_lines={limit if limit else 0}')
    print(f'\nnow run:  python scripts/train_halfkav2.py --data {out} ...')


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--data',        type=Path, required=True,
                   help='input text file (.txt or .gz)')
    p.add_argument('--out',         type=Path, required=True,
                   help='output binary file (.bin)')
    p.add_argument('--max-records', type=int,  default=None,
                   help='cap on positions written (e.g. 140_000_000 ≈ 18 GB)')
    p.add_argument('--skip-records', type=int, default=0,
                   help='skip this many data lines before writing -- use to '
                        'produce sequential chunks across multiple runs, e.g. '
                        '--skip-records 100_000_000 --max-records 100_000_000 '
                        'for chunk 1.')
    p.add_argument('--workers',     type=int,  default=4)
    p.add_argument('--cp-scale',    type=float, default=410.0)
    p.add_argument('--max-cp',      type=float, default=None,
                   help='drop positions with |cp| above this (near-mate noise '
                        'filter). Match whatever the consuming trainer uses, '
                        'e.g. 4000 for notebooks/eclipse_wdl_train.ipynb.')
    preprocess(p.parse_args())


if __name__ == '__main__':
    main()
