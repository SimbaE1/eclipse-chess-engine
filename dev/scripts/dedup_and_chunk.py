#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deduplicate the eval-labeled training text and emit IID HKAV2BIN chunks.

Runs LOCALLY (CPU only) so no Kaggle GPU quota is spent on preprocessing.
Produces the same `eval_chunk_NN.bin` files the notebook expects
(dev/notebooks/eclipse_wdl_train.ipynb globs `**/eval_chunk_*.bin`), but with
two improvements over scripts/preprocess_halfkav2.py's sequential chunking:

  1. GLOBAL DEDUP by *net input*. HalfKAv2 features depend only on the board
     placement + side-to-move (not castling/ep/move-counters), so two lines with
     the same `<board> <stm>` are the identical training input. We keep only the
     FIRST occurrence of each unique `<board> <stm>`. This removes the massive
     over-representation of common opening tabiyas (reached by millions of games)
     that otherwise act as an implicit importance weight drowning out rarer
     middlegame/endgame positions.

  2. IID CHUNKS. Records are scattered uniformly at random across the N output
     chunks, so every chunk is a representative sample of the whole set rather
     than one month of data. The notebook trains chunk-by-chunk, so month-ordered
     chunks meant the net saw one distribution at a time; IID chunks fix that.

  3. IN-CHECK FILTER. Positions where the side to move is in check are dropped
     (~7%): their eval is dominated by the forced reply (a noisy static-eval
     target), and the net rarely evaluates in-check leaves at inference (search
     extends checks).

Parsing (FEN -> HalfKAv2 feature indices), the record layout (130 bytes:
us[32] uint16, them[32] uint16, tgt float16), and the cp_scale/max_cp filter are
imported verbatim from preprocess_halfkav2.py, so a chunk here is bit-identical
in format to what that script / the notebook's live path would produce.

Two passes over the input:
  Pass A  hash `<board> <stm>` of every data line -> first-occurrence keep mask.
  Pass B  re-read; parse only kept lines (multiprocess) -> random chunk file.

Usage:
    python dev/scripts/dedup_and_chunk.py \\
        --data data/eval_training.txt.gz \\
        --out-dir data/chunks \\
        [--cp-scale 300 --max-cp 4000] \\
        [--chunk-size 100_000_000] [--workers 12] \\
        [--sample 5_000_000]     # cap input lines for a quick test run
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the EXACT parsing + record format the trainer expects.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_halfkav2 import (  # noqa: E402
    MAGIC, RECORD_DTYPE, RECORD_SIZE, _parse_line,
)


def _open_lines(path: Path):
    """Yield raw bytes lines from a .gz (via gzcat) or plain file, fast."""
    if str(path).endswith('.gz'):
        p = subprocess.Popen(['gzcat', str(path)], stdout=subprocess.PIPE,
                             bufsize=1024 * 1024)
        try:
            for line in p.stdout:
                yield line
        finally:
            p.stdout.close()
            p.wait()
    else:
        with open(path, 'rb') as f:
            for line in f:
                yield line


def _pos_key(bline: bytes) -> bytes:
    """Net-input identity: `<board> <stm>` (first two space fields of the FEN)."""
    fen = bline.split(b';', 1)[0]
    sp = fen.split(b' ', 2)
    return sp[0] + b' ' + sp[1] if len(sp) >= 2 else fen


# ── in-check filter ─────────────────────────────────────────────────────────
# Drop positions where the side to move is in check: their eval is dominated by
# the forced reply, so they're a noisy static-eval target and the net rarely
# evaluates in-check leaves at inference (search extends checks). Squares use
# the same a1=0..h8=63 (LERF) layout as _parse_board in preprocess_halfkav2.
_PT_CHK = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6}
_KN = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _stm_in_check(fen: str) -> bool:
    parts = fen.split(' ')
    board, stm = parts[0], (parts[1] if len(parts) > 1 else 'w')
    occ = [0] * 64                      # 0 empty else (color<<3 | pt)
    sq = 56
    for ch in board:
        if ch == '/':
            sq -= 16
            continue
        if ch.isdigit():
            sq += int(ch)
            continue
        pt = _PT_CHK.get(ch.lower())
        if pt is None or not (0 <= sq < 64):
            return False                # malformed -> let the parser reject it
        occ[sq] = ((0 if ch.isupper() else 1) << 3) | pt
        sq += 1
    us = 0 if stm == 'w' else 1
    enemy = 1 - us
    king = (us << 3) | 6
    ksq = -1
    for s in range(64):
        if occ[s] == king:
            ksq = s
            break
    if ksq < 0:
        return False
    kr, kf = ksq >> 3, ksq & 7
    en_n = (enemy << 3) | 2
    for dr, df in _KN:
        r, f = kr + dr, kf + df
        if 0 <= r < 8 and 0 <= f < 8 and occ[r * 8 + f] == en_n:
            return True
    en_p = (enemy << 3) | 1             # enemy pawns attack the king square
    pr = kr + 1 if us == 0 else kr - 1
    if 0 <= pr < 8:
        for f in (kf - 1, kf + 1):
            if 0 <= f < 8 and occ[pr * 8 + f] == en_p:
                return True
    for dirs, (a, b) in ((_ORTHO, (4, 5)), (_DIAG, (3, 5))):  # R/Q ortho, B/Q diag
        for dr, df in dirs:
            r, f = kr + dr, kf + df
            while 0 <= r < 8 and 0 <= f < 8:
                v = occ[r * 8 + f]
                if v:
                    if v >> 3 == enemy and (v & 7) in (a, b):
                        return True
                    break
                r += dr
                f += df
    return False


# ── Pass A: build the first-occurrence keep mask ────────────────────────────
def pass_a(path: Path, sample: int | None):
    """Return (keep: bool[n_lines], n_lines, n_unique).

    A data line = non-empty, non-comment. Both passes filter identically so
    line indices align. keep[i] is True iff line i is the first sighting of its
    `<board> <stm>`.
    """
    cap = sample if sample else 800_000_000
    hashes = np.empty(cap, dtype=np.uint64)
    n = 0
    t0 = time.time()
    for bline in _open_lines(path):
        s = bline.strip()
        if not s or s.startswith(b'#'):
            continue
        if n >= cap:
            if sample:
                break
            hashes = np.concatenate([hashes, np.empty(cap, dtype=np.uint64)])
            cap *= 2
        hashes[n] = hash(_pos_key(s)) & 0xFFFFFFFFFFFFFFFF
        n += 1
        if n % 50_000_000 == 0:
            print(f'  pass A: {n:,} lines  ({n/(time.time()-t0)/1e6:.1f}M/s)',
                  flush=True)
    hashes = hashes[:n]
    # first-occurrence index of each distinct hash value
    _, first_idx = np.unique(hashes, return_index=True)
    keep = np.zeros(n, dtype=bool)
    keep[first_idx] = True
    del hashes
    print(f'  pass A done: {n:,} data lines, {keep.sum():,} unique '
          f'({100*keep.sum()/max(1,n):.1f}%) in {time.time()-t0:.0f}s', flush=True)
    return keep, n, int(keep.sum())


# ── Pass B: parse kept lines, scatter to random chunk files ─────────────────
def _parse_batch(args):
    """Worker: parse a batch of str lines -> packed record bytes (multiple of
    RECORD_SIZE), preserving order."""
    lines, cp_scale, max_cp = args
    recs = []
    n_check = 0
    for line in lines:
        if isinstance(line, (bytes, bytearray)):
            line = line.decode('utf-8', 'replace')
        fen = line.split(';', 1)[0].strip()
        if _stm_in_check(fen):          # drop: side to move is in check
            n_check += 1
            continue
        r = _parse_line(line, cp_scale, max_cp)
        if r:
            recs.append(r)
    if not recs:
        return b'', n_check
    buf = np.empty(len(recs), dtype=RECORD_DTYPE)
    for i, (ua, ta, wp) in enumerate(recs):
        buf[i]['us'] = ua
        buf[i]['them'] = ta
        buf[i]['tgt'] = wp
    return buf.tobytes(), n_check


def pass_b(path: Path, keep, n_lines, out_dir: Path, n_chunks: int,
           cp_scale: float, max_cp: float, workers: int, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f'eval_chunk_{c:02d}.bin' for c in range(n_chunks)]
    files = [open(p, 'wb') for p in paths]
    counts = np.zeros(n_chunks, dtype=np.int64)
    for f in files:
        f.write(MAGIC)
        f.write(struct.pack('<q', 0))          # placeholder N, fixed up at end
    rng = np.random.default_rng(seed)

    BATCH = 50_000
    written = 0
    n_check = 0
    t0 = time.time()

    def _kept_batches():
        batch = []
        i = 0
        for bline in _open_lines(path):
            s = bline.strip()
            if not s or s.startswith(b'#'):
                continue
            if i >= n_lines:
                break
            if keep[i]:
                batch.append(bline)          # bytes; worker decodes
                if len(batch) == BATCH:
                    yield (batch, cp_scale, max_cp)
                    batch = []
            i += 1
        if batch:
            yield (batch, cp_scale, max_cp)

    with mp.Pool(workers) as pool:
        for data, nchk in pool.imap(_parse_batch, _kept_batches(), chunksize=4):
            n_check += nchk
            if not data:
                continue
            arr = np.frombuffer(data, dtype=RECORD_DTYPE)
            b = rng.integers(0, n_chunks, size=len(arr))
            for c in range(n_chunks):
                sub = arr[b == c]
                if len(sub):
                    files[c].write(sub.tobytes())
                    counts[c] += len(sub)
            written += len(arr)
            if written % 5_000_000 < BATCH:
                print(f'  pass B: {written:,} records  '
                      f'({written/(time.time()-t0)/1e3:.0f}k/s)', flush=True)

    for c, f in enumerate(files):
        f.seek(8)
        f.write(struct.pack('<q', int(counts[c])))
        f.close()
    print(f'  pass B done: {written:,} records in {time.time()-t0:.0f}s '
          f'(dropped {n_check:,} in-check)', flush=True)
    # Drop any chunk that ended up empty (keeps indices 0..K-1 contiguous & loadable).
    kept_paths = []
    for c in range(n_chunks):
        if counts[c] == 0:
            paths[c].unlink(missing_ok=True)
            continue
        kept_paths.append((paths[c], counts[c]))
    # Re-name to contiguous eval_chunk_NN.bin if any were dropped.
    final_paths, final_counts = [], []
    for new_idx, (p, cnt) in enumerate(kept_paths):
        target = out_dir / f'eval_chunk_{new_idx:02d}.bin'
        if p != target:
            p.rename(target)
        final_paths.append(target)
        final_counts.append(cnt)
    for p, cnt in zip(final_paths, final_counts):
        print(f'    {p.name}: {cnt:,} records ({p.stat().st_size/1e9:.2f} GB)')
    return final_paths, np.array(final_counts, dtype=np.int64)


def validate(paths):
    """Re-read each chunk header + a few records to confirm it's well-formed."""
    ok = True
    for p in paths:
        with open(p, 'rb') as f:
            magic = f.read(8)
            n = struct.unpack('<q', f.read(8))[0]
        expect = (p.stat().st_size - 16) // RECORD_SIZE
        good = magic == MAGIC and n == expect and (p.stat().st_size - 16) % RECORD_SIZE == 0
        mm = np.memmap(p, dtype=RECORD_DTYPE, mode='r', offset=16, shape=(n,))
        r0 = mm[0]
        # sanity: at least one active feature (< 45056) and a finite target
        n_active = int((r0['us'] < 45056).sum())
        tgt = float(r0['tgt'])
        good = good and n_active > 0 and 0.0 <= tgt <= 1.0
        print(f'  {p.name}: magic={magic==MAGIC} N={n:,}=={expect:,}({n==expect}) '
              f'rec0_active={n_active} rec0_tgt={tgt:.3f}  {"OK" if good else "BAD"}')
        ok = ok and good
    print('validation:', 'ALL OK' if ok else 'FAILURES')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--cp-scale', type=float, default=300.0)
    ap.add_argument('--max-cp', type=float, default=4000.0)
    ap.add_argument('--chunk-size', type=int, default=100_000_000)
    ap.add_argument('--workers', type=int, default=max(2, os.cpu_count() - 4))
    ap.add_argument('--sample', type=int, default=None,
                    help='cap input data lines (quick test)')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    print(f'input   : {args.data}')
    print(f'out-dir : {args.out_dir}')
    print(f'cp_scale={args.cp_scale}  max_cp={args.max_cp}  '
          f'chunk_size={args.chunk_size:,}  workers={args.workers}'
          + (f'  SAMPLE={args.sample:,}' if args.sample else ''))
    print()

    print('=== Pass A: dedup scan ===', flush=True)
    keep, n_lines, n_unique = pass_a(args.data, args.sample)
    n_chunks = max(1, (n_unique + args.chunk_size - 1) // args.chunk_size)
    print(f'\n-> {n_unique:,} unique positions -> {n_chunks} chunk(s)\n', flush=True)

    print('=== Pass B: parse + scatter ===', flush=True)
    paths, counts = pass_b(args.data, keep, n_lines, args.out_dir, n_chunks,
                           args.cp_scale, args.max_cp, args.workers, args.seed)

    print('\n=== Validate ===', flush=True)
    validate(paths)
    print(f'\nTOTAL records written: {int(counts.sum()):,}')


if __name__ == '__main__':
    main()
