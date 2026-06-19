# Eclipse — Kaggle WDL training workflow

This is the end-to-end pipeline for training the HalfKAv2 WDL net on Kaggle's
free GPUs. Local Mac does the data engineering (network-bound), Kaggle does
the training (compute-bound), local Mac packs the result back into a `.nnue`.

The two source-of-truth files are:

| What                            | Where                                  |
|---------------------------------|----------------------------------------|
| Data extraction                 | `dev/scripts/extract_lichess_wdl.py`       |
| Trainer (also embedded in the notebook) | `dev/scripts/train_halfkav2.py`    |
| Quantizer / `.nnue` packer      | `dev/scripts/convert_halfkav2_nnue.py`     |
| The notebook                    | `dev/notebooks/eclipse_wdl_train.ipynb`    |

When any of those scripts change in a way that affects the trained net,
either:
1. update the notebook's embedded code to match, or
2. upload the script as a notebook input and `from train_halfkav2 import ...`.

The notebook in this repo is self-contained (option 1) — copy-pasted code so
the notebook can be uploaded to Kaggle as-is with only the dataset to attach.

---

## Pipeline at a glance

```
┌──────────────────────────────┐         ┌────────────────────────────┐
│ Local (Mac)                  │         │ Kaggle                     │
│                              │         │                            │
│  lichess_db_*.pgn.zst        │         │  Dataset: wdl-training     │
│        │                     │         │   wdl_training.txt(.gz)    │
│        ▼                     │         │        │                   │
│  pgn-extract  (Elo prefilter)│         │        ▼                   │
│        │                     │         │  Notebook: eclipse_wdl     │
│        ▼                     │  upload │   train.ipynb (GPU/T4)     │
│  extract_lichess_wdl.py      │ ──────► │        │                   │
│   (WDL labels + mirror aug)  │         │        ▼                   │
│        │                     │         │  halfkav2.pt               │
│        ▼                     │ ◄────── │  (versioned output)        │
│  data/wdl_training.txt       │ download│                            │
│        │                     │         └────────────────────────────┘
│        ▼                     │
│  convert_halfkav2_nnue.py    │
│        │                     │
│        ▼                     │
│  data/eclipse.nnue           │
└──────────────────────────────┘
```

---

## Stage 1 — Local extraction (the part you're running right now)

Pulls game-outcome WDL labels from a streaming Lichess PGN dump.

### Filters (current)

| Filter         | Value | Why                                                  |
|----------------|-------|------------------------------------------------------|
| `min-elo`      | 2500  | Both players. Below this, win→loss conversion is noisy and corrupts WDL targets. |
| `min-tc-seconds` | 300 | TimeControl base in seconds. 300 = Rapid+, drops bullet noise. |
| `min-ply`      | 10    | Skips the opening-book region (positions where the result barely depends on the position). |
| `mirror-augment` | on (default) | Each emitted position is also written with colors swapped. Doubles the dataset and corrects the white-perspective bias that natural Lichess high-rating games have. |

### Recommended command

```bash
# one-time: write the Elo threshold to a pgn-extract tag-filter file
printf 'WhiteElo >= "2500"\nBlackElo >= "2500"\n' > /tmp/elo_filter.pgn

# run under nohup so an SSH disconnect doesn't kill it
nohup bash -c '
  zstd -dc lichess_db_2025-01.pgn.zst \
    | pgn-extract -s --quiet -t /tmp/elo_filter.pgn 2>/dev/null \
    | python3 dev/scripts/extract_lichess_wdl.py \
        --target 10_000_000 \
        --output data/wdl_training.txt
' > extract.log 2>&1 &
echo $! > extract.pid
```

### What's happening under the hood

- **pgn-extract** is a C tool that drops 99%+ of input games on Elo tags
  alone — adds ~3–4× throughput vs piping straight into Python.
- **Multiprocessing** in `extract_lichess_wdl.py` uses `cpu_count - 1`
  workers; chunks of ~1 MiB of PGN text are sent to workers, each chunk
  cut at the last blank-line game boundary so games never split.
- **Resumability**: after every fully processed chunk, the script atomically
  writes `<output>.ckpt` next to the output file with `{input_bytes,
  games_scanned, games_accepted, positions_emitted, filter_args}`.
  Re-running the same command after a crash/kill/disconnect detects the
  checkpoint, fast-forwards through the already-processed prefix of
  stdin, and appends. Filter args are recorded so resuming with different
  filters aborts (no silent distribution mixing).
- **Mirror augmentation** uses `board.mirror()` — vertical reflection plus
  color swap. STM flips, so the new STM's WDL is the original STM's
  outcome reversed. Fixed a 3× eval asymmetry in the previous net
  (K+Q-up was +1083 cp from white-POV but only −387 cp from black-POV).

### Monitoring a running extract

```bash
# emitted-positions progress
tail -n3 extract.log

# checkpoint state (canonical truth for resume)
cat data/wdl_training.txt.ckpt | python3 -m json.tool

# kill it cleanly (next resume picks up where it left off)
kill -INT "$(cat extract.pid)"
```

### Output format

One line per position:

```
<fen>;<W>;<D>;<L>
```

`(W, D, L)` is `(1, 0, 0)`, `(0, 1, 0)`, or `(0, 0, 1)` from side-to-move's
perspective.

---

## Stage 2 — Upload to Kaggle

### Compress first

`wdl_training.txt` is plain ASCII and compresses ~5× with gzip. Saves
upload time and Kaggle dataset storage.

```bash
gzip -k data/wdl_training.txt
ls -lh data/wdl_training.txt.gz
```

(`-k` keeps the original around for the local conversion step.)

### Create the dataset

1. https://www.kaggle.com/datasets → **New Dataset**.
2. Upload `wdl_training.txt.gz` (and optionally the `.ckpt` for reference).
3. Title: `eclipse-wdl-training` (the notebook expects this slug — change
   the path in the notebook if you pick a different name).
4. Visibility: Private (your account, your training data).
5. Click **Create**.

### Updating later

When you extract a larger / better dataset, **don't make a new dataset** —
upload a new version of the existing one. Kaggle Datasets are versioned;
your notebook can pin to a specific version or always pull `latest`.

---

## Stage 3 — Train on Kaggle GPU

### Open the notebook

1. `dev/notebooks/eclipse_wdl_train.ipynb` → upload to Kaggle as a new notebook
   (Code → New Notebook → File → Upload).
2. **Add data**: attach the `eclipse-partial` dataset (contains `eval_training.txt.gz`).
3. **Settings → Accelerator**: GPU T4 ×2 (DataParallel across both).
4. **Settings → Internet**: On — required for the checkpoint sync below.
5. **Add-ons → Secrets**: add a secret named `KAGGLE_API_TOKEN` with your
   Kaggle API bearer token (same one used by `combine_and_upload_*.py`
   locally, via `KAGGLE_API_TOKEN` env var).
6. **Save Version → Save & Run All** (repeat for each ~9h session).

### Cross-run checkpoint sync

`/kaggle/working` starts empty on every "Save & Run All", so cell 6 syncs
both the model weights (`halfkav2.pt`) and the training position
(`resume_state.pt` — epoch, chunk index, optimizer state, LR-scheduler
state, global step) through a private dataset, `simbae11/eclipse-checkpoint`,
via the Kaggle API:

- **Start of run**: `ckpt_download()` pulls the latest checkpoint version
  (no-op, prints "starting fresh", on the very first run before the
  dataset exists).
- **After every ~100M-position chunk**: `ckpt_upload()` pushes a new
  dataset version with the updated `halfkav2.pt` + `resume_state.pt`.

This makes "Save & Run All" fully resumable — no manual re-attaching of
the previous version's output between runs. If `KAGGLE_API_TOKEN` isn't
set as a Secret, sync is silently skipped and each run trains from scratch.

### What the notebook does

| Cell | Purpose |
|------|---------|
| 1 | Print env (Python / torch / CUDA / GPU name) |
| 2 | Locate `eval_training.txt(.gz)` under `/kaggle/input/` |
| 3 | FEN → HalfKAv2 feature indices (1:1 mirror of `src/nnue.cpp::feature_index`) |
| 4 | Dataset helpers (`HalfKAv2Dataset`, `HalfKAv2BinaryDataset`, streaming/memmap) |
| 5 | `HalfKAv2Net` — float mirror of the C++ inference path, with QAT |
| 6 | Checkpoint sync + automated chunked training loop (100M pos/chunk) |
| 7 | Output notes — download `halfkav2.pt` from the Output tab |

### Network architecture (current)

```
features (HalfKAv2, 45056 per perspective)
   │  Embedding [45057, 1024]  (last row is the padding sentinel, pinned to 0)
   ▼
ft_us [1024]      ft_them [1024]
   │ ClippedReLU(0,1)     │ ClippedReLU(0,1)
   └────────┬─────────────┘
            ▼
        concat [2048]
            │ Linear 2048 → 256, ClippedReLU
            ▼
          [256]
            │ Linear 256 → 64, ClippedReLU
            ▼
           [64]
            │ Linear 64 → 3
            ▼
       logits [W, D, L]
```

Sizes are `FT_OUT=1024`, `L1_OUT=256`, `L2_OUT=64`, `L3_OUT=3`. These must
stay in sync with `src/nnue.hpp` — drift = silently wrong inference. The
`.nnue` magic is bumped (`0xECCC0003`) when the feature set or layer
sizes change so old files are rejected by the loader instead of silently
producing garbage.

### Hyperparameters (current)

| Knob          | Value | Notes |
|---------------|-------|-------|
| optimizer     | Adam, `weight_decay=1e-4` | L2 regularization helps the bigger 256/64 head not overfit. |
| `lr`          | `1e-3` | Peak after warmup. |
| `warmup-steps`| 500 | Linear warmup. Adam's 1st-moment estimates are noisy on a fresh init with large batches; full lr in step 1 can push the embedding into a degenerate region. |
| LR schedule   | CosineAnnealingLR, `T_max=epochs` | Stepped per epoch after warmup. |
| `batch-size`  | 4096 | Fits in T4 16 GB. Bump to 8192/16384 on P100 if you have headroom. |
| `epochs`      | 5 | Default. Watch val-loss; bump if it's still falling. |
| `val-frac`    | 0.05 | Deterministic last-5% slice, capped at 10%. Held out across restarts so val examples never leak into train. |
| Loss          | Soft-label cross-entropy on log-softmax | The right loss for a probability-distribution target. MSE on softmax (the previous bug) under-trains the tails. |
| Padding row   | `ft.weight[FT_IN_FEATURES] = 0` after every step | The dataset pads short rows with the sentinel index; pinning the row keeps padding from accruing a non-zero contribution. |

### Reported per epoch

```
epoch N train_loss X.XXXXX  val_loss X.XXXXX  val_argmax_acc 0.XXX  (lr next: X.XXXXXX)
```

`val_argmax_acc` = fraction of val positions where `argmax(logits) ==
argmax(target)`. It's a coarse sanity check — the real signal is val_loss.
Game-outcome labels are noisy per-position (the same FEN can have all three
outcomes in different games), so val_argmax_acc tops out well below 1.0
even for a perfectly fit net.

### Kaggle session limits

- T4 sessions: ~9 hours max, then the kernel is killed.
- Output (`/kaggle/working/`) is persisted as **Notebook Output** on the
  version that finishes — versions that are killed mid-run lose `/kaggle/working`.
- If you need >9 hours: save mid-training checkpoints into `/kaggle/working`
  *and* call **Save Version → Save & Run All** at intervals, or split
  training across multiple notebooks loading the previous `.pt`.

---

## Stage 4 — Pack into `.nnue` locally

Once `halfkav2.pt` is in `/kaggle/working/`, click **Output** in the
notebook UI → download `halfkav2.pt` → drop it into `data/`:

```bash
python dev/scripts/convert_halfkav2_nnue.py from-torch \
    --state-dict data/halfkav2.pt \
    --out data/eclipse.nnue \
    --output-cp-per-unit 410.0
```

`--output-cp-per-unit 410.0` matches the tiny-NNUE / Stockfish sigmoid
convention and is recorded in the `.nnue` header. It's what the engine
uses to convert the net's raw output unit into centipawns at search
time, so it has to be consistent with the training-side sigmoid scale.

### Verify the loader accepts it

```bash
ECLIPSE_NNUE_PATH=$PWD/data/eclipse.nnue \
ECLIPSE_NNUE_TRAINED=1 \
./build/tests/test_nnue
```

Should end with `PASS  nnue`. Failure usually means the magic/version
doesn't match (rebuild the engine) or a tensor shape differs (the
notebook's `L1_OUT/L2_OUT` drifted from `convert_halfkav2_nnue.py`).

### Saturation warnings

`convert_halfkav2_nnue.py` will warn if too many weights clip during
int8/int16 quantization:

```
WARNING: l1.weight: 47/32768 weights saturated (0.14%). ...
```

A few tenths of a percent is fine; >1% means the trainer needs tighter
weight regularization or a smaller `lr`.

---

## What's different vs the older Kaggle pipeline

This is the "since v0.1.0" rundown for anyone resuming after a break:

- **Labels are game outcomes (WDL), not Stockfish cp.** Lines are 4-field
  `fen;W;D;L`. The trainer autodetects the older 2-field `fen;cp` format
  and converts via sigmoid, but warns — the cp path collapses the
  draw channel to zero and is now legacy-only.
- **Mirror augmentation is on by default.** Doubles dataset volume,
  guarantees symmetric W/B coverage. Pass `--no-mirror-augment` to
  disable (don't, unless debugging).
- **Loss is cross-entropy on log-softmax.** The previous MSE-on-softmax
  loss under-trained the tails and combined with the broken WDL→cp
  derivation produced a W/L binary classifier with the D channel
  permanently zeroed.
- **Bigger head: 1024×2 → 256 → 64 → 3 (was 128 → 32 → 1).** Three-class
  WDL output replaces the single cp output. Head capacity increased to
  match.
- **Validation hold-out.** Last-5% deterministic slice, capped at 10%.
  Reported each epoch alongside `val_argmax_acc`.
- **Linear warmup + cosine schedule.** Was constant lr.
- **Weight decay 1e-4** in Adam. Was 0.
- **Extraction is multiprocessed + resumable.** Single-core extract was
  ~3 k games/sec; the current pipeline is ~10–20× faster with workers,
  another ~3–4× on top via pgn-extract. Crash/disconnect-safe via the
  `.ckpt` file.

---

## Quick reference

```bash
# extract (local)
zstd -dc lichess_db_2025-01.pgn.zst \
  | pgn-extract -s --quiet -t /tmp/elo_filter.pgn 2>/dev/null \
  | python3 dev/scripts/extract_lichess_wdl.py \
      --target 10_000_000 --output data/wdl_training.txt

# upload (local → Kaggle)
gzip -k data/wdl_training.txt
# → upload data/wdl_training.txt.gz to kaggle dataset 'eclipse-wdl-training'

# train (Kaggle, GPU T4)
# → run dev/notebooks/eclipse_wdl_train.ipynb with the dataset attached

# pack (local)
python dev/scripts/convert_halfkav2_nnue.py from-torch \
    --state-dict data/halfkav2.pt \
    --out data/eclipse.nnue --output-cp-per-unit 410.0
```
