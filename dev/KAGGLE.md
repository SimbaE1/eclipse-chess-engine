# Eclipse — Kaggle NNUE training workflow

End-to-end pipeline for training the HalfKAv2 NNUE value net on Kaggle's free
GPUs. The local Mac does the data engineering (network/CPU-bound), Kaggle does
the training (GPU-bound), and the local Mac packs the result back into a
`.nnue`.

> **Labels are Stockfish depth-22 evaluations, not game outcomes.** Lichess runs
> Stockfish (~depth 22) server-side on a large fraction of rated games and embeds
> the score in the PGN (`{ [%eval 0.27] }`). We harvest those as per-position
> centipawn labels — far lower-noise than the old game-outcome WDL labels (a
> single game result smeared over every position). The net is a **single-logit
> win-probability** model (`win_prob = sigmoid(cp / cp_scale)`), trained with BCE.
> If you're looking for the old `fen;W;D;L` 3-class WDL pipeline, see
> ["What changed"](#what-changed-vs-the-old-wdl-pipeline) at the bottom — it's gone.

The source-of-truth files:

| What                                    | Where                                    |
|-----------------------------------------|------------------------------------------|
| Eval-label extraction (PGN → `fen;cp`)  | `dev/scripts/extract_lichess_evals.py`   |
| Text → binary HKAV2BIN chunks           | `dev/scripts/preprocess_halfkav2.py`     |
| Trainer (also embedded in the notebook) | `dev/scripts/train_halfkav2.py`          |
| Quantizer / `.nnue` packer              | `dev/scripts/convert_halfkav2_nnue.py`   |
| One-shot "pull latest + pack"           | `dev/scripts/fetch_latest_net.sh`        |
| The notebook (self-contained)           | `dev/notebooks/eclipse_wdl_train.ipynb`  |
| net2widernet launch runbook             | `dev/net2widernet.md`                    |

The notebook is **self-contained** (the trainer code is copy-pasted in), so it
runs on Kaggle with only datasets attached — no `import train_halfkav2`. When a
script changes in a way that affects the net, mirror the change into the
notebook (or it silently trains something else).

---

## Pipeline at a glance

```
┌─────────────────────────────────────┐        ┌───────────────────────────────┐
│ Local (Mac)                         │        │ Kaggle                        │
│                                     │        │                               │
│  lichess_db_*.pgn.zst               │        │  Dataset: eclipse-chunks-a    │
│        │ extract_lichess_evals.py   │        │   eval_chunk_00..06.bin (RAW) │
│        ▼  (harvest %eval comments)  │        │  Dataset: eclipse-checkpoint  │
│  data/eval_*.txt   <fen>;<cp>       │        │   halfkav2.pt + resume_state  │
│        │ preprocess_halfkav2.py     │ upload │        │                      │
│        ▼  --cp-scale 300 --max-cp.. │ ─────► │        ▼                      │
│  data/chunks/eval_chunk_NN.bin      │        │  Notebook (T4 ×2, DataParallel)│
│   (HKAV2BIN, 100M input-lines each) │        │   memmap chunk → train        │
│                                     │        │   net2widernet 2× widen        │
│  data/eclipse.nnue  ◄───────────────│download │   halfkav2.pt → checkpoint    │
│   convert_halfkav2_nnue.py (cp=300) │        │   (synced every ~100M chunk)  │
└─────────────────────────────────────┘        └───────────────────────────────┘
```

---

## Stage 1 — Extract eval labels (local)

`extract_lichess_evals.py` streams a Lichess monthly PGN dump, keeps only games
that pass the quality bar, and emits every `%eval`-annotated position as
`<fen>;<score_cp>` — centipawns **from the side-to-move's perspective** (Lichess
stores them in pawns from White's POV; the script converts sign + units). Mate
scores (`#N`) saturate to `±mate-cp`.

### Filters (script defaults)

| Flag               | Default | Why |
|--------------------|---------|-----|
| `--min-elo`        | 2500    | Both players; higher Elo ⇒ the embedded eval covers sharper, more instructive play. |
| `--min-tc-seconds` | 300     | TimeControl base seconds; drops bullet/blitz noise. |
| `--min-ply`        | 10      | Skips the opening-book region. |
| `--mate-cp`        | 1500    | `#N` mate scores saturate here (the sigmoid flattens past this anyway). |

A raw per-line prefilter (Elo/TC tags) runs before python-chess touches a game —
at 2500+ Elo only ~0.5% of games survive, so full SAN parsing on the rest would
dominate runtime.

```bash
# one monthly dump → eval-labelled text (nohup so a disconnect can't kill it)
nohup python3 dev/scripts/extract_lichess_evals.py \
    --input  lichess_db_2025-01.pgn.zst \
    --output data/eval_2025_01.txt \
    --min-elo 2500 --min-tc-seconds 300 \
    > extract.log 2>&1 &
```

Concatenate the months you want into one `data/eval_combined.txt`. (The current
deployed net was trained on ~**688M positions** across ~20 months.)

---

## Stage 1b — Preprocess into premade binary chunks (local)

**This is the step that saves GPU quota.** Parsing FEN → HalfKAv2 indices on
Kaggle costs ~40 min per 100M positions of GPU-session wall-clock. Instead we do
it **once, locally**, into raw `HKAV2BIN` chunks the notebook memmaps directly.

`preprocess_halfkav2.py` writes a binary file:

```
[0:8]  magic b'HKAV2BIN'
[8:16] N  int64        (number of positions)
[16:]  N × 130 bytes:  us_idx[32] uint16 · them_idx[32] uint16 · target float16 (win prob)
```

The target is `win_prob = sigmoid(cp / cp_scale)`. **The cp-scale and max-cp MUST
match the notebook's `TRAIN_CFG`** or a premade chunk won't equal what the
notebook would have built live:

```bash
# split data/eval_combined.txt into 100M-line chunks in INPUT-LINE space, so
# chunks are a clean non-overlapping partition despite the max-cp filter.
N=0
for skip in 0 100000000 200000000 300000000 400000000 500000000 600000000; do
  python3 dev/scripts/preprocess_halfkav2.py \
      --data data/eval_combined.txt \
      --out  data/chunks/eval_chunk_$(printf '%02d' $N).bin \
      --skip-records $skip --max-records 100000000 \
      --cp-scale 300 --max-cp 4000 --workers 8
  N=$((N+1))
done
```

Each chunk is ~13 GB (the short final chunk is just the EOF remainder). 688M
positions → 7 chunks (`eval_chunk_00..06.bin`, ~89 GB total).

---

## Stage 2 — Upload to Kaggle

The chunks go up **RAW** (Kaggle does not decompress on their end):

```bash
export KAGGLE_API_TOKEN=<your KGAT_… token>
kaggle datasets version -p data/chunks -m "rebuild chunks" --dir-mode tar
# (first time: kaggle datasets create -p data/chunks  → slug simbae11/eclipse-chunks-a)
```

- **`simbae11/eclipse-chunks-a`** — the 7 premade chunks (training data).
- **`simbae11/eclipse-checkpoint`** — `halfkav2.pt` + `resume_state.pt`; this is
  **not** something you upload by hand, the notebook syncs it (below).

Dataset uploads do **not** consume GPU quota — only kernel runs do. After
uploading you can gzip the local `.bin` + source text to reclaim disk.

---

## Stage 3 — Train on Kaggle GPU

Use **`dev/net2widernet.md`** for the exact, verified launch steps. The essentials:

1. Upload / open `dev/notebooks/eclipse_wdl_train.ipynb`.
2. **Add data**: attach `eclipse-chunks-a`. (`eclipse-checkpoint` does *not* need
   to be attached — it's API-downloaded via the secret below.)
3. **Settings → Accelerator: GPU `T4 ×2`** — **not** the generic "GPU", which can
   hand you a **Tesla P100** (CUDA sm_60) that the installed PyTorch (sm_70+)
   can't run (`CUDA error: no kernel image available`). See [Gotchas](#gotchas).
4. **Settings → Internet: On.**
5. **Add-ons → Secrets**: add `KAGGLE_API_TOKEN` (your Kaggle bearer token). This
   gates **both** the checkpoint download (warm start) **and** the per-chunk
   sync — without it the run trains from scratch and the widen never fires.
6. **Save Version → Save & Run All** (repeat per ~9–12 h session; it resumes).

### Cross-run checkpoint sync

`/kaggle/working` starts empty on every "Save & Run All", so cell 6 syncs the
weights (`halfkav2.pt`) and training position (`resume_state.pt` — epoch, chunk,
optimizer, LR-scheduler, global step) through the private `eclipse-checkpoint`
dataset via the Kaggle API:

- **Start**: `ckpt_download()` pulls the latest checkpoint (this is the warm-start
  source — the load path is the API download, *not* a mounted dataset).
- **After every ~100M-position chunk**: `ckpt_upload()` pushes a new version.

So sessions are fully resumable with no manual re-attaching.

### net2widernet (the current run)

`NET2WIDER=True` performs a one-time **function-preserving 2× widen** when the
checkpoint it loads is still the narrow shape — `FT 1024→2048, L1 512→1024,
L2 128→256`. The wide net starts computing exactly what the narrow `ep2c9` net
does, then trains on. Once a wide checkpoint has been saved, re-running just
resumes it (the widen won't re-fire). Full details + re-arming in
`dev/net2widernet.md`.

### Network architecture

Single-logit win-probability head (`sigmoid(out) = win prob`):

```
deployed (narrow):   features(HalfKAv2 45056) → 1024×2 → 512 → 128 → 1
net2widernet (wide): features(HalfKAv2 45056) → 2048×2 → 1024 → 256 → 1
```

`FT_OUT/L1_OUT/L2_OUT` in the notebook must stay in sync with `kFtOutSize` & the
layer sizes in `src/nnue.hpp` / `src/accumulator.hpp` and
`convert_halfkav2_nnue.py`. **A wide net will NOT load until the C++ side is
widened to match** — that's the gate before any strength test (see
`dev/net2widernet.md` §7 and `dev/RELEASE_CHECKLIST.md`).

### Hyperparameters (current `TRAIN_CFG`)

| Knob            | Value   | Notes |
|-----------------|---------|-------|
| optimizer       | Adam, `weight_decay=1e-4` | |
| `lr`            | `1e-3`  | At batch 16384; matches the SF NNUE convention (`2e-3` diverged). |
| `batch_size`    | 16384   | DataParallel splits 8192/GPU across the two T4s. |
| `warmup_steps`  | 500     | Linear warmup before the cosine schedule. |
| LR schedule     | Cosine, oversized `T_max` | **Deliberately under-anneals** — too-low LR makes the net regress. Don't "fix" the mismatch. |
| `cp_scale`      | 300     | `win_prob = sigmoid(cp/300)`. Must match the preprocess `--cp-scale`. |
| `max_cp`        | 4000    | Drops near-mate noise (`|cp|>4000`, ~0 gradient). Must match preprocess `--max-cp`. |
| `EPOCHS`        | 3       | ≈ 5–6 h GPU over the full set. |
| `CHUNK_SIZE`    | 100M    | Positions per checkpoint sync. |
| `VAL_SIZE`      | 200k    | Fixed val set, sliced once from `eval_chunk_00.bin`. |
| precision       | AMP fp16 + GradScaler | |
| Loss            | `BCEWithLogits` on the win-prob target | |

Per chunk the run reports `val=X.XXXXX` (the BCE val loss) and syncs the
checkpoint with that figure in the dataset-version note.

### Gotchas

- **P100 is unusable.** "GPU" can assign a Tesla P100 (sm_60); PyTorch 2.x cu12x
  supports sm_70+ only, so every CUDA kernel dies on the first forward pass. Pin
  **T4 ×2**. (The accelerator can't be set from `kernel-metadata.json`, so a CLI
  `kernels push` can reset it — verify in the UI.)
- **The chunk glob must be recursive.** Kaggle mounts datasets at either the flat
  `/kaggle/input/<slug>/…` or the nested `/kaggle/input/datasets/<owner>/<slug>/…`.
  The notebook globs `/kaggle/input/**/eval_chunk_*.bin` (recursive); a one-level
  glob silently finds nothing and the run dies at val-set build with
  `No premade chunk 0`. If you see that, the chunks mounted at a deeper path.
- **No `KAGGLE_API_TOKEN` secret ⇒ trains from scratch.** The message
  `no KAGGLE_API_TOKEN secret … training from scratch` (also fires on a transient
  startup connection error) means no warm start, no widen, no sync.
- **GPU quota: ~30 h/week, ~9–12 h/session.** Not CLI-queryable; a failed push
  saying `Maximum weekly GPU quota … reached` means wait for the rolling reset.

---

## Stage 4 — Pack into `.nnue` (local)

The one-shot path pulls the latest checkpoint and packs it with the **correct**
cp-scale:

```bash
export KAGGLE_API_TOKEN=<token>
dev/scripts/fetch_latest_net.sh            # → data/eclipse.nnue (cp=300)
```

Or do it manually — **note the `--output-cp-per-unit 300`**; the converter still
*defaults to 410*, which would be wrong for the current net:

```bash
python3 dev/scripts/convert_halfkav2_nnue.py from-torch \
    --state-dict data/halfkav2.pt \
    --out data/eclipse.nnue \
    --output-cp-per-unit 300
```

This value is recorded in the `.nnue` header and is what the engine uses to turn
the net's raw output into centipawns at search time — it must match the
training-side `cp_scale`.

### Verify + validate

```bash
ctest --test-dir build -R test_nnue          # loader accepts magic/shape
```

A shape mismatch means the notebook's layer sizes drifted from
`convert_halfkav2_nnue.py` / `src/nnue.hpp`. The converter also warns if too many
weights saturate during int8/int16 quantization — a few tenths of a percent is
fine; >1% means tighten regularization or lower `lr`.

**Bigger ≠ better until the games say so.** Before promoting any net (especially
the wide one), run an SPRT / match vs the current net per
`dev/RELEASE_CHECKLIST.md`.

---

## What changed vs the old WDL pipeline

For anyone resuming after a long break — the pipeline was rebuilt:

- **Labels: Stockfish depth-22 evals, not game outcomes.** Lines are 2-field
  `fen;cp` (STM perspective), harvested from Lichess `%eval` comments. The old
  4-field `fen;W;D;L` game-outcome path is retired (a single game result applied
  to every position in the game is far noisier than a per-position eval).
- **Head: single-logit win-probability, not 3-class WDL.** Architecture is
  `… → 512 → 128 → 1` (narrow) — the old `… → 256 → 64 → 3` WDL head is gone.
  Loss is BCE on `sigmoid(cp/cp_scale)`.
- **Premade binary chunks.** FEN→HalfKAv2 parsing moved off Kaggle into local
  `preprocess_halfkav2.py` (HKAV2BIN), saving many GPU-quota hours.
- **net2widernet.** Active work is a function-preserving 2× widen of the deployed
  net, not a from-scratch train.
- **cp-scale 300** (was 410) for more gradient in the typical range; the converter
  must be told `--output-cp-per-unit 300`.
- **T4 ×2 + DataParallel, AMP fp16, batch 16384**, with per-chunk checkpoint sync
  through `eclipse-checkpoint`.
