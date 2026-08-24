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
| Kaggle launch mechanics (historical)    | `dev/net2widernet.md`                    |

`dev/net2widernet.md` describes the *previous* run (the 2× widen that produced
the deployed net) and is superseded — its Kaggle mechanics (§1, §4, §6) are still
correct, its plan is not. **This file is the live procedure.**

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
│        ▼  (harvest %eval comments)  │        │  Dataset: eclipse-checkpoint- │
│  data/eval_*.txt   <fen>;<cp>       │        │   sf16: halfkav2.pt + resume  │
│        │ preprocess_halfkav2.py     │ upload │        │                      │
│        ▼  --cp-scale 300 --max-cp.. │ ─────► │        ▼                      │
│  data/chunks/eval_chunk_NN.bin      │        │  Notebook (T4 ×2, DataParallel)│
│   (HKAV2BIN, 100M input-lines each) │        │   memmap chunk → train        │
│                                     │        │   1024x2 → 16 → 32 → 1        │
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

The recipe above produces ~13 GB chunks. **What is actually on Kaggle today is
finer-grained**: `simbae11/eclipse-chunks-a` holds **192 files** named
`eval_chunk_000..191.bin`, 507 MB / 3.9M positions each (the last is short),
**746.7M positions, ~30 GB compressed**. The notebook keys premade chunks by the
integer in the filename, so either granularity loads — but the training loop
treats one *file* as one iteration, which is what `SYNC_EVERY` and the cosine
`T_max` have to be sized against. See [Gotchas](#gotchas).

---

## Stage 2 — Upload to Kaggle

The chunks go up **RAW** (Kaggle does not decompress on their end):

```bash
export KAGGLE_API_KEY=<your KGAT_… token>
kaggle datasets version -p data/chunks -m "rebuild chunks" --dir-mode tar
# (first time: kaggle datasets create -p data/chunks  → slug simbae11/eclipse-chunks-a)
```

- **`simbae11/eclipse-chunks-a`** — the 7 premade chunks (training data).
- **`simbae11/eclipse-checkpoint-sf16`** — `halfkav2.pt` + `resume_state.pt` for
  the current `1024x2 → 16 → 32` retrain; this is **not** something you upload by
  hand, the notebook syncs it (below). The notebook creates it on its first
  `ckpt_upload()`, so it will not exist until the first chunk completes.
- **`simbae11/eclipse-checkpoint`** — the **deployed** `2048x2 → 1024 → 256` net.
  Frozen. Do not point a training run at this slug: `ckpt_upload()` versions the
  dataset in place, and it is the only copy of the net currently in `data/`.

Dataset uploads do **not** consume GPU quota — only kernel runs do. After
uploading you can gzip the local `.bin` + source text to reclaim disk.

---

## Stage 3 — Train on Kaggle GPU

1. Upload / open `dev/notebooks/eclipse_wdl_train.ipynb`.
2. **Add data**: attach `eclipse-chunks-a`. The checkpoint dataset does *not* get
   attached — it's API-downloaded via the secret below. (`kernel-metadata.json`
   lists only `eclipse-chunks-a` for the same reason.)
3. **Accelerator: GPU `T4 ×2`** — **not** the generic/P100 option, which hands you
   a **Tesla P100** (CUDA sm_60) that the installed PyTorch (sm_70+) can't run
   (`CUDA error: no kernel image available`). This *is* settable from
   `kernel-metadata.json` via **`"machine_shape": "NvidiaTeslaT4"`** (the field
   isn't in `kernel init`'s template, but `kernels push` sends it and
   `kernels pull -m` shows it). `NvidiaTeslaT4` is Kaggle's T4 **×2** shape —
   there is no single-T4 option. The other legal values are `NvidiaTeslaP100`
   and `Tpu1VmV38`. Keep the field in the file: without it a CLI push sends
   `None` and silently drops the accelerator. Cell 6 prints the device names as a
   receipt. See [Gotchas](#gotchas).
4. **Settings → Internet: On.**
5. **Add-ons → Secrets**: add **`KAGGLE_API_KEY`** (your Kaggle bearer token; the
   notebook also accepts the older name `KAGGLE_API_TOKEN`). This gates **both**
   the checkpoint download (resume) **and** the per-chunk sync — without it the
   run restarts from scratch every session and the weights die with the session.
   **This step is UI-only.** Unlike the accelerator, a secret cannot be set from
   `kernel-metadata.json` and there is no `kaggle secrets` command — a
   `kernels push` does *not* carry it. Attaching it once to the notebook is
   enough; it survives later pushes. **Verify it took** by reading the first
   lines of the run's log (see [Watching a run](#watching-a-run)): you want
   `using Kaggle secret KAGGLE_API_KEY`, not
   `no Kaggle API secret ... checkpoint sync disabled`. If you see the latter,
   stop the run — every GPU-hour it burns is thrown away.
6. **Save Version → Save & Run All** (repeat per ~9–12 h session; it resumes).

### Cross-run checkpoint sync

`/kaggle/working` starts empty on every "Save & Run All", so cell 6 syncs the
weights (`halfkav2.pt`) and training position (`resume_state.pt` — epoch, chunk,
optimizer, LR-scheduler, global step) through the private checkpoint dataset
(`CKPT_DATASET`, currently `eclipse-checkpoint-sf16`) via the Kaggle API:

- **Start**: `ckpt_download()` pulls the latest checkpoint (this is the resume
  source — the load path is the API download, *not* a mounted dataset).
- **Every `SYNC_EVERY` chunk-files** (currently 24 ≈ 15 min of work at the
  measured 37 s/chunk, ≈48 uploads across a 6-epoch run): `ckpt_upload()` pushes
  a new dataset version, printing its size and wall-clock so the sync tax stays
  visible. A `_pending_sync` flag forces a final upload when a chunk loop ends,
  so no epoch boundary is lost.
- **Chunk 0 of every epoch also syncs**, unconditionally. This is a deliberate
  early probe: it proves the secret, the dataset slug and the upload path all
  work about a minute into the session, instead of letting a broken sync hide
  until the first `SYNC_EVERY` boundary.

So sessions are fully resumable with no manual re-attaching.

The notebook also **shape-gates** the checkpoint before resuming: if the
downloaded `halfkav2.pt` has an `ft.weight` that doesn't match this run's
`FT_OUT`, it says so and trains from scratch instead of dying in `load_state_dict`.
Seeing that message unexpectedly means `CKPT_DATASET` is pointing at another net.

### Watching a run

Mid-run, `kaggle kernels output` writes nothing, `kaggle kernels logs` prints
nothing, and the SDK's `kernels_logs()` returns an empty list — all three only
have content **after** the version finishes. The only way to see a *running*
log is the streaming call, which yields dicts with a `data` key and blocks
forever at the tail, so bound it:

```python
import itertools
from kaggle.api.kaggle_api_extended import KaggleApi
a = KaggleApi(); a.authenticate()
for rec in itertools.islice(a.kernels_logs_stream('simbae11/tcec-chess-engine'), 200):
    print((rec.get('data') or '').rstrip())
```

The first ~15 lines are the receipt for everything that can silently go wrong:
the secret (`using Kaggle secret KAGGLE_API_KEY`), the accelerator
(`GPU 0/1: Tesla T4`), the chunk count (`found 192 premade chunk(s)`), the
architecture, and `LR schedule: N chunks/epoch × E epochs = T cosine steps`.
**Check them before walking away** — the run does not fail on any of these, it
just quietly produces less than you asked for.

`kaggle kernels status <slug>` gives just the state
(`RUNNING` / `COMPLETE` / `ERROR`).

### The sf16 retrain (the current run)

This is a **from-scratch** train at the Stockfish-modern shape. The previous run
(`NET2WIDER`) widened the net 2× in a *function-preserving* way, which by
construction bought zero strength and cost ~4× the time: the deployed net spends
**36.5 µs** per `nnue::evaluate()` where comparable engines spend 0.2–0.5 µs, and
its feature-transformer output is only ~1.2% nonzero (51 of 4096 activations), so
the 1024-wide L1 is mostly multiplying by zero.

The fix is the layer *after* the FT, not the FT itself. The FT→L1 product is
evaluated **sparsely** — its cost scales with `L1_OUT`, not with the input width —
so `1024 → 16` is by far the largest speedup available. `1024x2 → 16 → 32 → 1` is
Stockfish's own shape, so it is not an accuracy gamble: the representation lives
in the 1024-wide feature transformer, and the hidden stack after it is small on
purpose. The `NET2WIDER` code has been deleted from the notebook (see git
history if you need it).

Measured on the same fixed-100,024-node `bench` (Apple silicon, NEON):

| Net | nps |
|---|---|
| `2048x2 → 1024 → 256`, scalar kernels | 53k–64k |
| `2048x2 → 1024 → 256`, NEON kernels | 85k–96k |
| `1024x2 → 16 → 32`, NEON kernels (random weights) | 142k–157k |

### Network architecture

Single-logit win-probability head (`sigmoid(out) = win prob`):

```
original:            features(HalfKAv2 45056) → 1024×2 → 512  → 128 → 1
deployed (widened):  features(HalfKAv2 45056) → 2048×2 → 1024 → 256 → 1
sf16 (training now): features(HalfKAv2 45056) → 1024×2 →   16 →  32 → 1
```

`FT_OUT/L1_OUT/L2_OUT` in the notebook are the contract with the C++ engine. The
**converter** infers them from the checkpoint's tensor shapes, so it needs no
flag; the **engine** does not, and will reject a mismatched file at load time.

### Switching the engine to a new architecture

Three constants in two headers, and nothing else:

| Notebook | C++ |
|---|---|
| `FT_OUT` | `kFtOutSize` — `src/accumulator.hpp:19` |
| `L1_OUT` | `kL1OutSize` — `src/nnue.hpp:46` |
| `L2_OUT` | `kL2OutSize` — `src/nnue.hpp:47` |

All four SIMD ladders (AVX-512 / AVX2 / NEON / scalar) already compile and run at
both the current and the narrow shape — layers too small to fill one vector
dispatch to a scalar core via `if constexpr`, and the sparse kernel handles
`kOut = 16`. This has been verified end to end: at `1024/16/32` the engine loads a
converted net, prints `NNUE loaded: … (HalfKAv2-1024x2-16-32, output_cp/unit=300)`
and searches normally.

**Do not flip these until a real checkpoint exists.** The moment you do, the
deployed 189 MB net stops loading and `ctest` goes red. Flip them, rebuild,
`ctest`, then SPRT per `dev/RELEASE_CHECKLIST.md` — the new net has to *earn* the
swap on the board, not on the clock.

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
| `EPOCHS`        | 6       | From-scratch run; ~6 fits a 12 h session at this size (the net is 46.2M params vs 185M for the wide one). |
| `CHUNK_SIZE`    | 100M    | Live-preprocessing size only. With premade chunks attached one iteration is one **file** (3.9M positions). |
| `SYNC_EVERY`    | 8       | Chunk-files per `ckpt_upload()`. Syncing every file would push ~1150 versions of ~550 MB and cost more wall-clock than the training. |
| `VAL_SIZE`      | 200k    | Fixed val set, sliced once from `eval_chunk_00.bin`. |
| precision       | AMP fp16 + GradScaler | |
| Loss            | `BCEWithLogits` on the win-prob target | |

Per chunk the run reports `val=X.XXXXX` (the BCE val loss) and syncs the
checkpoint with that figure in the dataset-version note.

### Gotchas

- **P100 is unusable.** A P100 (sm_60) fails every CUDA kernel on the first
  forward pass; PyTorch 2.x cu12x is sm_70+. Pin `"machine_shape":
  "NvidiaTeslaT4"` in `kernel-metadata.json` (= T4 ×2) and confirm with
  `kaggle kernels pull -m` after a push.
- **A chunk-file is not `CHUNK_SIZE`.** `eclipse-chunks-a` is **192 files of 3.9M
  positions** (746.7M total), not 7 × 100M. One training-loop iteration is one
  *file*, and `cosine.step()` / `ckpt_upload()` fire per iteration — so anything
  counting iterations must use `len(PREMADE_CHUNKS)`. Deriving the cosine `T_max`
  from `685M/CHUNK_SIZE` gives 42 against 1152 real steps, which runs the schedule
  past its half-period and turns the anneal into a cyclic LR.
- **No `torch.compile`.** It breaks DataParallel replication on the two T4s. The
  notebook deliberately doesn't call it.
- **The chunk glob must be recursive.** Kaggle mounts datasets at either the flat
  `/kaggle/input/<slug>/…` or the nested `/kaggle/input/datasets/<owner>/<slug>/…`.
  The notebook globs `/kaggle/input/**/eval_chunk_*.bin` (recursive); a one-level
  glob silently finds nothing and the run dies at val-set build with
  `No premade chunk 0`. If you see that, the chunks mounted at a deeper path.
- **No API secret ⇒ the run is disposable.** The message
  `no Kaggle API secret … checkpoint sync disabled; weights will NOT survive the
  end of this session` (it also fires on a transient startup connection error)
  means no resume and no sync — `/kaggle/working` is wiped between sessions, so
  everything the run learns is lost. Fix the secret and restart; don't let it
  burn quota. The notebook prints `using Kaggle secret <name>` when auth worked.
- **GPU quota: ~30 h/week, ~9–12 h/session.** Not CLI-queryable; a failed push
  saying `Maximum weekly GPU quota … reached` means wait for the rolling reset.

---

## Stage 4 — Pack into `.nnue` (local)

The one-shot path pulls the latest checkpoint and packs it with the **correct**
cp-scale:

```bash
export KAGGLE_API_KEY=<token>
dev/scripts/fetch_latest_net.sh --dataset simbae11/eclipse-checkpoint-sf16
# → data/eclipse.nnue (cp=300)
```

Or do it manually. `from-torch` now defaults `--output-cp-per-unit` to **300**
(it used to default to 410, which silently miscalibrated every eval), and it
reads the layer widths out of the checkpoint, printing them as a receipt:

```bash
python3 dev/scripts/convert_halfkav2_nnue.py from-torch \
    --state-dict data/halfkav2.pt \
    --out data/eclipse.nnue
# checkpoint architecture: 45056 -> 1024x2 -> 16 -> 32 -> 1
# wrote data/eclipse.nnue (88.03 MB, output_cp/unit=300.0)
```

This value is recorded in the `.nnue` header and is what the engine uses to turn
the net's raw output into centipawns at search time — it must match the
training-side `cp_scale`.

### Verify + validate

```bash
ctest --test-dir build -R test_nnue          # loader accepts magic/shape
```

A shape mismatch here means the engine's `kFtOutSize`/`kL1OutSize`/`kL2OutSize`
don't match the net you just packed — see "Switching the engine to a new
architecture" above. The converter also warns if too many weights saturate during
int8/int16 quantization — a few tenths of a percent is fine; >1% means tighten
regularization or lower `lr`.

**Faster ≠ better until the games say so.** Before promoting any net, run an
SPRT / match vs the current net per `dev/RELEASE_CHECKLIST.md`. The sf16 net is
a ~1.6× throughput win over the deployed one on top of the ~1.6× the NEON kernels
already bought, but it is a *different function* trained from scratch, so it has
to beat the incumbent on the board.

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
- **net2widernet, then away from it.** The 2× widen produced the currently
  deployed net and was a dead end (function-preserving ⇒ no strength, 4× the
  cost). Active work is a from-scratch train at `1024x2 → 16 → 32 → 1`.
- **cp-scale 300** (was 410) for more gradient in the typical range; this is now
  the converter's default for `from-torch`.
- **T4 ×2 + DataParallel, AMP fp16, batch 16384**, with per-chunk checkpoint sync
  through `eclipse-checkpoint`.
