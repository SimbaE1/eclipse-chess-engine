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
| Quiet filter + endgame weighting        | `tests/data_filter.cpp` → `build/tests/data_filter` |
| Dedup + IID shuffle → HKAV2BIN chunks   | `dev/scripts/dedup_and_chunk.py`         |
| Text → binary chunks (old, sequential)  | `dev/scripts/preprocess_halfkav2.py`     |
| Trainer (older standalone copy)         | `dev/scripts/train_halfkav2.py`          |
| Short fine-tune kernel (current nets)   | `dev/jobs/ft_steps3/push/steps.py`       |
| Session carry-over babysitter           | `dev/scripts/train_babysit.sh`           |
| Quantizer / `.nnue` packer              | `dev/scripts/convert_halfkav2_nnue.py`   |
| One-shot "pull latest + pack"           | `dev/scripts/fetch_latest_net.sh`        |
| The notebook (self-contained)           | `dev/notebooks/eclipse_wdl_train.ipynb`  |
| Kaggle launch mechanics (historical)    | `dev/net2widernet.md`                    |

`dev/net2widernet.md` describes the 2× widen that produced today's
`2048x2 → 1024 → 256` shape and is superseded — its Kaggle mechanics (§1, §4, §6)
are still correct, its plan is not. **This file is the live procedure.**

> **Where things stand (November 2026).** Every shipped net since v1.1.0 is the
> wide `2048x2 → 1024 → 256` shape, *warm-started* from the previous net rather
> than trained from scratch. The lineage is ep2c9 → ep2c44 (v39 warm restart,
> v1.1.0) → `eclipse_v10_mix20` (v1.2.0): a **2,000-step** fine-tune on 80%
> quiet/endgame-weighted + 20% unfiltered data. Short fine-tunes beat the full
> 2-epoch runs on the board (see [Fine-tuning](#fine-tuning-what-the-sweeps-showed)).
> The October `1024x2 → 16 → 32` from-scratch run (`eclipse-checkpoint-sf16`)
> finished but was never SPRT'd. Next on the list: piece-count **layer stacks**
> (7 per-bucket heads on the shared feature transformer).

The notebook is **self-contained** (the trainer code is copy-pasted in), so it
runs on Kaggle with only datasets attached — no `import train_halfkav2`. When a
script changes in a way that affects the net, mirror the change into the
notebook (or it silently trains something else).

---

## Pipeline at a glance

```
Local (Mac)                                         Kaggle
──────────────────────────────────────────          ───────────────────────────────────
lichess_db_*.pgn.zst
  │ extract_lichess_evals.py  (harvest %eval)
  ▼
eval_*.txt   <fen>;<cp>
  │ data_filter   (drop in-check / SEE>0 capture /
  │                promotion; weight 7-18 men ×2-3)
  ▼
<fen>;<cp>;<w>
  │ dedup_and_chunk.py  (global dedup, IID       upload   eclipse-chunks-quiet (194 chunks,
  ▼                      chunks, leak-free val)  ─────►     val_holdout, val_legacy,
eval_chunk_NNN.bin  (HKAV2BIN, 3.9M pos each)               init_weights.pt)
                                                            eclipse-chunks-a (unfiltered)
                                                                  │
                                                            notebook (long run) or
                                                            fine-tune kernel (short)
                                                            T4 ×2, warm start, AMP
data/eclipse_v*.nnue ◄── convert_halfkav2_nnue.py ◄──────  halfkav2.pt / ft_*.pt
                         (cp=300)
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

## Stage 1c — Quiet filter, dedup and IID chunks (local, current pipeline)

The current datasets are built with two extra steps instead of the sequential
`preprocess_halfkav2.py` split above:

1. **`data_filter`** (C++, built with the tests) drops positions that are a
   poor *static-eval* target: side to move in check, any legal promotion, or a
   legal capture with SEE > 0. Eclipse plays AB's root move, and AB only asks
   the net about positions qsearch has already resolved, so a label that depends
   on a pending tactic is noise. It also tags each line with a sampling weight:
   ×3 for ≤12 men, ×2 for 13–18 men (where our own games showed most costly
   errors), ×1 otherwise and ×1 inside tablebase range.
2. **`dedup_and_chunk.py`** deduplicates by net input (`<board> <stm>`, which
   removes the millions of copies of common opening positions), expands
   weights *after* dedup, scatters records IID across chunks, and routes a
   crc32 hash slice to a leak-free `val_holdout.bin`.

```bash
gzcat data/eval_training.txt.gz | build/tests/data_filter \
  | gzip -1 > data/eval_training_quiet.txt.gz
python3 dev/scripts/dedup_and_chunk.py --help    # chunk count, holdout-mod, workers
```

---

## Stage 2 — Upload to Kaggle

The chunks go up **RAW** (Kaggle does not decompress on their end):

```bash
export KAGGLE_API_KEY=<your KGAT_… token>
kaggle datasets version -p data/chunks -m "rebuild chunks" --dir-mode tar
# (first time: kaggle datasets create -p data/chunks  → slug simbae11/eclipse-chunks-a)
```

- **`simbae11/eclipse-chunks-quiet`** — the current training set: 194 filtered,
  weighted, deduplicated chunks, plus `val_holdout.bin`, `val_legacy.bin` (a
  sample of the old unfiltered distribution, to catch forgetting) and
  `init_weights.pt` (ep2c44, dequantized byte-exact from its `.nnue`) for warm
  starts.
- **`simbae11/eclipse-chunks-a`** — the older unfiltered chunks (192 × 3.9M).
  The fine-tune kernels mix a fraction of these into every batch; both datasets
  use `eval_chunk_NNN.bin` names, so scripts must match on the dataset slug in
  the mount path.
- **Checkpoint datasets, one per run** (`CKPT_DATASET` + `RUN_ID` in the
  notebook): `eclipse-checkpoint-quiet-v2` is the latest. Never reuse a slug
  for a new run, and never mount `eclipse-checkpoint-quiet` (it holds the bad
  v48 checkpoint). `eclipse-checkpoint` holds the original wide net and
  `eclipse-checkpoint-sf16` the from-scratch narrow run; neither should be
  written to.
- **Making a dataset public can't be undone** (Kaggle has no public → private).
  Decide per dataset before you flip it.

Dataset uploads do **not** consume GPU quota — only kernel runs do. After
uploading you can gzip the local `.bin` + source text to reclaim disk.

---

## Stage 3 — Train on Kaggle GPU

This is the **long-run** path (the notebook, one or more 12 h sessions). For
the short fine-tunes that currently produce the best nets, see
[Fine-tuning](#fine-tuning-what-the-sweeps-showed) below.

1. Set `RUN_ID`, `CKPT_DATASET`, `ARCH` and `EPOCHS` in the configuration cell of
   `dev/notebooks/eclipse_wdl_train.ipynb`. `RUN_ID` is written into
   `resume_state.pt` and must match on resume, so a session can never pick up
   another run's weights.
2. **Data**: `kernel-metadata.json` attaches `eclipse-chunks-quiet`. Push with
   `kaggle kernels push -p dev/notebooks`.
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
5. **Secrets (UI-launched runs only).** On a **CLI-pushed** kernel the secret is
   *never* readable (`UserSecretsClient` → "Connection error"), so the notebook
   falls back to carry-over through the babysitter (next section). For a run
   started from the Kaggle UI: add **`KAGGLE_API_KEY`** (your Kaggle bearer token; the
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
(`CKPT_DATASET`, currently `eclipse-checkpoint-quiet-v2`) via the Kaggle API:

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

So UI-launched sessions are fully resumable with no manual re-attaching.

### Carry-over without the secret (CLI-pushed runs)

Because a CLI push can't read the secret, the notebook instead:

- stops itself cleanly at `SESSION_MAX_S` (11 h), before Kaggle's 12 h kill,
  printing `=== SESSION STOP (...)`; it prints `All done!` only when the run is
  actually complete;
- on start, falls back to the **mounted** `CKPT_DATASET` (only paths under that
  exact slug) when the API download isn't available; a mounted-but-empty
  dataset is a hard error.

`dev/scripts/train_babysit.sh` (run hourly by a LaunchAgent) does the rest:
when a session stops it copies `halfkav2.pt` + `resume_state.pt` from the kernel
output into `CKPT_DATASET`, waits until Kaggle shows the new files (by
`creationDate`, since sizes are identical between versions), mounts the dataset
and re-pushes. It never resumes a kernel that was cancelled or died with a
Traceback. `BABYSIT_*` environment variables override its settings for tests.

**If a run has no carry-over at all (no sync line, no babysitter), never cancel
it mid-way** — wait for completion, or everything it learned is lost.

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
(`GPU 0/1: Tesla T4`), the chunk count (`found 194 premade chunk(s)` for chunks-quiet), the
architecture, and `LR schedule: N chunks/epoch × E epochs = T cosine steps`.
**Check them before walking away** — the run does not fail on any of these, it
just quietly produces less than you asked for.

`kaggle kernels status <slug>` gives just the state
(`RUNNING` / `COMPLETE` / `ERROR`).

### The sf16 from-scratch run (October 2026, untested)

`ARCH = 'sf16'` trains Stockfish's own shape, `1024x2 → 16 → 32 → 1`, from
scratch. The wide net is slow for its size (36.5 µs per `nnue::evaluate()`;
the feature transformer output is ~1.2% nonzero), and the FT→L1 product is
evaluated sparsely, so shrinking L1 from 1024 to 16 is the largest speedup
available:

| Net | nps (fixed 100k-node `bench`, Apple silicon) |
|---|---|
| `2048x2 → 1024 → 256`, scalar kernels | 53k–64k |
| `2048x2 → 1024 → 256`, NEON kernels | 85k–96k |
| `1024x2 → 16 → 32`, NEON kernels (random weights) | 142k–157k |

The run finished (`eclipse-checkpoint-sf16`) but has not been SPRT'd against
the wide nets. Since then the warm-restart fine-tunes have added ~+100 Elo to
the wide lineage, so the bar it has to clear has moved.

### Network architecture

Single-logit win-probability head (`sigmoid(out) = win prob`):

```
original:            features(HalfKAv2 45056) → 1024×2 → 512  → 128 → 1
deployed ('wide'):   features(HalfKAv2 45056) → 2048×2 → 1024 → 256 → 1
'sf16' (untested):   features(HalfKAv2 45056) → 1024×2 →   16 →  32 → 1
```

The notebook's `ARCH` preset (`'wide'` or `'sf16'`) sets `FT_OUT/L1_OUT/L2_OUT`,
which are the contract with the C++ engine. The
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

Flipping them makes the deployed 189 MB net stop loading, so build the narrow
engine into a separate directory (e.g. `build-sf16/`) for testing and SPRT the
two *binaries* against each other, each with its own net. Note the SPRT harness
bug fixed in November 2026: before the fix, a per-side `EvalFile` override was
silently replaced by the shared one, so the earlier small-vs-wide SPRTs ran
without a usable net and are invalid.

### Hyperparameters (current `TRAIN_CFG`, warm restart)

| Knob            | Value   | Notes |
|-----------------|---------|-------|
| optimizer       | Adam, `weight_decay=1e-4`, `grad_clip=1.0` | |
| `lr`            | `5e-5`  | **Warm-restart** peak, cosine down to `lr/10`. A 1,000-step GPU sweep from ep2c44 picked it: 5e-5 beat 1.5e-4, 4e-4 and 1e-3 on every val bucket and on unseen SF game positions. `1e-3` is the *from-scratch* value; using it on a warm restart (v48) wasted a 9 h session. |
| `batch_size`    | 16384   | DataParallel splits 8192/GPU across the two T4s. |
| `warmup_steps`  | 500     | Linear warmup before the cosine schedule. |
| LR schedule     | Cosine  | For from-scratch runs the oversized `T_max` (under-annealing) is deliberate: too low an LR made the net regress. |
| `cp_scale`      | 300     | `win_prob = sigmoid(cp/300)`. Must match the chunk builder's `--cp-scale`. |
| `max_cp`        | 4000    | Drops near-mate noise. Must match the chunk builder's `--max-cp`. |
| `EPOCHS`        | 2       | 2 × 194 chunks ≈ 7 h on T4 ×2 with the in-RAM loader. |
| `qat`           | True    | Fake-quantize to the engine's int8/int16 scales while training. |
| precision       | AMP fp16 + GradScaler | |
| Loss            | `BCEWithLogits` on the win-prob target | |
| data loader     | whole chunk in RAM, vectorized batches | The per-sample memmap loader page-faulted on `/kaggle/input` at ~2 s/step; in-RAM is ~256 ms/step (63k pos/s). |

Per chunk the run reports `train`, `val` (quiet holdout), `val_legacy`, val **by
piece count** (`men[<=12 13-18 19-25 26+]`), lr, step and pos/s. Read the
buckets and `val_legacy`, not just `val`: the unweighted quiet val hides
endgame gains, and `val_legacy` is where forgetting shows.

### Fine-tuning: what the sweeps showed

The best nets now come from **short** fine-tunes run as standalone Kaggle script
kernels (`dev/jobs/ft_steps*/push/steps.py`, kernels `simbae11/eclipse-ft-*`).
Each kernel trains several configs from `init_weights.pt` in one ~1 h session
and saves `ft_<config>_s<step>.pt` at chosen steps. With a constant LR one run
gives many step counts; a cosine schedule depends on its total length, so each
decay length is its own config.

What the November 2026 sweeps found (all from ep2c44, lr 5e-5, batch 16384):

| Net | Data | Steps | Result |
|---|---|---|---|
| smoke | 100% quiet | 1,000 | **+103** vs ep2c44 (179 games, stopped) |
| quiet-v2 final | 100% quiet | 2 epochs (~48k) | +109 vs ep2c44 (SPRT H1), but **−41 vs smoke** |
| quiet_s500 | 100% quiet | 500 | −45 vs smoke (H0) |
| mix50_s2000 | 50% quiet / 50% unfiltered | 2,000 | −37 vs smoke (H0) |
| mix10_s1000 | 90 / 10 | 1,000 | −11.5 ± 12.7 vs smoke (2,000-game cap) |
| **mix20_cos2000** (v1.2.0) | 80 / 20, cosine | 2,000 | +1.4 ± 12.8 vs smoke (2,000-game cap) |

- The quiet-data gain is complete by ~1,000 steps. After that, pure-quiet
  training only *forgets* tactics: tactical-position MAE goes 183 (ep2c44) →
  208 at 1k steps → ~231 at the end of the long run.
- Mixing in unfiltered data stops the forgetting. mix50 held tactical MAE at
  ep2c44's level for 8,000 steps, but lost some of the quiet gain and lost on
  the board. 20% keeps both.
- Offline metrics (val loss, MAE, ACPL on our own games) ranked these nets
  differently from SPRT more than once. Use them to choose what to SPRT.



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
- **The Kaggle CLI needs the token in its environment** (`KAGGLE_API_TOKEN`).
- **The notebook overwrote fixes once.** v48 regressed three fixes v39 had
  (warm-restart lr, checkpoint carry-over, fast loader) because a different
  notebook version was pushed over it. Before trusting a run, check its first
  lines for the lr, the carry-over mode and the step time.
- **GPU quota: ~30 h/week, ~9–12 h/session.** Not CLI-queryable; a failed push
  saying `Maximum weekly GPU quota … reached` means wait for the rolling reset.

---

## Stage 4 — Pack into `.nnue` (local)

The one-shot path pulls the latest checkpoint and packs it with the **correct**
cp-scale:

```bash
export KAGGLE_API_KEY=<token>
dev/scripts/fetch_latest_net.sh --dataset simbae11/eclipse-checkpoint-quiet-v2
# → data/eclipse.nnue (cp=300)
```

Or do it manually. `from-torch` now defaults `--output-cp-per-unit` to **300**
(it used to default to 410, which silently miscalibrated every eval), and it
reads the layer widths out of the checkpoint, printing them as a receipt:

```bash
python3 dev/scripts/convert_halfkav2_nnue.py from-torch \
    --state-dict data/halfkav2.pt \
    --out data/eclipse.nnue
# checkpoint architecture: 45056 -> 2048x2 -> 1024 -> 256 -> 1
# wrote data/eclipse.nnue (~189 MB, output_cp/unit=300.0)
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

**Better val ≠ better play until the games say so.** Before promoting any net,
SPRT it against the current one (see "Test the candidate: SPRT" in
[`DEVELOPMENT.md`](DEVELOPMENT.md)).

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
- **net2widernet.** The 2× widen produced the wide shape every shipped net
  still uses. By itself it was function-preserving (no strength, 4× the cost);
  the strength came from the training after it.
- **Warm restarts on quiet, endgame-weighted data (Aug–Nov 2026).** Deduped IID
  chunks, an in-check filter (v39 → ep2c44), then the `data_filter` quiet set
  and short fine-tunes (v1.2.0).
- **cp-scale 300** (was 410) for more gradient in the typical range; this is now
  the converter's default for `from-torch`.
- **T4 ×2 + DataParallel, AMP fp16, batch 16384**, with checkpoint carry-over
  through a per-run `CKPT_DATASET`.
