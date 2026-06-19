# net2widernet — Kaggle launch runbook

The exact, verified procedure to push the datasets + notebook to Kaggle and
kick off the **net2widernet** training run (function-preserving 2× widen of the
deployed net). Written for the **GPU quota reset (5 PM today)** — everything that
can be done before the reset is already done; at 5 PM you run **one command**.

> ⚠️ **Never commit the Kaggle token.** This repo is public. The token lives only
> in your shell as `$KAGGLE_API_TOKEN` (and as a Kaggle *Secret* in the notebook).
> Every command below assumes you have exported it first:
> ```bash
> export KAGGLE_API_TOKEN=<your KGAT_… token>
> ```

---

## 0. State as of this writing (all verified ✅)

| Thing | State | Needed for the run |
|---|---|---|
| `simbae11/eclipse-chunks-a` (7 premade HKAV2BIN chunks) | **ready** | training data |
| `simbae11/eclipse-checkpoint` (`halfkav2.pt` 189 MB = **narrow** net, `resume_state.pt`) | **ready** | warm-start source — the narrow shape is what makes the widen fire |
| `dev/notebooks/eclipse_wdl_train.ipynb` | `NET2WIDER=True`, FT 1024→2048 / L1 512→1024 / L2 128→256 | the trainer |
| `dev/notebooks/kernel-metadata.json` | `id: simbae11/tcec-chess-engine`, `enable_gpu: true`, `dataset_sources: [eclipse-chunks-a, eclipse-checkpoint]` | push target |
| GPU quota | **exhausted until ~5 PM** | the only blocker |

**So the datasets do NOT need re-uploading.** The single action at 5 PM is the
**notebook push**, which on Kaggle is *save + Run All* — i.e. it starts training.

---

## 1. At 5 PM — pre-flight (10 seconds)

Confirm the quota is actually back. This is a read-only call (no quota cost):

```bash
kaggle kernels status simbae11/tcec-chess-engine
```

You can't query remaining GPU hours from the CLI, so the real test is the push
itself (next step). If it fails with `Maximum weekly GPU quota … reached`, the
window hasn't rolled over yet — wait 10–15 min and retry.

---

## 2. Launch the run (the one command)

```bash
cd /Users/ezra/eclipse-chess-engine
kaggle kernels push -p dev/notebooks/
```

`kernels push` **saves the notebook AND runs it** (Save & Run All) — that is what
starts net2widernet training. A successful push prints the kernel URL.

> **Path note:** it's `dev/notebooks/` now (the repo was restructured); the old
> `notebooks/` path no longer exists.

---

## 3. Confirm it's training

```bash
kaggle kernels status simbae11/tcec-chess-engine        # -> RUNNING
```

Or open the kernel: <https://www.kaggle.com/code/simbae11/tcec-chess-engine>.
In the first ~minute of logs you should see, in order:

1. `found N premade chunk(s): [0, 1, 2, …]`
2. `Resuming weights from …/halfkav2.pt`
3. `net2widernet: widening FT 1024->2048, L1 512->1024, L2 128->256 (function-preserving)`
4. `net2widernet: fresh optimizer/schedule for the widened net`
5. training step/loss lines

If you see (3), the widen fired correctly. If instead it says it resumed a wide
checkpoint, that means a previous wide run already overwrote the checkpoint — see
§6.

---

## 4. Why this works (the dependency that matters)

net2widernet is a **one-time warm start**. The notebook (`search()`-side cell)
widens the net only when the checkpoint it loads is the **narrow** shape
(`ft.weight == [1024, 45056]`). Right now `eclipse-checkpoint/halfkav2.pt` is that
narrow net, so the widen fires, then training continues from the
function-preserving 2× net.

**Cross-run checkpoint sync** is automatic: the notebook's cell 6 pulls
`halfkav2.pt` + `resume_state.pt` from `eclipse-checkpoint` at start and pushes a
new version after every ~100 M-position chunk, using the Kaggle **Secret** named
`KAGGLE_API_TOKEN`. That secret is already configured (prior runs updated the
checkpoint dataset). If a run ever prints `no KAGGLE_API_TOKEN secret … training
from scratch`, re-add it: notebook → **Add-ons → Secrets → `KAGGLE_API_TOKEN`**.

---

## 5. After the first chunk saves (important)

Once the run saves its first wide checkpoint, `eclipse-checkpoint` now holds the
**wide** net. From then on:

- **Re-running is safe** — it resumes the wide net (the widen branch won't
  re-fire because the shape no longer matches narrow). Subsequent ~9 h Kaggle
  sessions just `kaggle kernels push -p dev/notebooks/` again to continue.
- **Do NOT** re-upload the narrow net over the checkpoint, or you'd discard wide
  training progress.

---

## 6. (Only if needed) re-pushing datasets

You normally skip this — both datasets are already up. Use these only if a file
changed.

**Premade chunks** (local copies still in `data/chunks/`, ~89 GB):
```bash
kaggle datasets version -p data/chunks -m "rebuild chunks" --dir-mode tar
```

**Restart net2widernet from scratch** (re-arm the narrow warm start): restore the
narrow `halfkav2.pt` as a new version of `eclipse-checkpoint` (and drop the
`resume_state.pt` so it doesn't resume mid-schedule), then push the notebook.
Keep a copy of the narrow net before the first wide run overwrites it if you want
this option later.

---

## 7. After training — deploy the wide net (separate task)

When the run has converged (watch `val=` in the logs / the checkpoint history):

1. Pull + pack the net locally:
   ```bash
   export KAGGLE_API_TOKEN=<token>
   dev/scripts/fetch_latest_net.sh        # downloads latest checkpoint -> data/eclipse.nnue (cp=300)
   ```
2. **The wide net will NOT load into the current engine until the C++ side is
   widened to match** — this is the gate before any strength test:
   - `kFtOutSize` in `src/nnue.hpp` (1024 → 2048)
   - accumulator width + SIMD ladders in `src/nnue.cpp` / `src/accumulator.hpp`
   - `dev/scripts/convert_halfkav2_nnue.py` layer sizes
   Rebuild, run `ctest`, then load the net.
3. **Validate before promoting** — SPRT / match vs the current `ep2c9` net; only
   ship if it's clearly stronger (see `dev/RELEASE_CHECKLIST.md`). Bigger ≠ better
   until the games say so.

---

## Quick reference

```bash
export KAGGLE_API_TOKEN=<your KGAT_… token>

# at 5 PM, from repo root:
kaggle kernels push -p dev/notebooks/          # save + run = start training
kaggle kernels status simbae11/tcec-chess-engine   # RUNNING?

# each later session, to continue:
kaggle kernels push -p dev/notebooks/
```

**Troubleshooting**
- `Maximum weekly GPU quota … reached` → window hasn't reset; wait and retry.
- push succeeds but no `net2widernet: widening …` line → checkpoint is already
  wide (a prior run advanced it); that's fine, it's resuming. §6 to re-arm.
- `no KAGGLE_API_TOKEN secret` in logs → add the notebook Secret (§4).
