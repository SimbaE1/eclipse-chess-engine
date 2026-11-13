# Eclipse — development

Internals, build/test loop, and the net-improvement pipeline. For installing and
*playing* the engine, see [`README.md`](../README.md); for a fresh-machine bootstrap
see [`SETUP.md`](../SETUP.md); for training data + GPU training see
[`KAGGLE.md`](KAGGLE.md).

> **Status (November 2026):** v1.2.0, about 2200–2300 on Lichess. Search,
> NNUE, ponder, Syzygy and Chess960 all work. The active fronts are net quality
> (fine-tuning on quiet, endgame-weighted data; next, piece-count layer stacks)
> and SPRT-gated search tuning.

## How it plays

1. **Priors** — the policy that decides which moves MCTS investigates. Near the
   root (within `PolicyDepth` plies, default 2) every child is scored by the
   NNUE value net in one batched forward pass and softmaxed into priors; deeper
   in the tree, fast MVV-LVA heuristics (SEE-filtered captures/promotions) take
   over to keep nps high. (A separate Lc0-transformer policy net, `src/policy.cpp`
   via the `PolicyFile` option, is available for root info / the MLH "moves-left"
   time-management head, but the in-tree expansion priors come from the value
   net + heuristics.)
2. **Search** — NNUE + MCTS (PUCT, multi-threaded) explores for the best move.
3. **Alpha-beta** — a Lazy-SMP alpha-beta search (`src/ab.cpp`; SEE, qsearch,
   LMR, a lockless 4-way-bucket TT) runs **in parallel** with MCTS on its own
   `AbThreads`. When the two disagree, `src/search.cpp` cross-checks MCTS's move
   with AB, can extend both searches when AB's eval is still drifting with depth,
   and runs a short validation MCTS on AB's move.
4. **Final move** — at the root, AB's move overrides the MCTS visit gate when
   its Q is within `AbOverrideQMargin` of the tree's best (`src/mcts.cpp`). The
   default, `-2.0`, means *always play AB's move*: in the 2026-09-30 runs it beat
   a 0.02 margin by 44-13-11, so MCTS now mainly supplies the second opinion,
   the time it buys AB through pondering, and the move when AB has nothing.

With `go depth` / `go infinite` (no time budget), or when `Threads` ≤
`AbThreads`, the two run **sequentially** instead: MCTS first, then AB in a
small slice of the budget.

**Architecture**

- **Value:** HalfKAv2 `45056 → 2048×2 → 1024 → 256 → 1`, int16/int8 quantized
  (`src/nnue.cpp`), with incremental accumulator updates and AVX-512/AVX2/NEON
  SIMD inference (scalar fallback). The `.nnue` header carries
  `output_cp_per_unit`, the cp↔win-prob scale the net was trained at —
  **currently 300** (see the pipeline note below).
- **Policy:** value-net priors near the root (above); optional Lc0 transformer
  via ONNX Runtime (`src/policy.cpp`) for the MLH "moves-left" head.
- **Search:** MCTS + parallel AB (`src/mcts.cpp`, `src/ab.cpp`,
  `src/search.cpp`); leaf evals are batched (`nnue::evaluate_batch`) and nodes
  come from a pooled allocator capped by `MctsTreeMB`; Syzygy tablebase probing
  at root and interior (`src/syzygy.cpp`). cp↔Q conversions use tanh / clamped
  atanh, so a won position maps near ±1 instead of being squashed. Search
  threads run on 16 MB stacks and AB recursion is capped at `kMaxSearchPly`
  (the old 512 KB default stack overflowed mid-game and crashed the bot).
- **Time mgmt:** MLH-divided budget in `cmd_go` (`src/uci.cpp`), enforced as a
  single absolute deadline across every search phase for both clock and
  movetime searches. The old ~120 ms per-move floor is gone (2026-07-08: the
  AB clock-check stride and the unclamped `go movetime` phases were the bulk
  of it) — measured overshoot at movetime 50–300 is within ±5 ms without the
  ONNX policy net loaded; the root policy probe still adds its inference cost
  per move when `PolicyFile` is set.

### Search tunables (UCI options)

All are runtime `setoption`s, so you can A/B them in a match without rebuilding.

| Option | Default | What |
|---|---|---|
| `Threads` | 4 | **Total** search threads (MCTS + AB) |
| `AbThreads` | derived | AB (Lazy-SMP) threads, carved out of `Threads`. Defaults to `max(1, Threads/4)` until set explicitly; 0 disables AB |
| `Hash` | 256 | AB transposition table (MB) |
| `MctsHash` | 64 | MCTS position-value cache (MB) |
| `MctsTreeMB` | 8192 | Cap on the MCTS node pool (MB). Without it one bot session grew to 48 GB. The bot uses 24576 |
| `Cpuct` | 1.70 | PUCT exploration coefficient (higher = wider) |
| `FpuOffset` | 0.20 | First-Play-Urgency discount on unvisited children |
| `PolicyDepth` | 2 | plies from root that get NNUE-scored priors (`-1` = heuristics everywhere; cost grows ~35×/ply past 2) |
| `SelectVisitFrac` | 0.60 | final root move is best-value among children with ≥ frac·max visits (1.0 = pure most-visited) |
| `SelectQMargin` | 0.02 | Q margin below which selection prefers more visits |
| `AbOverrideQMargin` | -2.0 | AB's root move replaces the tree's pick if `Q(ab) ≥ Q(best) − margin`. -2.0 = always, 0.0 = when strictly better, 2.0 = never |
| `OverrideMargin` | 50 | cp gap at which `search.cpp` treats AB and MCTS as disagreeing (triggers the deeper cross-check / validation) |
| `SyzygyPath` | — | Syzygy tablebase directory |
| `EvalFile` | dev path | NNUE file. The compiled-in default is the developer's checkout, so always set it |
| `PolicyFile` / `PolicyMode` | — / `nnue` | Optional Lc0 ONNX policy net (root info + MLH moves-left time management). `PolicyMode onnx` uses it for priors too (~1000× slower; A/B only) |
| `UCI_Chess960` | false | Chess960 castling notation (king-takes-rook) |

The production bot runs `Threads 8`, `AbThreads 2`, `MctsTreeMB 24576`, 5-man
Syzygy, ponder on, PGO build.

## Repo layout

| Path | What |
|---|---|
| `src/` | Engine (C++20): search, NNUE, policy, UCI, movegen, SEE, TT, Syzygy |
| `tests/` | 14 unit + smoke tests (`ctest`), plus dev tools: `data_filter` (training-data quiet filter), `nnue_eval_server`, `tactic_trace_tool` |
| `dev/scripts/` | Data, training, conversion, and **match/eval tooling** |
| `dev/notebooks/` | `eclipse_wdl_train.ipynb` — the Kaggle training notebook |
| `data/` | Nets (`eclipse.nnue`, `eclipse_v*.nnue` candidates), books, benchmark PGNs (nets are git-ignored; they ship via GitHub Releases) |
| `dev/sprt_runs/` | SPRT output, one directory per run (git-ignored) |
| `scripts/` | User-facing: `pgo_build.sh`, `play_human.py` |
| `extern/` | Vendored deps (e.g. Fathom for Syzygy) |
| [`SETUP.md`](../SETUP.md) | Fresh-machine install → build → run |
| [`KAGGLE.md`](KAGGLE.md) | End-to-end data + GPU training pipeline |

---

## Development loop

Build and test (see [`SETUP.md`](../SETUP.md) for first-time dependencies):

```bash
cmake -S . -B build          # configure (Release by default, -march=native on)
cmake --build build -j        # -> build/src/eclipse + build/tests/test_*
ECLIPSE_NNUE_PATH=$PWD/data/eclipse.nnue ECLIPSE_SYZYGY_PATH=$HOME/syzygy \
  ctest --test-dir build      # 14 tests; the net/TB tests SKIP without those env vars
```

For an assertions-on build (validates the incremental NNUE accumulator against a
from-scratch refresh on every move — invaluable when touching `do_move`/search):

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j && ctest --test-dir build-debug
```

Run a quick standalone search to eyeball nps / depth / score:

```bash
dev/scripts/engine_diag.sh data/eclipse.nnue 3000 4   # <net> [movetime_ms] [threads]
```

Or the built-in fixed-node benchmark (100k nodes from the start position):

```bash
printf 'setoption name EvalFile value data/eclipse.nnue\nsetoption name Threads value 4\nisready\nbench\nquit\n' | ./build/src/eclipse
```

Source map for the hot paths: `search()` orchestration in `src/search.cpp`,
the MCTS worker loop in `src/mcts.cpp`, NNUE inference in `src/nnue.cpp`, UCI +
time management in `src/uci.cpp`.

### Profiling tip

To profile the real search hot path, send a `position` command **after** the net
loads — otherwise the root accumulator is built before the net is present
(`computed=false`) and every leaf takes the cold full-refresh path instead of the
batched one, which is misleading. The `bench` command already does this. Sample
on macOS with `sample <pid> 6 -mayDie`; watch `__psynch_mutexwait` /
`__ulock_wait` for lock contention at high thread counts.

---

## Improvement pipeline

The loop for making the engine stronger, end to end:

```
  data → train (Kaggle GPU) → fetch + pack → SPRT → release
   │          │                    │            │        │
extract   notebook /          fetch_latest   sprt_run  data/eclipse.nnue,
dedup     fine-tune kernel    _net.sh or      .sh       bot config,
filter                        convert (cp=300)          GitHub release
```

### 1. Data + training (Kaggle)

Data engineering runs locally (network-bound), training runs on Kaggle's free
T4s (compute-bound). Full details in [`KAGGLE.md`](KAGGLE.md). Long runs use the
notebook `dev/notebooks/eclipse_wdl_train.ipynb`, which checkpoints to a
per-run Kaggle dataset (`CKPT_DATASET`, e.g. `simbae11/eclipse-checkpoint-quiet-v2`).
Short fine-tunes (the current nets) run as small standalone Kaggle script
kernels that save a `.pt` at each snapshot step.

| Stage | Script |
|---|---|
| Extract Stockfish-labelled positions from Lichess dumps | `dev/scripts/extract_lichess_evals.py` |
| Quiet filter + endgame weighting | `build/tests/data_filter` (from `tests/data_filter.cpp`) |
| Dedup + IID shuffle into HKAV2BIN chunks | `dev/scripts/dedup_and_chunk.py` |
| Train HalfKAv2 | the notebook (`dev/scripts/train_halfkav2.py` is the older standalone copy) |
| Pack to `.nnue` | `dev/scripts/convert_halfkav2_nnue.py` |

### 2. Fetch the latest checkpoint → `data/eclipse.nnue`

One command downloads the newest checkpoint and packs it with the correct
scale:

```bash
export KAGGLE_API_KEY=KGAT_xxxxxxxx          # Bearer token, not stored in repo
dev/scripts/fetch_latest_net.sh --dataset simbae11/eclipse-checkpoint-quiet-v2
#   -> data/eclipse.nnue (cp=300), verified
```

For a fine-tune kernel, download its output (`kaggle kernels output <slug> -p out/`)
and convert the snapshot you want with `convert_halfkav2_nnue.py from-torch`.

It reports the checkpoint's epoch/chunk, converts `halfkav2.pt` with
`--output-cp-per-unit 300`, and verifies the value landed in the file header.

> **The cp_scale field.** The notebook trains at `cp_scale=300`, and
> `convert_halfkav2_nnue.py from-torch` now defaults to 300 to match (it used to
> default to 410, which silently miscalibrated every win-prob and MCTS-Q
> conversion, since the engine reads that field out of the file header at load
> time). If you ever change `TRAIN_CFG['cp_scale']`, change the converter default
> with it or pass `--output-cp-per-unit` explicitly.
>
> The converter reads the **layer widths** from the checkpoint's own tensor
> shapes, so it handles both the deployed `2048x2 -> 1024 -> 256` net and the
> `1024x2 -> 16 -> 32` retrain without a flag. The engine is the half that does
> *not* adapt — see "Switching the engine to a new architecture" in
> `dev/KAGGLE.md`.

### 3. Test the candidate: SPRT

Every net and every search change has to earn its place in an **SPRT**
(sequential probability ratio test) self-play match. `dev/scripts/sprt_run.sh`
wraps cutechess-cli and stops by itself once the result is decided
(LLR ≥ +2.94 → H1, the candidate is stronger; LLR ≤ −2.94 → H0, it isn't).

One-time setup: build cutechess-cli and put it on `PATH` (background jobs need
the `export` in their own script), and fetch an imbalanced opening book so
near-equal engines produce decisive games:

```bash
curl -sL -o /tmp/uho.zip \
  https://github.com/official-stockfish/books/raw/master/UHO_4060_v2.epd.zip
unzip -o /tmp/uho.zip -d data/books/      # sprt_run.sh finds it automatically
```

**A net:** pass the *same* binary twice and give each side its own net, so the
code is held exactly constant:

```bash
dev/scripts/sprt_run.sh build/src/eclipse build/src/eclipse \
    -t 60+0.6 -0 -5 -1 5 -c 4 -T 2 -A 1 -M 2048 \
    -N option.EvalFile=data/eclipse_candidate.nnue \
    -O option.EvalFile=data/eclipse.nnue \
    -l cand_vs_cur
```

**A code change:** build the baseline into a second directory (or a worktree)
and pass both binaries. `-T/-A` set MCTS/AB threads per engine; use `-T 2 -A 1`
or more to run AB in parallel like the bot does, and budget `-c` as roughly
cores ÷ threads-per-engine. `-s` runs a 4-game smoke test first, and
`SPRT_DRYRUN=1` prints the cutechess command without running it.

Things that have bitten us:

- `-N`/`-O` overrides of a key that is also in the shared options used to be
  silently overridden by cutechess's `-each`. That's fixed (the harness now
  moves such keys out of `-each`), but check the run's header lines show the
  per-side `EvalFile` you expect.
- Bounds: `-0 0 -1 5` (default) asks "is it ≥5 Elo better"; `-0 -5 -1 5` is the
  symmetric head-to-head used to pick between two candidates.
- A run that hits the game cap without deciding is a tie. Between two
  candidates that is still a choice: take the higher point estimate.
- Offline metrics (val loss, MAE on held-out positions, ACPL on our own games)
  have repeatedly disagreed with SPRT results. Use them to pick what to test,
  never to decide.

Output goes to `dev/sprt_runs/<timestamp>_<label>/` (`games.pgn`, `log.txt`).
`python3 dev/scripts/sprt_status.py` prints one line per run with the current
score, Elo and LLR.

**Before a run, stop the Lichess bot if it's on the same machine** (it competes
for cores). The bot's LaunchAgent is re-bootstrapped hourly by the training
babysitter, so a full stop needs `launchctl bootout` **and** `launchctl
disable`. A SIGINT or plain bootout forfeits the game in progress.

#### Quick, unscored matches

For eyeballing behaviour rather than deciding anything, `run_match.sh` plays a
fixed number of games and the PGN readers summarise them:

```bash
dev/scripts/run_match.sh \
  --net1 data/eclipse_candidate.nnue --net2 data/eclipse.nnue \
  --name1 cand --name2 cur --tc 5+3 --games 40 \
  --book data/books/UHO_4060_v2.epd --pgn /tmp/cand_vs_cur.pgn
python3 dev/scripts/match_score.py /tmp/cand_vs_cur.pgn    # running W/D/L
python3 dev/scripts/match_depth.py /tmp/cand_vs_cur.pgn    # depth + time/move per net
```

### 4. Promote the winner

Copy the net into place, point the bot's `EvalFile` at it
(`~/lichess-bot/config-eclipse.yml`), and ship it as a release asset per
[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md):

```bash
cp data/eclipse_candidate.nnue data/eclipse.nnue
```

---

## Testing & evaluation toolkit (reference)

| Script | Purpose |
|---|---|
| `dev/scripts/sprt_run.sh` | SPRT self-play harness (the deciding test); `-N`/`-O` per-side options, `-s` smoke |
| `dev/scripts/sprt_status.py` | One-screen progress of every run under `dev/sprt_runs/` |
| `dev/scripts/run_match.sh` | cutechess launcher for fixed-length matches; `--book`, `--debug`, `--dry-run` |
| `dev/scripts/match_score.py` | Per-game results + running W/D/L score from a PGN |
| `dev/scripts/match_depth.py` | Per-net avg/max depth and time-per-move from PGN comments |
| `dev/scripts/analyze_accuracy.py` | Stockfish accuracy% + ACPL per game (cached; `--no-cache`) |
| `dev/scripts/engine_diag.sh` | Live nps/depth/seldepth from one standalone search |
| `dev/scripts/ponderhit_stats.sh` | Ponderhit count + avg nps from a cutechess `-debug` log |
| `dev/scripts/fetch_latest_net.sh` | Download latest Kaggle checkpoint → `data/eclipse.nnue` (cp=300) |
| `dev/scripts/bench_vs_stockfish.py` | Calibrate strength vs Stockfish at a target Elo |

`watch` keeps any of the PGN readers live:

```bash
watch -n 30 'python3 dev/scripts/match_score.py /tmp/cand_vs_cur.pgn'
```
