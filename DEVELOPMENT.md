# Eclipse — development

Internals, build/test loop, and the net-improvement pipeline. For installing and
*playing* the engine, see [`README.md`](README.md); for a fresh-machine bootstrap
see [`SETUP.md`](SETUP.md); for training data + GPU training see
[`KAGGLE.md`](KAGGLE.md).

> **Status:** late proof-of-concept. The search, NNUE, policy, ponder, and
> Syzygy paths all work; strength tuning and the training-data buildout are the
> active fronts.

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
3. **Tactics** — NNUE + alpha-beta runs alongside MCTS; when it finds a tactic
   the MCTS missed, it raises that move's Q so the tree adopts it
   (`adjust_root_q` / reconciliation in `src/search.cpp`).

**Architecture**

- **Value:** HalfKAv2 `45056 → 1024×2 → 512 → 128 → 1`, int16/int8 quantized
  (`src/nnue.cpp`), with incremental accumulator updates and AVX-512/AVX2/NEON
  SIMD inference (scalar fallback). The `.nnue` header carries
  `output_cp_per_unit`, the cp↔win-prob scale the net was trained at —
  **currently 300** (see the pipeline note below).
- **Policy:** value-net priors near the root (above); optional Lc0 transformer
  via ONNX Runtime (`src/policy.cpp`) for the MLH "moves-left" head.
- **Search:** MCTS with an AB verifier (`src/mcts.cpp`, `src/ab.cpp`,
  `src/search.cpp`); leaf evals are batched (`nnue::evaluate_batch`) and nodes
  come from a per-thread pooled allocator; Syzygy tablebase probing at root and
  interior (`src/syzygy.cpp`).
- **Time mgmt:** MLH-divided budget in `cmd_go` (`src/uci.cpp`). Note the fixed
  ~120 ms per-move floor documented there — ultra-bullet (increment < floor)
  can flag; 5+3 and slower are safe.

### Search tunables (UCI options)

All are runtime `setoption`s, so you can A/B them in a match without rebuilding.

| Option | Default | What |
|---|---|---|
| `Threads` | 1 | MCTS worker threads |
| `AbThreads` | 1 | alpha-beta verifier threads (0 disables) |
| `Hash` | 256 | AB transposition table (MB) |
| `MctsHash` | 64 | MCTS position-value cache (MB) |
| `Cpuct` | 1.70 | PUCT exploration coefficient (higher = wider) |
| `FpuOffset` | 0.20 | First-Play-Urgency discount on unvisited children |
| `PolicyDepth` | 2 | plies from root that get NNUE-scored priors (`-1` = heuristics everywhere; cost grows ~35×/ply past 2) |
| `SelectVisitFrac` | 0.60 | final root move is best-value among children with ≥ frac·max visits (1.0 = pure most-visited) |
| `SelectQMargin` | 0.02 | Q margin below which selection prefers more visits |
| `SyzygyPath` | — | Syzygy tablebase directory |
| `EvalFile` / `PolicyFile` | — | NNUE / ONNX policy paths |

## Repo layout

| Path | What |
|---|---|
| `src/` | Engine (C++20): search, NNUE, policy, UCI, movegen, Syzygy |
| `tests/` | Unit + smoke tests (`ctest`) |
| `scripts/` | Data, training, conversion, and **match/eval tooling** |
| `notebooks/` | `eclipse_wdl_train.ipynb` — the Kaggle training notebook |
| `data/` | Nets (`eclipse.nnue`, `policy.onnx`), books, benchmark PGNs (nets are git-ignored; they ship via GitHub Releases) |
| `extern/` | Vendored deps (e.g. Fathom for Syzygy) |
| [`SETUP.md`](SETUP.md) | Fresh-machine install → build → run |
| [`KAGGLE.md`](KAGGLE.md) | End-to-end data + GPU training pipeline |

---

## Development loop

Build and test (see [`SETUP.md`](SETUP.md) for first-time dependencies):

```bash
cmake -S . -B build          # configure (Release by default, -march=native on)
cmake --build build -j        # -> build/src/eclipse + build/tests/test_*
ctest --test-dir build        # full unit + smoke suite (should be 100%)
```

For an assertions-on build (validates the incremental NNUE accumulator against a
from-scratch refresh on every move — invaluable when touching `do_move`/search):

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j && ctest --test-dir build-debug
```

Run a quick standalone search to eyeball nps / depth / score:

```bash
scripts/engine_diag.sh data/eclipse.nnue 3000 4   # <net> [movetime_ms] [threads]
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
  data → train (Kaggle GPU) → fetch + pack → test → promote
   │          │                    │           │        │
extract   notebook            fetch_latest   run_match  copy winner
lichess   eclipse_wdl_train   _net.sh        + analyze  to data/eclipse.nnue
          .ipynb              (cp=300)        accuracy
```

### 1. Data + training (Kaggle)

Data engineering runs locally (network-bound), training runs on Kaggle's free
T4s (compute-bound). Full details in [`KAGGLE.md`](KAGGLE.md); the notebook is
`notebooks/eclipse_wdl_train.ipynb`. Each chunk it trains is checkpointed to the
Kaggle dataset `simbae11/eclipse-checkpoint` as `halfkav2.pt` +
`resume_state.pt`.

| Stage | Script |
|---|---|
| Extract Lichess positions | `scripts/extract_lichess_wdl.py`, `scripts/sample_lichess.py` |
| (Optional) Stockfish labels | `scripts/label_with_stockfish.py` |
| Train HalfKAv2 | `scripts/train_halfkav2.py` (mirrored in the notebook) |
| Pack to `.nnue` | `scripts/convert_halfkav2_nnue.py` |

### 2. Fetch the latest checkpoint → `data/eclipse.nnue`

One command downloads the newest checkpoint and packs it with the correct
scale:

```bash
export KAGGLE_API_TOKEN=KGAT_xxxxxxxx        # Bearer token, not stored in repo
scripts/fetch_latest_net.sh                  # -> data/eclipse.nnue (cp=300), verified
```

It reports the checkpoint's epoch/chunk, converts `halfkav2.pt` with
`--output-cp-per-unit 300`, and verifies the value landed in the file header.

> **⚠ The cp_scale=300 gotcha.** The notebook trains at `cp_scale=300`, but
> `convert_halfkav2_nnue.py` still **defaults to 410**. A bare conversion bakes
> 410 into the `.nnue`, and the engine reads that field at load time for every
> win-prob and MCTS-Q conversion — silently miscalibrating the net. Always pass
> `--output-cp-per-unit 300` (which `fetch_latest_net.sh` does for you).

### 3. Test the candidate vs the current net

Play the candidate against the incumbent. Use the imbalanced opening book so
two near-equal nets produce decisive games instead of a draw-fest (fair because
each opening is played with both colors via `-repeat`):

```bash
# one-time: get the UHO book
curl -sL -o /tmp/uho.zip \
  https://github.com/official-stockfish/books/raw/master/UHO_4060_v2.epd.zip
unzip -o /tmp/uho.zip -d data/books/

scripts/run_match.sh \
  --net1 data/eclipse_candidate.nnue --net2 data/eclipse.nnue \
  --name1 cand --name2 cur \
  --tc 5+3 --games 40 \
  --book data/books/UHO_4060_v2.epd \
  --pgn /tmp/cand_vs_cur.pgn
```

Watch and analyze (all read the live PGN safely — cutechess only appends a game
once it finishes):

```bash
python3 scripts/match_score.py /tmp/cand_vs_cur.pgn          # results + running score
python3 scripts/match_depth.py /tmp/cand_vs_cur.pgn          # depth + time/move per net
python3 scripts/analyze_accuracy.py /tmp/cand_vs_cur.pgn \   # SF accuracy + ACPL (cached)
    --stockfish "$(which stockfish)" --depth 15
```

`analyze_accuracy.py` caches per game (keyed by PGN path + game + depth), so
re-running on a growing match only evaluates new games. For ponderhits / nps,
launch with `scripts/run_match.sh --debug ...` and read the log:

```bash
scripts/ponderhit_stats.sh /tmp/cand_vs_cur_debug.log
```

> **Reading the numbers.** Accuracy% saturates near the top and is a poor
> discriminator between two strong, similar nets — and it's inflated by a
> shallow SF reference and drawish balanced games. Trust the **match score**
> for the verdict and **ACPL** as the finer signal; small accuracy gaps are
> noise. The opening book is what makes a small-N match conclusive.

### 4. Promote the winner

If the candidate is clearly stronger, make it the engine's net:

```bash
cp data/eclipse_candidate.nnue data/eclipse.nnue
```

(`data/eclipse.nnue` is what UCI `EvalFile` and the test suite default to.)

---

## Testing & evaluation toolkit (reference)

| Script | Purpose |
|---|---|
| `scripts/run_match.sh` | cutechess launcher; `--book` for imbalanced openings, `--debug`, `--dry-run` |
| `scripts/match_score.py` | Per-game results + running W/D/L score from a PGN |
| `scripts/match_depth.py` | Per-net avg/max depth and time-per-move from PGN comments |
| `scripts/analyze_accuracy.py` | Stockfish accuracy% + ACPL per game (cached; `--no-cache`) |
| `scripts/engine_diag.sh` | Live nps/depth/seldepth from one standalone search |
| `scripts/ponderhit_stats.sh` | Ponderhit count + avg nps from a cutechess `-debug` log |
| `scripts/fetch_latest_net.sh` | Download latest Kaggle checkpoint → `data/eclipse.nnue` (cp=300) |
| `scripts/bench_vs_stockfish.py` | Calibrate strength vs Stockfish at a target Elo |

`watch` keeps any of the PGN readers live:

```bash
watch -n 30 'python3 scripts/match_score.py /tmp/cand_vs_cur.pgn'
```
