# Eclipse — setup on a fresh machine

Steps to get from "fresh clone" to "engine playing chess" on a new Mac. Once
it's running, see [`README.md`](README.md) for how to play and configure it, and
[`DEVELOPMENT.md`](DEVELOPMENT.md) for the engine internals and net-improvement
pipeline.

## 1. Dependencies (Homebrew)

```bash
brew install cmake stockfish lc0 cutechess onnxruntime
```

- **cmake** — build system
- **stockfish** — reference engine for benchmarking + label generation
- **lc0** — only needed for converting the Leela policy network to ONNX (one-time)
- **cutechess** — engine-vs-engine tournaments for strength measurement
- **onnxruntime** — runtime for the policy net inference at search time

Python deps (for training/data scripts, not required just to run the engine):

```bash
pip install python-chess torch numpy
```

## 2. Build the engine

```bash
cmake -S . -B build
cmake --build build -j
```

Produces:
- `build/src/eclipse` — UCI engine binary
- `build/tests/test_*` — unit + smoke tests

## 3. (Optional) Get the policy network

By default Eclipse derives MCTS priors from the NNUE value net — this Lc0
transformer is an **optional** A/B path, consulted only when you set the UCI
options `PolicyMode onnx` + `PolicyFile data/policy.onnx`. Without it, MCTS uses
NNUE/heuristic priors and the time-manager's moves-left estimate falls back to a
fixed default. **Skip this whole section for normal play.** Source format is a
Leela `.pb` file which `lc0` converts to ONNX:

1. Download `t1-256x10-distilled-swa-2432500.pb` (or any other Lc0 net you
   prefer) from https://training.lczero.org/networks/
2. Convert it:
   ```bash
   lc0 leela2onnx \
       --input=/path/to/t1-256x10-distilled-swa-2432500.pb \
       --output=data/policy.onnx
   ```
   Output is ~80 MB. The conversion logs the network architecture; verify it
   says `Policy: Attention` and `Value: WDL` — those are the tensor names
   `policy.cpp` expects.

## 4. Get the NNUE weights

The trained NNUE is the only weight file normal play needs; it's published as a
GitHub Release asset. `gh` (already auth'd if you cloned via `gh repo clone`):

```bash
gh release download v0.9.0 --pattern '*.nnue' -D data/
mv data/eclipse_*.nnue data/eclipse.nnue          # what UCI EvalFile expects
```

(Use the latest release tag — `gh release list` shows them.) Alternatively grab
them via the Releases page (this repo is private, so use `gh` until/unless you
make it public):
```
https://github.com/SimbaE1/eclipse-chess-engine/releases
```

If you'd rather scp from another machine you've already built on:
```bash
scp m1-air:/Users/ezra/TCEC/eclipse/data/eclipse.nnue data/
# data/policy.onnx too, but only if you use the optional PolicyMode onnx path
```

With the release in `data/`, you can skip step 3 (policy conversion) entirely —
`policy.onnx` is the same artifact `lc0 leela2onnx` would produce.

The 1858-entry policy index → UCI move map (`data/lc0_policy_map.txt`) is
required for the policy net to produce meaningful priors. It's committed to
the repo since it's tiny (~17 KB) and identical for every Lc0 net, so a
fresh clone has everything it needs after the release-download step.

## 5. Verify

```bash
ECLIPSE_NNUE_PATH=$PWD/data/eclipse.nnue \
ECLIPSE_NNUE_TRAINED=1 \
./build/tests/test_nnue
```

Expected output ends with `PASS  nnue`. If it fails, the NNUE file is corrupted
or the engine was built against a different `kFtOutSize` than the file's header.

## 6. Run it

```bash
./build/src/eclipse
```

Then in the UCI prompt:
```
uci
setoption name EvalFile value PATH/TO/eclipse/data/eclipse.nnue
# optional A/B path only: setoption name PolicyMode onnx + PolicyFile <policy.onnx>
isready
position startpos
go movetime 5000
```

Or use the friendly Python wrapper to play interactively:

```bash
python scripts/play_human.py --side w --engine-time-ms 5000
```

## 7. Benchmark against Stockfish at a target Elo

```bash
cutechess-cli \
    -engine cmd="$PWD/build/src/eclipse" name="Eclipse" proto=uci \
        option.EvalFile="$PWD/data/eclipse.nnue" \
    -engine cmd="$(which stockfish)" name="SF-1800" proto=uci \
        option.UCI_LimitStrength=true option.UCI_Elo=1800 \
    -each tc=60+1 -rounds 10 -games 2 -repeat -recover \
    -pgnout data/eclipse_vs_sf1800.pgn -ratinginterval 4
```

PGNs of benchmark runs are committed to `data/*.pgn`. Strength results so far:

| NNUE version | Training positions | vs SF-1800 | Elo |
|---|---|---|---|
| v1 | 677,840 | (see PGN) | TBD |
| v2 | 1,580,000 | (see PGN) | TBD |

## Training a new NNUE

Full pipeline lives in `scripts/`:

| Stage | Script | Notes |
|---|---|---|
| Extract Lichess positions | `sample_lichess.py` / `extract_lichess_evals.py` | Streaming, filters by Elo + TC |
| Label with Stockfish | `label_with_stockfish.py` | Optional — only if not using Lichess's built-in evals |
| Train HalfKAv2 net | `train_halfkav2.py` | PyTorch, supports `--device cuda/mps/cpu` |
| Pack to .nnue | `convert_halfkav2_nnue.py` | Quantizes float32 -> int8/int16 |

Recommended workflow for the Mac Pro: extract data locally (network-bound),
upload to Kaggle, train on a T4 in a notebook, download the `.pt`, pack locally.
See [`KAGGLE.md`](KAGGLE.md) for the end-to-end Kaggle pipeline and
[`notebooks/eclipse_wdl_train.ipynb`](notebooks/eclipse_wdl_train.ipynb) for
the notebook itself.

## Architecture overview

Eclipse is multi-threaded MCTS + a HalfKAv2 NNUE value net (`45056 → 1024×2 →
512 → 128 → 1`, incremental accumulator, AVX-512/AVX2/NEON SIMD) with an
alpha-beta tactical verifier and Syzygy probing. Full details, the search
tunables, and the source map are in [`DEVELOPMENT.md`](DEVELOPMENT.md).
