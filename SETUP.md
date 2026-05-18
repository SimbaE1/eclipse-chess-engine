# Eclipse — setup on a fresh machine

Steps to get from "fresh clone" to "engine playing chess" on a new Mac.

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

## 3. Get the policy network

The Lc0 transformer that drives MCTS priors. Source format is a Leela `.pb` file
which `lc0` converts to ONNX:

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

## 4. Get the NNUE weights (and policy ONNX shortcut)

Both the trained NNUE and the converted Lc0 policy are bundled in the repo's
GitHub Release. `gh` (already auth'd if you cloned via `gh repo clone`):

```bash
gh release download v0.1.0 --pattern '*.nnue' --pattern 'policy.onnx' -D data/
mv data/eclipse_v2_1.58M.nnue data/eclipse.nnue   # what UCI EvalFile expects
```

Alternatively grab them via plain HTTPS (no auth needed for public releases;
this repo is private, so use `gh` until/unless you make it public):
```
https://github.com/SimbaE1/eclipse-chess-engine/releases/tag/v0.1.0
```

If you'd rather scp from another machine you've already built on:
```bash
scp m1-air:/Users/ezra/TCEC/eclipse/data/eclipse.nnue data/
scp m1-air:/Users/ezra/TCEC/eclipse/data/policy.onnx data/
```

With the release in `data/`, you can skip step 3 (policy conversion) entirely —
`policy.onnx` is the same artifact `lc0 leela2onnx` would produce.

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
setoption name PolicyFile value PATH/TO/eclipse/data/policy.onnx
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
        option.PolicyFile="$PWD/data/policy.onnx" \
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
| Train HalfKP net | `train_halfkp.py` | PyTorch, supports `--device cuda/mps/cpu` |
| Pack to .nnue | `convert_halfkp_nnue.py` | Quantizes float32 -> int8/int16 |

Recommended workflow for the Mac Pro: extract data locally (network-bound),
upload to Kaggle, train on a T4 in a notebook, download the `.pt`, pack locally.
See conversation history for details.

## Architecture overview

- **Search**: MCTS with PUCT, single-threaded
- **Value**: HalfKP-512x2-32-32 NNUE (40960 features -> 512 per perspective ->
  32 -> 32 -> 1, int16/int8 quantized) in `src/nnue.cpp`
- **Policy**: Lc0 transformer (10 encoder layers, 8 heads) loaded via
  ONNX Runtime in `src/policy.cpp`
- **Time mgmt**: MLH head from the policy net divides remaining time across
  estimated moves left (see `cmd_go` in `src/uci.cpp`)

Phase 2/3 work (incremental NNUE accumulator, SIMD inference, parallel MCTS)
is tracked in TODO comments throughout the source.
