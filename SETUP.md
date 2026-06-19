# Setting up Eclipse

A detailed, step-by-step guide to get from a fresh machine to Eclipse playing
chess. For a condensed version and how to *play* once it's running, see
[`README.md`](README.md).

There are four required steps — **dependencies → clone → build → get the net** —
plus optional tablebases. Budget ~5 minutes.

---

## 1. Install dependencies

Eclipse needs **CMake** (build system), a **C++20 compiler**, and **ONNX
Runtime** (used by the search; required to build).

### macOS (Homebrew)

```bash
brew install cmake onnxruntime
```

Xcode Command Line Tools provide the compiler (`xcode-select --install` if you
don't have them).

### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install -y build-essential cmake git
```

Install ONNX Runtime from the [official releases](https://github.com/microsoft/onnxruntime/releases)
(grab a `onnxruntime-linux-x64-*.tgz`, extract it), then point CMake at it in
step 3 with `-DONNXRUNTIME_ROOT=/path/to/onnxruntime`.

---

## 2. Clone the repository

Eclipse uses the [Fathom](https://github.com/jdart1/Fathom) library for Syzygy
tablebase probing as a **git submodule**, so clone with `--recursive`:

```bash
git clone --recursive https://github.com/SimbaE1/eclipse-chess-engine.git
cd eclipse-chess-engine
```

Already cloned without `--recursive`? Pull the submodule in:

```bash
git submodule update --init --recursive
```

---

## 3. Build

```bash
cmake -S . -B build
cmake --build build -j
```

On Linux, if ONNX Runtime isn't auto-found:

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build -j
```

This produces:

- `build/src/eclipse` — the UCI engine binary
- `build/tests/test_*` — unit and smoke tests

The build is Release with `-march=native` (tuned for your CPU) by default.

---

## 4. Get the neural network

The trained NNUE weights ship as a **GitHub Release** asset (~90 MB, too big for
the repo). The engine looks for it at `data/eclipse.nnue` by default.

With the [GitHub CLI](https://cli.github.com/):

```bash
gh release download v0.9.0 --pattern '*.nnue' -D data/
mv data/eclipse_*.nnue data/eclipse.nnue
```

Without `gh`, download the latest `.nnue` from the
[releases page](https://github.com/SimbaE1/eclipse-chess-engine/releases/latest)
and save it as `data/eclipse.nnue`.

> The small policy-index map (`data/lc0_policy_map.txt`) is already in the repo,
> so once the `.nnue` is in place you have everything needed for normal play.

---

## 5. (Optional) Syzygy endgame tablebases

For perfect endgame play, download [Syzygy tablebases](https://syzygy-tables.info/)
(3–4–5-piece sets are ~1 GB and plenty for casual play; 6-piece is ~150 GB). Put
them in a folder and pass that path as the `SyzygyPath` UCI option. Without
tablebases Eclipse plays endgames from its own evaluation — it just won't be
provably perfect.

---

## 6. Verify it works

Quick sanity check — load the net and run the built-in benchmark:

```bash
printf 'setoption name EvalFile value data/eclipse.nnue\nsetoption name Threads value 4\nisready\nbench\nquit\n' | ./build/src/eclipse
```

You should see a `NNUE loaded:` line and a `Benchmark: … nodes … nps` summary. If
you'd rather run the test suite:

```bash
ctest --test-dir build
```

---

## 7. Play

You're done — head back to [`README.md`](README.md#play-against-it) for playing
in a GUI, in the terminal, or over raw UCI.

The fastest way to try it: load it into a UCI GUI (Cute Chess, Arena, …) with
`EvalFile` set to your `data/eclipse.nnue`, or challenge the live bot at
[@EclipseBOT](https://lichess.org/@/EclipseBOT).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not locate onnxruntime` | `brew install onnxruntime` (macOS), or pass `-DONNXRUNTIME_ROOT=/path/to/onnxruntime` (Linux). |
| Build errors mentioning `extern/fathom` / `tbprobe` | The submodule wasn't fetched — run `git submodule update --init --recursive`. |
| `failed to load NNUE` / version mismatch | The `.nnue` is for a different build, or got truncated. Re-download the net for your release and rebuild the engine. |
| Engine runs but plays instantly / weakly | `EvalFile` isn't set or points at a missing file — confirm `data/eclipse.nnue` exists and the path is correct. |

---

## Building/training it yourself

Everything for hacking on the engine or training a new network lives under
[`dev/`](dev/) — start with [`dev/DEVELOPMENT.md`](dev/DEVELOPMENT.md) (internals,
build/test loop) and [`dev/KAGGLE.md`](dev/KAGGLE.md) (the training pipeline). The
optional Lc0 policy network (an A/B experiment, not needed for normal play) is
documented there too.
