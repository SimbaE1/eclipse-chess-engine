# Setting up Eclipse

A detailed, step-by-step guide to get from a fresh machine to Eclipse playing
chess. For a condensed version and how to *play* once it's running, see
[`README.md`](README.md).

There are four required steps — **dependencies → clone → build → get the net** —
plus optional tablebases. Budget ~5 minutes.

---

## 1. Install dependencies

Eclipse needs **CMake ≥ 3.20** (build system), a **C++20 compiler**, and **ONNX
Runtime**. ONNX Runtime only runs the optional Lc0 policy net (`PolicyFile`),
which normal play doesn't use, but the engine links against it, so it's required
to build.

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

The trained NNUE weights ship as a **GitHub Release** asset (~190 MB, too big for
the repo). The current net is `eclipse_v10_mix20.nnue` from **v1.2.0**. Save it as
`data/eclipse.nnue` and point the `EvalFile` option at it.

> **Always set `EvalFile`.** The engine's built-in default path is the
> developer's own checkout, so on any other machine it starts with no net until
> you set the option. Every GUI and the commands below do this explicitly.

With the [GitHub CLI](https://cli.github.com/):

```bash
gh release download v1.2.0 --pattern '*.nnue' -D data/
mv data/eclipse_v10_mix20.nnue data/eclipse.nnue
```

Without `gh`, download the latest `.nnue` from the
[releases page](https://github.com/SimbaE1/eclipse-chess-engine/releases/latest)
and save it as `data/eclipse.nnue`.

> The small policy-index map (`data/lc0_policy_map.txt`) is already in the repo,
> so once the `.nnue` is in place you have everything needed for normal play.

---

## 5. (Optional) Syzygy endgame tablebases

For perfect endgame play, download Syzygy tablebases, put them in a folder, and
pass that path as the `SyzygyPath` UCI option. Without tablebases Eclipse plays
endgames from its own evaluation — it just won't be provably perfect.

> **Important: download BOTH the WDL (`.rtbw`) and DTZ (`.rtbz`) files.** WDL
> alone tells the engine a position is winning but not *how* to make progress, so
> it can shuffle a won endgame (e.g. K+Q vs K) into a 50-move or repetition draw.
> The DTZ files are what drive it to mate.

The 3–4–5-piece set (~1 GB total for WDL+DTZ) is plenty for casual play; 6-piece
is ~150 GB. From the Lichess mirror:

```bash
mkdir -p ~/syzygy && cd ~/syzygy
base=https://tablebase.lichess.ovh/tables/standard
for dir in 3-4-5-wdl 3-4-5-dtz; do
  curl -s "$base/$dir/" | grep -oE '[A-Za-z0-9]+\.(rtbw|rtbz)' | sort -u \
    | xargs -P 8 -I {} sh -c '[ -f "{}" ] || curl -s -O '"$base/$dir"'/{}'
done
```

Then run Eclipse with `SyzygyPath` set to the full path of that folder (e.g.
`/home/you/syzygy`; UCI options don't expand `~`).

---

## 6. Verify it works

Quick sanity check — load the net and run the built-in benchmark:

```bash
printf 'setoption name EvalFile value data/eclipse.nnue\nsetoption name Threads value 4\nisready\nbench\nquit\n' | ./build/src/eclipse
```

You should see a `NNUE loaded: … (HalfKAv2-2048x2-1024-256, output_cp/unit=300)`
line and a `Benchmark: … nodes … nps` summary. If you'd rather run the test suite
(14 tests). The net and tablebase tests skip themselves unless you point them at
the files:

```bash
ECLIPSE_NNUE_PATH=$PWD/data/eclipse.nnue ECLIPSE_SYZYGY_PATH=$HOME/syzygy \
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
| Engine runs but plays instantly / weakly | `EvalFile` isn't set or points at a missing file — confirm `data/eclipse.nnue` exists and set `EvalFile` to its full path. |
| Memory use climbs into many GB on long thinks | That's the MCTS tree. Cap it with `MctsTreeMB` (default 8192). |
| Build is slow or the binary crashes on another machine | The default build uses `-march=native`. To build a portable binary, configure with `-DECLIPSE_NATIVE_ARCH=OFF`. |

---

## Building/training it yourself

Everything for hacking on the engine or training a new network lives under
[`dev/`](dev/) — start with [`dev/DEVELOPMENT.md`](dev/DEVELOPMENT.md) (internals,
build/test loop) and [`dev/KAGGLE.md`](dev/KAGGLE.md) (the training pipeline). The
optional Lc0 policy network (an A/B experiment, not needed for normal play) is
documented there too.
