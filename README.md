# Eclipse

A UCI chess engine aiming for TCEC-competitive strength. Eclipse pairs an
**MCTS search** with a **HalfKAv2 NNUE** evaluation and an alpha-beta tactical
verifier — so it plays positionally like a neural engine but doesn't fall for
tactics. It runs in any standard chess GUI and on the command line.

> **Status:** late proof-of-concept — fully playable. It speaks UCI, supports
> multi-threading, pondering, and Syzygy tablebases.

This page is for **running and playing** Eclipse. If you want to hack on the
engine itself, see [`DEVELOPMENT.md`](DEVELOPMENT.md).

---

## 1. Get it

Eclipse builds from source; the neural net ships separately as a release asset
(it's ~90 MB, too big for the repo).

```bash
# clone (private repo — use the GitHub CLI)
gh repo clone SimbaE1/eclipse-chess-engine
cd eclipse-chess-engine

# install build + runtime dependencies (macOS / Homebrew)
brew install cmake onnxruntime

# build (Release, tuned for your CPU)
cmake -S . -B build
cmake --build build -j
# -> build/src/eclipse
```

Then grab the neural net into `data/`:

```bash
gh release download v0.9.0 --pattern '*.nnue' -D data/
mv data/eclipse_*.nnue data/eclipse.nnue      # the name EvalFile defaults to
```

> First time on a fresh machine, or hitting build/dependency trouble? The
> step-by-step bootstrap (including the optional policy net and a verify step)
> is in [`SETUP.md`](SETUP.md).

Optional but recommended for strong endgame play — point Eclipse at
[Syzygy tablebases](https://syzygy-tables.info/) you've downloaded (3–4–5 man is
plenty for casual play).

---

## 2. Play against it

### In a chess GUI (recommended)

Eclipse is a standard UCI engine, so it drops into any UCI GUI — **Cute Chess**,
**Arena**, **BanksiaGUI**, **Scid vS. PC**, etc. Add a new engine and point it at:

- **Command:** `…/eclipse-chess-engine/build/src/eclipse`
- Then set these engine options in the GUI:
  - `EvalFile` → `…/eclipse-chess-engine/data/eclipse.nnue` *(required)*
  - `Threads` → how many CPU cores to use (e.g. 4)
  - `Hash` → transposition memory in MB (e.g. 256)
  - `SyzygyPath` → your tablebase folder *(optional)*

Now start a game against it like any other engine.

### In the terminal (quick game, no GUI)

A small Python helper lets you play right in the terminal (needs
`pip install python-chess`):

```bash
python scripts/play_human.py --side w --tc 5+3 --threads 4
```

- `--side w|b` — which color **you** play (default white)
- `--tc 5+3` — time control (5 min + 3 s); or `--tc fixed --engine-time-ms 3000`
  for a flat think-time per move
- `--threads`, `--hash`, `--syzygy` — passed through to the engine

Type your moves in UCI form (e.g. `e2e4`, `g1f3`, `e7e8q` to promote).

### Raw UCI (for the curious)

```bash
./build/src/eclipse
```
```
uci
setoption name EvalFile value data/eclipse.nnue
setoption name Threads value 4
setoption name Hash value 256
setoption name SyzygyPath value /path/to/syzygy
isready
position startpos
go movetime 5000
```
It replies with `bestmove …`. `quit` to exit.

---

## 3. Configure it

The options you'll actually touch as a player:

| Option | Default | What it does |
|---|---|---|
| `EvalFile` | — | Path to the `.nnue` net **(required)** |
| `Threads` | 1 | CPU cores for search — set this to your core count |
| `Hash` | 256 | Search memory (MB) |
| `SyzygyPath` | — | Folder of Syzygy tablebases for perfect endgames |
| `Ponder` | on | Let it think on your time (the GUI manages this) |

Eclipse also exposes search-strength knobs (`Cpuct`, `PolicyDepth`, …) for
tinkerers — those and their trade-offs are documented in
[`DEVELOPMENT.md`](DEVELOPMENT.md#search-tunables-uci-options).

---

## 4. Test it / measure its strength

**Quick self-check** — a fixed-node benchmark prints nodes/sec:

```bash
printf 'setoption name EvalFile value data/eclipse.nnue\nsetoption name Threads value 4\nisready\nbench\nquit\n' | ./build/src/eclipse
```

**Play it against another engine** with [cutechess-cli](https://github.com/cutechess/cutechess).
For example, a 10-game match at 5 min + 3 s vs. another UCI engine, giving
Eclipse 4 cores and its tablebases (no result adjudication):

```bash
cutechess-cli \
  -engine name=eclipse cmd=./build/src/eclipse \
    option.EvalFile=$PWD/data/eclipse.nnue \
    option.Threads=4 option.Hash=256 \
    option.SyzygyPath=/path/to/syzygy \
  -engine name=opponent cmd=/path/to/other-engine \
    option.Hash=256 \
  -each tc=5+3 proto=uci \
  -games 10 -repeat -concurrency 1 \
  -pgnout /tmp/eclipse_match.pgn
```

Then read off the result:

```bash
python3 scripts/match_score.py /tmp/eclipse_match.pgn     # running W/D/L score
```

For deeper analysis (per-net depth/time, Stockfish accuracy + ACPL, live
watching) and the net-vs-net testing workflow, see
[`DEVELOPMENT.md`](DEVELOPMENT.md#testing--evaluation-toolkit-reference).

---

## Documentation

| Doc | For |
|---|---|
| **README.md** (this) | Download, play, configure, test |
| [`SETUP.md`](SETUP.md) | Fresh-machine bootstrap, step by step |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | Engine internals, build/test loop, net-improvement pipeline |
| [`KAGGLE.md`](KAGGLE.md) | Training a new NNUE (data + GPU pipeline) |

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
