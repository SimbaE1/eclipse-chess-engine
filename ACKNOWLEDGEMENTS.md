# Acknowledgements

Eclipse stands on a lot of shoulders. This file credits the people, projects,
and ideas it draws from. None of them endorse Eclipse — any mistakes here are
ours, not theirs.

## Ideas & research

- **DeepMind** — for pioneering the modern Monte-Carlo Tree Search + neural
  network approach (AlphaGo / AlphaZero). Eclipse's PUCT tree search and its
  "neural intuition guided by search" design come straight from that lineage.
- **The Stockfish team & the NNUE community** — for **NNUE** and the
  **HalfKAv2** feature set Eclipse's evaluation net is built on, and for the
  practical conventions we follow (king-relative perspective-doubled features,
  the 127/16 quantization scales, Stockfish-style LMR, make/unmake with a
  state-info stack). Eclipse's training labels are Stockfish depth-22
  evaluations.
- **The Leela Chess Zero (Lc0) team** — Eclipse's move-policy priors come from
  an Lc0 transformer policy net, and its time management uses that net's
  **moves-left head (MLH)** to estimate how much game is left.
- **Ronald de Man** — for **Syzygy** endgame tablebases, which give Eclipse
  perfect play once the board is small enough.

## Data & training

- **Lichess** — both the platform Eclipse plays on ([@EclipseBOT](https://lichess.org/@/EclipseBOT))
  and, via the **Lichess open database**, the source of the games behind the
  training set.
- **Stockfish** — again, as the engine that produced the per-position evaluation
  labels the value net is trained on.
- **nnue-pytorch** (`github.com/glinscott/nnue-pytorch`) — Gary Linscott and the
  Stockfish NNUE contributors, for the trainer design and binpack format that
  informed Eclipse's training pipeline.
- **Kaggle** — for the free GPU sessions the net is trained on.

## Libraries & tools

- **Fathom** ([jdart1/Fathom](https://github.com/jdart1/Fathom)) — Syzygy
  tablebase probing in C, vendored as a submodule.
- **python-chess** (Niklas Fiekas) — board logic, PGN parsing, and SVG board
  rendering, used in the training data tooling and the README game card.
- **PyTorch** & **NumPy** — neural-network training.
- **ONNX Runtime** — inference for the Lc0 policy/MLH net at runtime.
- **lichess-bot** ([lichess-bot-devs](https://github.com/lichess-bot-devs/lichess-bot))
  — the bridge that runs Eclipse as a Lichess bot.
- **CMake** & the **GCC/Clang** toolchains — build, LTO, and PGO.
- **shields.io** — the live rating badges in the README.

## README status card

- **Ken Wu** ([@KenWuqianghao](https://github.com/KenWuqianghao)) — for
  [Lichess-Readme](https://github.com/KenWuqianghao/Lichess-Readme) and
  [Lichess-Game-Readme](https://github.com/KenWuqianghao/Lichess-Game-Readme),
  which inspired the "live rating + latest game" card in our README. Ours is a
  no-server variation (live shields.io badges + a GitHub Action that renders the
  board), but the idea is his.

## Development

- **Anthropic — Claude & Claude Code** — used throughout Eclipse's development
  for engineering, debugging, and tooling.

---

*Something missing or miscredited? Open an issue or a PR — we'd rather fix it.*
