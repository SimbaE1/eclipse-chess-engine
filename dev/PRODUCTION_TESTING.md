# Production SPRT handoff — AB improvements bundle (2026-07-08)

Instructions for the Claude instance (or human) on the production machine to
run the long time-control SPRT that decides whether main's alpha-beta
improvements bundle stays or gets split up. Written by the dev-machine session
that produced the bundle; context below is everything you need.

## What is being tested, and why here

Commit `6e79c7b` ("feat(ab): search quality improvements in the alpha-beta
verifier") bundles seven changes: dedicated capture movegen in qsearch,
qsearch TT, qsearch in-check soundness, a castling-misclassified-as-capture
fix, PV-node pruning gates, an "improving" heuristic, and piece-to
continuation history.

A 1038-game SPRT on the dev machine at **10+0.1, Threads=1/AbThreads=1
(sequential mode)** was inconclusive and mildly negative: 506-520-12,
Elo −4.0 ± 21.1, LLR −0.28 of ±2.94. Two reasons not to trust that as final:

1. **Topology.** Sequential mode gives AB only a ~25% time slice after MCTS.
   Production runs parallel mode (dedicated AB thread alongside MCTS), which
   weights the changed code much more heavily.
2. **TC realism.** At 10+0.1 the AB slices are ~40ms; the bundle trades some
   depth for accuracy (PV gates), which may only pay off with real budgets.

This machine has the cores and the idle time to answer properly.

## Setup

```bash
cd <repo>   # the production checkout
git fetch origin
git checkout main && git pull

# Build the candidate (main)
cmake -S . -B build && cmake --build build -j
cp build/src/eclipse /tmp/eclipse_new

# Build the baseline: pre-bundle HEAD + identical timing fixes, prepared as
# branch sprt-baseline-ab-2026-07-08. NEVER test against a baseline without
# the timing fixes — pre-fix builds forfeit low-TC games on time and the
# match measures flagging, not chess.
git checkout sprt-baseline-ab-2026-07-08
cmake -S . -B build-baseline && cmake --build build-baseline -j
cp build-baseline/src/eclipse /tmp/eclipse_old
git checkout main

# Opening book (gitignored, fetch if data/books/ is empty)
mkdir -p data/books && cd data/books
curl -sL -o UHO.zip https://github.com/official-stockfish/books/raw/master/UHO_Lichess_4852_v1.epd.zip
unzip -o UHO.zip && rm UHO.zip && cd ../..

# cutechess-cli must be on PATH (brew install cutechess or existing install)
```

**If the Lichess bot runs on this machine, pause it first** (it competes for
cores and its games would be played by a busy engine). Resume it after.

## Run

Smoke test the wiring (4 games, ~1 min):

```bash
dev/scripts/sprt_run.sh /tmp/eclipse_new /tmp/eclipse_old -s -T 2 -A 1
```

Expect 4 finished games with **zero** "loses on time" results. Then the real
run — production topology (parallel AB), long TC:

```bash
nohup dev/scripts/sprt_run.sh /tmp/eclipse_new /tmp/eclipse_old \
    -t 60+0.6 -g 1500 -T 2 -A 1 -c <CONC> \
    > /tmp/sprt_ab_bundle.log 2>&1 &
```

Pick `<CONC>` as `physical_cores / 3`, minimum 2 (each game runs two engines
of Threads+AbThreads=3 threads, but only one side thinks at a time). On an
8-core machine use `-c 2`; 18-core, `-c 6`.

Duration: a 60+0.6 game runs ~3-5 min. At `-c 2` budget roughly 2 days for
the full 1500; the run stops early the moment LLR crosses ±2.94. If that's
too long, `-t 30+0.3 -g 1500` overnight is an acceptable middle ground —
prefer it over cutting the game count.

## Read the result

The log's final `SPRT:` line decides:

- **LLR ≥ +2.94 (H1)** — bundle is worth ≥ ~5 Elo here: keep it, report back,
  and the next work item is MCTS constant tuning (see task list / dev notes).
- **LLR ≤ −2.94 (H0)** — bundle does not help at production settings: do NOT
  revert wholesale. The bundle contains objective bug fixes (castling
  misclassification, qsearch in-check unsoundness) that should stay
  regardless. Split the remaining pieces — PV gates first (prime suspect:
  costs depth), then improving heuristic, then piece-to cont-history — into
  single-change branches off `sprt-baseline-ab-2026-07-08` and SPRT each at
  10+0.1/-T 1 on the dev machine for cheap screening.
- **Inconclusive after 1500 games** — the bundle is within a few Elo of
  neutral at this TC too. Keep it (bug fixes + no regression + faster
  movegen), close this question, and move on to MCTS tuning.

Report back: final W-L-D, Elo ± error, LLR, any time losses (should be zero;
if not, that's a new timing bug — check `timemargin` losses in the PGN), and
the log + `dev/sprt_runs/<timestamp>/games.pgn` kept for analysis.
