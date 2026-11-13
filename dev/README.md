# Eclipse — developer material

Everything here is for **building on / improving** Eclipse, not for using it. If
you just want to play the engine, see the top-level [`README.md`](../README.md)
and [`SETUP.md`](../SETUP.md).

| Path | What |
|---|---|
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | How the search works, UCI tunables, build/test loop, SPRT testing, source map |
| [`KAGGLE.md`](KAGGLE.md) | NNUE training pipeline (data → filter → GPU → `.nnue`) and what the fine-tune sweeps showed |
| [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) | SPRT → version bump → tag → GitHub release → bot deploy |
| [`net2widernet.md`](net2widernet.md) | *Historical:* the 2× net-widen run (its Kaggle mechanics are still accurate) |
| [`PRODUCTION_TESTING.md`](PRODUCTION_TESTING.md) | *Historical:* July 2026 handoff for the AB-bundle SPRT |
| `scripts/` | Data extraction, chunking, net conversion, SPRT harness, match/evaluation tooling, training babysitter |
| `notebooks/` | The Kaggle training notebook (long runs) |
| `sprt_runs/` | SPRT output, one directory per run (git-ignored) |
