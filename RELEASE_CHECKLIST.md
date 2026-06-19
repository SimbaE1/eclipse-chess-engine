# Eclipse v1.0 Release Checklist

V1 is **criteria-gated, not date-gated.** Ship when every box below is green —
a public announcement is a one-shot first impression, so don't rush it.

## 1. Time management — proven stable in live play
The won-game time-flag has regressed 3× (see memory `project_time_flag_regression`);
the latest fixes (validation-MCTS bound `e29a44e`, AB-verifier ponder-awareness
`cf8c660`) are verified synthetically but the original failure could not be
reproduced offline. Trust only after real mileage.

- [ ] **Zero time-forfeits across ≥50 live games / several days** of continuous play.
- [ ] Spot-check a won, long-pondered endgame game: bestmove always arrives before the flag.
- [ ] If it flags even once: stop, capture the engine's own stderr (`info string`
      lines lichess-bot swallows) during the flag, and diagnose before continuing.

## 2. net2widernet — validated *stronger*, not just bigger
A function-preserving 2× widen starts equal to ep2c9 and should improve with
training, but an undertrained/mis-tuned wide net can land neutral or worse.
"Bigger" is a hypothesis until the games say otherwise.

- [ ] Wide net trained to convergence (val loss plateaued), checkpoint exported.
- [ ] Head-to-head vs current ep2c9 net shows a **clear, significant Elo gain**
      (SPRT pass, or a few hundred games with non-overlapping error bars).
- [ ] Engine C++ side updated for the wider net (`kFtOutSize` in `src/nnue.hpp`,
      accumulator width + SIMD ladders in `src/nnue.cpp`/`src/accumulator.hpp`,
      `scripts/convert_halfkav2_nnue.py`) — and it loads + benches correctly.
- [ ] Deployed net tagged/published so the README download link is reproducible.

## 3. Rating settled
- [ ] Live rating has plateaued (low RD, stable over the last ~20 games) after
      recovering from the pre-fix flag-loss suppression.
- [ ] Announcement can state an honest strength figure backed by that rating.

## 4. Release hygiene
- [ ] Bump version `0.0.1` → `1.0.0` in **`src/uci.cpp` (`kEngineVersion`)** and
      **`CMakeLists.txt` (`project(... VERSION ...)`)**.
- [ ] `git tag v1.0.0` + push the tag; cut a GitHub release with notes.
- [ ] README accurate end-to-end: build steps, **net download link**, Syzygy setup,
      feature list, UCI options. Run the documented setup clean to confirm.
- [ ] `scripts/pgo_build.sh` produces the deploy binary from a clean checkout; 9/9 ctest pass.
- [ ] Bot survives sustained load: 100-games/day cap, matchmaking loop, no hangs/crashes.

## 5. Announcement (only after 1–4 are green)
- [ ] Bot profile/bio describes it (open-source NNUE + MCTS + AB verifier + Syzygy).
- [ ] Post where it fits: lichess forum / r/chess / r/ComputerChess — link the repo + release.
- [ ] Be upfront about strength and that it's open source; invite challenges.
