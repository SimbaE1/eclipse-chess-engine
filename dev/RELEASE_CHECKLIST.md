# Eclipse release checklist

How a version gets from "this looks stronger" to a tagged GitHub release. v1.0.0
(June 2026) shipped against a one-off, criteria-gated checklist (no time
forfeits over 50+ live games, a settled rating, a validated net). Everything
since has followed the routine below.

## 1. It has to win on the board

- [ ] **SPRT against the current release** passes H1, or, when choosing between
      two candidates, a symmetric `-0 -5 -1 5` run picks the one with the higher
      point estimate. See "Test the candidate: SPRT" in
      [`DEVELOPMENT.md`](DEVELOPMENT.md). Val loss, MAE and ACPL don't count.
- [ ] Record each result (W-L-D, Elo ± error, LLR, game count, TC) for the
      release notes.
- [ ] Zero time forfeits in the SPRT PGNs. A loss on time is a timing bug, not
      a chess result.
- [ ] If the net changed, it was tested **against the net the bot actually
      runs**, not only against another candidate. If it wasn't, say so in the
      notes.

## 2. Build and test

- [ ] `ECLIPSE_NNUE_PATH=$PWD/data/eclipse.nnue ECLIPSE_SYZYGY_PATH=$HOME/syzygy ctest --test-dir build`:
      14/14 pass, with the net and tablebase tests actually running, not skipped.
- [ ] `./scripts/pgo_build.sh` produces the deploy binary from a clean checkout.
- [ ] `bench` prints the expected `NNUE loaded: … (HalfKAv2-<shape>, output_cp/unit=300)`.

## 3. Version bump (one commit: `chore: bump version to X.Y.Z`)

- [ ] `src/uci.cpp`: `kEngineVersion`
- [ ] `CMakeLists.txt`: `project(... VERSION X.Y.Z)`
- [ ] `README.md` + `SETUP.md`: the `gh release download vX.Y.Z` lines, the
      net's file name, the status line and the strength figure.

## 4. Tag and publish

- [ ] `git tag vX.Y.Z && git push origin main vX.Y.Z`
- [ ] `gh release create vX.Y.Z data/<net>.nnue --title "Eclipse vX.Y.Z: …" --notes-file …`
      Notes: what the net is (lineage, data, steps, lr), an SPRT results table,
      search changes with the option names and defaults that changed.
- [ ] Download the asset from the release and `bench` it once, to check the
      uploaded file is the one you tested.

## 5. Deploy to the bot

- [ ] Point `EvalFile` in the lichess-bot config at the new net; rebuild the PGO binary.
- [ ] Stop the bot between games, not mid-game (SIGINT forfeits the game in
      progress), and restart it.
- [ ] Watch the first few games for crashes, hangs and time forfeits.
