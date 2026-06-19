// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#include "eval.hpp"
#include "move.hpp"
#include "position.hpp"

namespace eclipse {

// SearchLimits lives in types.hpp now (added with a ponder field for the
// uci_ponder path). 0 means unlimited for the three numeric bounds; depth
// is reinterpreted as a visit ceiling under MCTS (see mcts.cpp worker_loop).
// Leaving the default non-zero used to silently cap every search at 64
// visits, starving the policy net of any real search budget regardless of
// the movetime/time-control the GUI sent.

struct SearchInfo {
    SearchLimits      limits;
    std::atomic<bool> stop{false};

    // Worker count for multi-threaded MCTS. Set by the UCI Threads option,
    // hard-clamped to [1, 128]. 1 keeps the single-threaded code path.
    int               threads = 1;

    // AB-vs-MCTS override threshold in cp. AB only overrides the played move
    // when its evaluation is at least this many cp above MCTS's. 150 (one
    // piece) was the historical default, but in real tactical positions
    // MCTS's misses are usually 30-100 cp, so the override never fired. 50
    // cp = half a pawn — fires for real tactical wins, ignores fluctuations
    // of ~a tempo. Tunable via UCI `OverrideMargin`. Mate scores always
    // override regardless of this value (see search.cpp).
    int               override_margin = 50;

    // Number of threads to dedicate to the Alpha-Beta verifier.
    int               ab_threads = 1;

    // Filled during search. Atomic because multiple MCTS workers fetch_add
    // concurrently.
    std::atomic<std::int64_t>               nodes_searched{0};
    std::chrono::steady_clock::time_point   start_time{};

    // Absolute wall-clock instant past which the move MUST already be out.
    // Derived once at search start from limits.hard_limit_ms; every phase
    // clamps its own budget to what remains before this, and time_up() reports
    // true once it passes regardless of which phase's per-phase budget is
    // active. Default (epoch) means "no hard deadline" — phases that reset
    // start_time (extensions) still honour this because it is absolute, not
    // relative to start_time. See SearchLimits::hard_limit_ms.
    std::chrono::steady_clock::time_point   hard_deadline{};

    // Wall-clock ms (steady_clock epoch) at which `ponderhit` arrived, or -1
    // if not yet received. The UCI main thread writes this and the search
    // thread reads it from time_up() — both non-atomic `start_time` and
    // `limits.ponder` were previously mutated from the main thread on
    // ponderhit while the search thread read them concurrently, a data race
    // that could make time_up() see a stale start_time and report time-up
    // immediately, starving MCTS of any iterations and yielding `bestmove
    // 0000`. This atomic is the only ponderhit signal the search thread
    // reads.
    std::atomic<std::int64_t> ponderhit_at_ms{-1};

    Move  best_move  = MoveNone;
    Score best_score = -kInfinite;

    // When true, worker_loop suppresses UCI info lines. Used for internal
    // searches (validation MCTS) that run from a different position than the
    // GUI's current root — their PV moves would be flagged as illegal otherwise.
    bool silent = false;

    // Optional external abort flag. An internal sub-search (the validation
    // MCTS) runs on its OWN SearchInfo, so the parent's `stop` atomic — the one
    // the UCI `join_search_thread()`/ponder-miss path sets — never reaches it.
    // Pointing this at the parent's `&stop` makes time_up() honour that abort,
    // so a ponder miss can't leave the validator running (and blocking the
    // join) after the move it belongs to is already over. Null for top-level
    // searches, which use their own `stop` directly.
    const std::atomic<bool>* ext_stop = nullptr;

    bool time_up() const noexcept;

    // ms remaining until hard_deadline (clamped to >= 0). Returns a large
    // sentinel when no hard deadline is set, so call sites can min() against it
    // unconditionally. Used by each phase to size or skip its budget so their
    // cumulative wall time never crosses the deadline.
    std::int64_t ms_until_hard_deadline() const noexcept;
};

// Iterative-deepening negamax with alpha-beta and quiescence. Prints `info`
// lines per completed depth, leaves the final result in `info.best_move`.
// Returns the best move (also written into info).
Move search(Position& pos, SearchInfo& info);

}  // namespace eclipse
