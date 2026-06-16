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

    Move  best_move  = MoveNone;
    Score best_score = -kInfinite;

    // When true, worker_loop suppresses UCI info lines. Used for internal
    // searches (validation MCTS) that run from a different position than the
    // GUI's current root — their PV moves would be flagged as illegal otherwise.
    bool silent = false;

    bool time_up() const noexcept;
};

// Iterative-deepening negamax with alpha-beta and quiescence. Prints `info`
// lines per completed depth, leaves the final result in `info.best_move`.
// Returns the best move (also written into info).
Move search(Position& pos, SearchInfo& info);

}  // namespace eclipse
