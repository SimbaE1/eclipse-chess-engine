// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#include "eval.hpp"
#include "move.hpp"
#include "position.hpp"

namespace eclipse {

struct SearchLimits {
    int     depth     = 64;   // max iterative-deepening depth
    int64_t time_ms   = 0;    // soft time bound; 0 = unlimited
    int64_t nodes     = 0;    // node bound; 0 = unlimited (not enforced yet)
    bool    infinite  = false;
};

struct SearchInfo {
    SearchLimits      limits;
    std::atomic<bool> stop{false};

    // Filled during search.
    std::int64_t                            nodes_searched = 0;
    std::chrono::steady_clock::time_point   start_time{};

    Move  best_move  = MoveNone;
    Score best_score = -kInfinite;

    bool time_up() const noexcept;
};

// Iterative-deepening negamax with alpha-beta and quiescence. Prints `info`
// lines per completed depth, leaves the final result in `info.best_move`.
// Returns the best move (also written into info).
Move search(Position& pos, SearchInfo& info);

}  // namespace eclipse
