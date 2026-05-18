// SPDX-License-Identifier: GPL-3.0-or-later
#include "search.hpp"

#include <algorithm>
#include <iostream>
#include <chrono>

#include "mcts.hpp"

namespace eclipse {

bool SearchInfo::time_up() const noexcept {
    if (limits.infinite || limits.time_ms <= 0) return false;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();
    return elapsed >= limits.time_ms;
}

Move search(Position& pos, SearchInfo& info) {
    info.nodes_searched = 0;
    info.start_time     = std::chrono::steady_clock::now();
    info.best_move      = MoveNone;
    info.best_score     = -kInfinite;

    mcts::MCTS mcts_search(pos, info);
    info.best_move = mcts_search.search();

    return info.best_move;
}

}  // namespace eclipse
