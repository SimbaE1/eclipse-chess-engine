// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <chrono>
#include <cstdint>

#include "eval.hpp"
#include "move.hpp"
#include "position.hpp"

namespace eclipse::ab {

struct Result {
    Move  move       = MoveNone;
    Score score      = 0;            // from root side-to-move's perspective
    int   reached_d  = 0;            // depth actually completed
    std::int64_t nodes = 0;
};

// Iterative-deepening alpha-beta search with quiescence at the leaves.
// Used as a tactical sanity check on MCTS's chosen move: small depths
// (4-6 ply) catch immediate hangs / forks that an MCTS tree with few
// visits on the refutation can still miss.
//
// Stops as soon as either `max_depth` is reached or `time_budget_ms`
// has elapsed; whichever comes first. Returns the best move found at
// the deepest completed iteration. If no time is available at all
// (budget <= 0) the depth-1 result is returned.
Result find_best_move(Position& pos, int max_depth, std::int64_t time_budget_ms);

}  // namespace eclipse::ab
