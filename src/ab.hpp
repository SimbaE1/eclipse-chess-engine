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

// Scores one specific move with an iterative-deepening full-window search,
// from `pos`'s side-to-move perspective (same convention as Result::score).
// Reuses whatever is already cached in the global TT, so a call right after
// find_best_move() on the same position is cheap — most of the relevant
// subtree is already resolved.
//
// Used to get AB's independent opinion of a move chosen by another search
// component (e.g. MCTS) directly, instead of only comparing against AB's
// own top pick: AB's normal root loop only gives an exact score to the
// first-ordered move, everything else gets a cheap alpha-beta bound, so a
// non-top-ordered move (the common case when AB and MCTS disagree) never
// gets an honest number unless asked for explicitly like this.
Score score_move(Position& pos, Move m, int max_depth, std::int64_t time_budget_ms);

}  // namespace eclipse::ab
