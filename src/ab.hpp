// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>

#include "eval.hpp"
#include "move.hpp"
#include "position.hpp"

namespace eclipse::ab {

struct Result {
    Move  move       = MoveNone;
    Score score      = 0;            // from root side-to-move's perspective
    int   reached_d  = 0;            // depth actually completed
    std::int64_t nodes = 0;
    std::int64_t elapsed_ms = 0;     // wall time spent in find_best_move (for nps)
    // Root score at each completed depth (depth_scores[i] == score after
    // depth i+1), root side-to-move perspective. Lets callers see whether the
    // eval is still drifting with depth -- a move that looks fine shallow but
    // erodes deeper (a slow-burn positional trap) -- and decide to search on.
    std::vector<Score> depth_scores;
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
//
// `num_threads` > 1 runs Lazy SMP: helper threads share the global TT to push
// the main worker deeper in the same wall-clock. Pass the thread count only
// where those threads are actually free (e.g. the reconciliation extension,
// where MCTS is idle) -- the steady-state parallel phase should pass its
// reserved AB-thread count so it doesn't oversubscribe against MCTS.
//
// `stop`, when non-null, is polled during the search (same cheap node-stride as
// the time check): once it goes true the search aborts and returns the deepest
// result completed so far. The caller passes &SearchInfo::stop so that a `stop`
// / ponder-miss / quit unblocks the AB thread promptly instead of running out
// its time budget — otherwise the join that gates the next search stalls and
// the wasted work is billed to the real move's clock.
// `ponder_hit_ms`, when non-null (e.g. &SearchInfo::ponderhit_at_ms), makes the
// time budget ponder-aware: it is measured from the ponderhit instant, and the
// search runs unbounded on the opponent's free time until the hit arrives. Pass
// it only for a worker launched at `go ponder` (the parallel verifier); a search
// started after the hit should leave it null and use the plain start-relative
// budget.
Result find_best_move(Position& pos, int max_depth, std::int64_t time_budget_ms,
                      int num_threads = 1, const std::atomic<bool>* stop = nullptr,
                      const std::atomic<std::int64_t>* ponder_hit_ms = nullptr);

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

// Result of find_tactic_node(): the specific node, several plies into `pos`,
// where AB's iterative deepening first discovered a refutation/tactic --
// AB's "aha moment" -- along with its value from that node's own
// side-to-move perspective, in MCTS Q-units ([-1, 1], cp / output_cp_per_unit).
struct TacticNode {
    bool               found = false;
    std::vector<Move>  path;        // moves from `pos` down to (and including) the tactic node
    float               seed_q = 0.0f;       // value AT the tactic node, ITS OWN side-to-move's perspective
    Score                root_score_cp = 0;  // value of `pos` itself (root of this trace), pos's own STM perspective, at aha_depth
    int                  aha_depth = 0;       // AB depth at which the swing was confirmed stable
};

// Runs iterative deepening at `pos`, recording the score AND principal
// variation at every completed depth (PV extracted by walking the TT --
// every node visited during search already stores its best move there).
// Scans the depth sequence for the shallowest point where the score jumps
// sharply AND stays close to the new value for the next couple of
// completed depths -- a real, newly-discovered line, not aspiration-window
// noise from one unlucky re-search. The node where that depth's PV first
// diverges from the previous depth's PV is the tactic node: the position
// AB didn't realize was bad/good until it searched one ply deeper than
// before.
//
// Used to seed MCTS's evaluation at that exact node (see
// MCTS::set_value_seed) instead of leaving it to NNUE judgment + visit
// allocation that may never reach it. Returns found=false if no stable
// swing is found in the time/depth budget -- callers should treat that as
// "AB didn't find anything conclusive enough to trust," not retry harder.
TacticNode find_tactic_node(Position& pos, int max_depth, std::int64_t time_budget_ms);

}  // namespace eclipse::ab
