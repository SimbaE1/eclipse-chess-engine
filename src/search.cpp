// SPDX-License-Identifier: GPL-3.0-or-later
#include "search.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>

#include "ab.hpp"
#include "mcts.hpp"

namespace eclipse {

namespace {
// Slice of the move-time given to the AB tactical verifier. 10% leaves
// MCTS plenty of search budget while still letting AB reach depth 5-7 in
// most positions, which is enough to catch 2-3 ply hangs that even a
// well-visited MCTS tree can miss.
constexpr std::int64_t kAbBudgetPctNum = 10;
constexpr std::int64_t kAbBudgetPctDen = 100;
// Floor / ceiling so very short or very long time controls still get
// useful tactical coverage without burning the whole clock on it.
constexpr std::int64_t kAbBudgetMinMs  = 50;
constexpr std::int64_t kAbBudgetMaxMs  = 1500;
constexpr int          kAbMaxDepth     = 7;
// Override MCTS only when AB sees a clearly better line. 150cp ≈ a piece's
// worth — below that the AB result is plausibly noise vs MCTS's broader
// positional understanding, so we trust MCTS.
constexpr Score        kOverrideMarginCp = 150;
}  // namespace

bool SearchInfo::time_up() const noexcept {
    if (limits.infinite || limits.time_ms <= 0) return false;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();
    return elapsed >= limits.time_ms;
}

Move search(Position& pos, SearchInfo& info) {
    info.nodes_searched.store(0, std::memory_order_relaxed);
    info.start_time     = std::chrono::steady_clock::now();
    info.best_move      = MoveNone;
    info.best_score     = -kInfinite;

    // Reserve a slice for the AB verifier before MCTS gets the budget.
    // If the caller asked for `infinite` or `depth N`, time_ms is 0 and
    // both phases run without a time cap (only depth / stop control them).
    const std::int64_t total_ms = info.limits.time_ms;
    std::int64_t ab_budget_ms   = 0;
    if (total_ms > 0) {
        ab_budget_ms = total_ms * kAbBudgetPctNum / kAbBudgetPctDen;
        ab_budget_ms = std::clamp(ab_budget_ms, kAbBudgetMinMs, kAbBudgetMaxMs);
        // Never reserve more than half the budget — MCTS still does the
        // bulk of the work.
        if (ab_budget_ms > total_ms / 2) ab_budget_ms = total_ms / 2;
        info.limits.time_ms = total_ms - ab_budget_ms;
    }

    mcts::MCTS mcts_search(pos, info);
    info.best_move = mcts_search.search();

    // Restore the original budget so any caller inspecting limits.time_ms
    // after search() sees what it set, not the MCTS-internal slice.
    info.limits.time_ms = total_ms;

    // AB verifier. Re-uses `pos` (do_move/undo_move keep it balanced) and
    // works off the same NNUE accumulator. Disabled when there is no time
    // pressure AND no MCTS move (something is very wrong) — otherwise we
    // always run it for the tactical safety net.
    if (info.best_move == MoveNone || ab_budget_ms <= 0) {
        // No move or no budget: nothing useful to verify.
        return info.best_move;
    }

    const ab::Result ab = ab::find_best_move(pos, kAbMaxDepth, ab_budget_ms);
    if (ab.move == MoveNone || ab.reached_d == 0) {
        // AB didn't complete a single iteration — trust MCTS.
        return info.best_move;
    }

    if (ab.move != info.best_move &&
        ab.score >= info.best_score + kOverrideMarginCp) {
        std::cout << "info string AB override: "
                  << info.best_move.to_uci() << " (mcts "
                  << info.best_score << "cp) -> "
                  << ab.move.to_uci() << " (ab "
                  << ab.score << "cp, d=" << ab.reached_d
                  << ", nodes=" << ab.nodes << ")"
                  << std::endl;
        info.best_move  = ab.move;
        info.best_score = ab.score;
    } else {
        std::cout << "info string AB verify: agree on "
                  << info.best_move.to_uci()
                  << " (mcts " << info.best_score
                  << "cp, ab "  << ab.score
                  << "cp, d="   << ab.reached_d
                  << ", nodes=" << ab.nodes << ")"
                  << std::endl;
    }

    return info.best_move;
}

}  // namespace eclipse
