// SPDX-License-Identifier: GPL-3.0-or-later
#include "search.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <utility>

#include "ab.hpp"
#include "mcts.hpp"

namespace eclipse {

namespace {
// Depth ceiling on the AB verifier. Pegged high enough that the time
// budget is the binding constraint in any realistic position.
constexpr int kAbMaxDepth = 32;

// Sequential-mode AB budget: only used when Threads=1, so AB runs AFTER
// MCTS on the same thread. Capped tight because every ms here is a ms
// MCTS doesn't get.
constexpr std::int64_t kAbSeqBudgetPctNum = 10;
constexpr std::int64_t kAbSeqBudgetPctDen = 100;
constexpr std::int64_t kAbSeqBudgetMinMs  = 50;
constexpr std::int64_t kAbSeqBudgetMaxMs  = 1500;

// Override MCTS only when AB sees a clearly better line. 150cp ≈ a piece —
// below that the AB result is plausibly noise vs MCTS's broader positional
// understanding, so we trust MCTS.
constexpr Score kOverrideMarginCp = 150;

void log_ab_outcome(const ab::Result& ab, Move mcts_move, Score mcts_cp, bool overrode) {
    std::cout << "info string AB " << (overrode ? "override" : "verify")
              << ": mcts " << mcts_move.to_uci() << " " << mcts_cp << "cp"
              << " vs ab " << ab.move.to_uci()    << " " << ab.score << "cp"
              << " (d=" << ab.reached_d << ", nodes=" << ab.nodes << ")"
              << std::endl;
}
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

    const int          total_threads = info.threads;
    const std::int64_t total_time_ms = info.limits.time_ms;

    // Parallel AB+MCTS when (a) we have at least 2 threads to split AND
    // (b) there is a non-zero time budget. With time_ms==0 (go depth /
    // go infinite) we can't meaningfully time-share, so fall back to
    // sequential — AB runs AFTER MCTS with a small slice carved out.
    const bool parallel_ab = (total_threads >= 2) && (total_time_ms > 0);

    // Position copy for the AB worker. POD-ish; the Accumulator carries
    // over with computed=true so AB's first NNUE eval is on the fast path.
    // Lives across the whole search() scope so the thread can keep
    // referencing it.
    Position    ab_pos = pos;
    ab::Result  ab_result;
    std::thread ab_thread;

    const int          orig_threads = info.threads;
    const std::int64_t orig_time_ms = info.limits.time_ms;

    if (parallel_ab) {
        // AB takes one thread; MCTS gets the remaining threads. Both run
        // concurrently for the FULL move budget — no sequential reserve.
        info.threads = total_threads - 1;
        ab_thread = std::thread([&ab_result, &ab_pos, total_time_ms]() {
            ab_result = ab::find_best_move(ab_pos, kAbMaxDepth, total_time_ms);
        });
    } else if (total_time_ms > 0) {
        // Sequential mode: reserve 10% (clamped) for the post-MCTS AB run.
        std::int64_t ab_budget = total_time_ms * kAbSeqBudgetPctNum / kAbSeqBudgetPctDen;
        ab_budget = std::clamp(ab_budget, kAbSeqBudgetMinMs, kAbSeqBudgetMaxMs);
        if (ab_budget > total_time_ms / 2) ab_budget = total_time_ms / 2;
        info.limits.time_ms = total_time_ms - ab_budget;
    }

    // ----- MCTS phase -----
    {
        mcts::MCTS mcts_search(pos, info);
        info.best_move = mcts_search.search();
    }
    info.limits.time_ms = orig_time_ms;  // restore for caller introspection
    info.threads        = orig_threads;

    // ----- AB phase -----
    if (parallel_ab) {
        ab_thread.join();
    } else if (total_time_ms > 0) {
        // Run the small reserved slice now, on this thread.
        std::int64_t ab_budget = total_time_ms * kAbSeqBudgetPctNum / kAbSeqBudgetPctDen;
        ab_budget = std::clamp(ab_budget, kAbSeqBudgetMinMs, kAbSeqBudgetMaxMs);
        if (ab_budget > total_time_ms / 2) ab_budget = total_time_ms / 2;
        ab_result = ab::find_best_move(pos, kAbMaxDepth, ab_budget);
    } else {
        // Depth- or infinite-bound: skip AB; just trust MCTS.
        return info.best_move;
    }

    if (ab_result.move == MoveNone || ab_result.reached_d == 0) {
        return info.best_move;
    }

    if (info.best_move != MoveNone &&
        ab_result.move != info.best_move &&
        ab_result.score >= info.best_score + kOverrideMarginCp) {
        log_ab_outcome(ab_result, info.best_move, info.best_score, /*overrode=*/true);
        info.best_move  = ab_result.move;
        info.best_score = ab_result.score;
    } else {
        log_ab_outcome(ab_result, info.best_move, info.best_score, /*overrode=*/false);
    }

    return info.best_move;
}

}  // namespace eclipse
