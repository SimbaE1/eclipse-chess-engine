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

void log_ab_outcome(const ab::Result& ab, Move mcts_move, Score mcts_cp, bool overrode) {
    std::cout << "info string AB " << (overrode ? "override" : "verify")
              << ": mcts " << mcts_move.to_uci() << " " << mcts_cp << "cp"
              << " vs ab " << ab.move.to_uci()    << " " << ab.score << "cp"
              << " (d=" << ab.reached_d << ", nodes=" << ab.nodes << ")"
              << std::endl;
}
}  // namespace

bool SearchInfo::time_up() const noexcept {
    if (limits.nodes > 0 && nodes_searched.load(std::memory_order_relaxed) >= limits.nodes) return true;
    if (limits.depth > 0 && nodes_searched.load(std::memory_order_relaxed) >= limits.depth) return true;
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

    // Forced Move Pruning: if there is only one legal move, just play it immediately.
    // Skips search setup and thread launching overhead.
    MoveList moves;
    generate_legal_moves(pos, moves);
    if (moves.size == 1 && !info.limits.nodes && !info.limits.depth) {
        std::cout << "info string forced move pruning: only " << moves[0].to_uci() << " is legal" << std::endl;
        return moves[0];
    }

    // Parallel AB+MCTS when (a) we have enough threads to dedicate to AB AND
    // (b) there is a non-zero time budget. With time_ms==0 (go depth /
    // go infinite) we can't meaningfully time-share, so fall back to
    // sequential — AB runs AFTER MCTS with a small slice carved out.
    const bool parallel_ab = (total_threads > info.ab_threads) && (info.ab_threads > 0) && (total_time_ms > 0);

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
        // AB takes dedicated threads; MCTS gets the remaining threads. Both run
        // concurrently for the FULL move budget — no sequential reserve.
        info.threads = total_threads - info.ab_threads;
        ab_thread = std::thread([&ab_result, &ab_pos, total_time_ms]() {
            ab_result = ab::find_best_move(ab_pos, kAbMaxDepth, total_time_ms);
        });
    } else if (total_time_ms > 0 && info.ab_threads > 0) {
        // Sequential mode: reserve 10% (clamped) for the post-MCTS AB run.
        std::int64_t ab_budget = total_time_ms * kAbSeqBudgetPctNum / kAbSeqBudgetPctDen;
        ab_budget = std::clamp(ab_budget, kAbSeqBudgetMinMs, kAbSeqBudgetMaxMs);
        if (ab_budget > total_time_ms / 2) ab_budget = total_time_ms / 2;
        info.limits.time_ms = total_time_ms - ab_budget;
    }

    // ----- MCTS phase -----
    mcts::MCTS mcts_search(pos, info);
    mcts_search.run();

    // ----- AB phase (completion) -----
    if (parallel_ab) {
        ab_thread.join();
    } else if (total_time_ms > 0 && info.ab_threads > 0) {
        // Run the small reserved slice now, on this thread.
        std::int64_t ab_budget = total_time_ms * kAbSeqBudgetPctNum / kAbSeqBudgetPctDen;
        ab_budget = std::clamp(ab_budget, kAbSeqBudgetMinMs, kAbSeqBudgetMaxMs);
        if (ab_budget > total_time_ms / 2) ab_budget = total_time_ms / 2;
        ab_result = ab::find_best_move(pos, kAbMaxDepth, ab_budget);
    }

    info.limits.time_ms = orig_time_ms;  // restore for caller introspection
    info.threads        = orig_threads;

    if (ab_result.move != MoveNone && ab_result.reached_d > 0) {
        mcts_search.adjust_root_q(ab_result.move, ab_result.score);
    }

    info.best_move = mcts_search.get_best_move();

    // Hard override check: if AB is significantly better than what MCTS
    // finally chose (even after adjustment), we still take it.
    if (ab_result.move != MoveNone &&
        ab_result.move != info.best_move &&
        ab_result.score >= info.best_score + info.override_margin) {
        log_ab_outcome(ab_result, info.best_move, info.best_score, /*overrode=*/true);
        info.best_move  = ab_result.move;
        info.best_score = ab_result.score;
    } else if (ab_result.move != MoveNone) {
        log_ab_outcome(ab_result, info.best_move, info.best_score, /*overrode=*/false);
    }

    return info.best_move;
}

}  // namespace eclipse
