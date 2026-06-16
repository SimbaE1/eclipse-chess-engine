// SPDX-License-Identifier: GPL-3.0-or-later
#include "search.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <utility>

#include "ab.hpp"
#include "bitboard.hpp"
#include "mcts.hpp"
#include "movegen.hpp"
#include "syzygy.hpp"

namespace eclipse {

namespace {
// Depth ceiling on the AB verifier. Pegged high enough that the time
// budget is the binding constraint in any realistic position.
constexpr int kAbMaxDepth = 32;

// Sequential-mode AB budget: only used when Threads=1, so AB runs AFTER
// MCTS on the same thread. Capped tight because every ms here is a ms
// MCTS doesn't get.
constexpr std::int64_t kAbSeqBudgetPctNum = 5;
constexpr std::int64_t kAbSeqBudgetPctDen = 100;
constexpr std::int64_t kAbSeqBudgetMinMs  = 25;
constexpr std::int64_t kAbSeqBudgetMaxMs  = 500;

// Validation budget: when AB disagrees with MCTS on a non-mate move, carve
// out this fraction of the move budget to run MCTS on the position AFTER
// AB's move. The verdict tells us whether MCTS, given a chance to look at
// the opponent's best replies, still agrees AB's tactic holds — instead of
// blindly trusting AB's depth-limited score. 1/8 of the budget leaves 7/8
// for the main MCTS+AB phase, which is enough headroom in any sane TC.
constexpr std::int64_t kValidationBudgetNum = 1;
constexpr std::int64_t kValidationBudgetDen = 8;
constexpr std::int64_t kValidationBudgetMinMs = 100;

// Mate scores always override regardless of margin. They are objective truth
// (within AB's search depth) and validation cannot disprove them.
bool is_mate_score(Score s) noexcept {
    return s >= kMateInMaxPly || s <= -kMateInMaxPly;
}

void log_ab_outcome(const ab::Result& ab, Move mcts_move, Score mcts_cp, const char* tag) {
    std::cout << "info string AB " << tag
              << ": mcts " << mcts_move.to_uci() << " " << mcts_cp << "cp"
              << " vs ab " << ab.move.to_uci()    << " " << ab.score << "cp"
              << " (d=" << ab.reached_d << ", nodes=" << ab.nodes << ")"
              << std::endl;
}
}  // namespace

bool SearchInfo::time_up() const noexcept {
    if (limits.nodes > 0 && static_cast<std::uint64_t>(nodes_searched.load(std::memory_order_relaxed)) >= limits.nodes) return true;
    if (limits.depth > 0 && nodes_searched.load(std::memory_order_relaxed) >= limits.depth) return true;
    if (limits.ponder) return false;
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
        info.best_move = moves[0];
        std::cout << "bestmove " << info.best_move.to_uci() << std::endl;
        return moves[0];
    }

    // Root Syzygy probe: if all pieces fit in the tablebase, skip MCTS/AB
    // entirely and return the DTZ-optimal move immediately.
    if (syzygy::is_enabled() && !info.limits.ponder && !info.limits.infinite &&
        pos.castling_rights() == NoCastling &&
        static_cast<unsigned>(popcount(pos.occupied())) <= syzygy::max_pieces()) {
        const syzygy::RootBest rb = syzygy::probe_root_best(pos);
        if (rb.wdl != syzygy::kTbFailed) {
            // TB_PROMOTES_* → Eclipse PieceType
            static constexpr PieceType kPromoMap[5] = {
                NoPieceType, Queen, Rook, Bishop, Knight
            };
            const auto from_sq = static_cast<Square>(rb.from);
            const auto to_sq   = static_cast<Square>(rb.to);
            const PieceType promo = kPromoMap[std::min(rb.promotes, 4u)];
            Move tb_move = MoveNone;
            for (int i = 0; i < moves.size; ++i) {
                const Move m = moves[i];
                if (m.from() != from_sq || m.to() != to_sq) continue;
                if (m.type() == Move::Promotion) {
                    if (m.promotion_piece() != promo) continue;
                } else if (promo != NoPieceType) {
                    continue;
                }
                tb_move = m;
                break;
            }
            if (tb_move != MoveNone) {
                const char* wdl_str =
                    rb.wdl == syzygy::kTbWin         ? "win"
                  : rb.wdl == syzygy::kTbCursedWin   ? "cursed-win"
                  : rb.wdl == syzygy::kTbDraw         ? "draw"
                  : rb.wdl == syzygy::kTbBlessedLoss  ? "blessed-loss"
                  :                                      "loss";
                std::cout << "info string TB root: wdl=" << wdl_str
                          << " dtz=" << rb.dtz << std::endl;
                info.best_move = tb_move;
                std::cout << "bestmove " << tb_move.to_uci() << std::endl;
                return tb_move;
            }
        }
    }

    // Parallel AB+MCTS when (a) we have enough threads to dedicate to AB AND
    // (b) there is a non-zero time budget. With time_ms==0 (go depth /
    // go infinite) we can't meaningfully time-share, so fall back to
    // sequential — AB runs AFTER MCTS with a small slice carved out.
    const bool parallel_ab = (total_threads > info.ab_threads) && (info.ab_threads > 0) && (total_time_ms > 0);

    // Reserve a slice of the budget for the post-search validation phase
    // (focused MCTS at the position after AB's recommended move). Only
    // carved out when there's time to spare; otherwise we skip validation
    // and fall back to a simple agree/disagree check at the end.
    const bool                     do_validation_phase = (total_time_ms >= 400);
    const std::int64_t             validation_budget   = do_validation_phase
        ? std::max(kValidationBudgetMinMs,
                   total_time_ms * kValidationBudgetNum / kValidationBudgetDen)
        : 0;
    const std::int64_t main_phase_time = total_time_ms - validation_budget;

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
        // concurrently for the main phase (7/8 of the budget); the remaining
        // 1/8 is reserved for validation if AB disagrees with MCTS.
        info.threads        = total_threads - info.ab_threads;
        info.limits.time_ms = main_phase_time;
        ab_thread = std::thread([&ab_result, &ab_pos, main_phase_time]() {
            ab_result = ab::find_best_move(ab_pos, kAbMaxDepth, main_phase_time);
        });
    } else if (total_time_ms > 0 && info.ab_threads > 0) {
        // Sequential mode: reserve 10% (clamped) for the post-MCTS AB run,
        // plus the validation slice if it's enabled.
        std::int64_t ab_budget = total_time_ms * kAbSeqBudgetPctNum / kAbSeqBudgetPctDen;
        ab_budget = std::clamp(ab_budget, kAbSeqBudgetMinMs, kAbSeqBudgetMaxMs);
        if (ab_budget > total_time_ms / 2) ab_budget = total_time_ms / 2;
        if (ab_budget < 1) ab_budget = 1;
        info.limits.time_ms = main_phase_time - ab_budget;
        if (info.limits.time_ms < 1) info.limits.time_ms = 1;
    } else if (do_validation_phase) {
        info.limits.time_ms = main_phase_time;
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
        if (ab_budget < 1) ab_budget = 1;
        ab_result = ab::find_best_move(pos, kAbMaxDepth, ab_budget);
    }

    info.limits.time_ms = orig_time_ms;  // restore for caller introspection
    info.threads        = orig_threads;

    if (ab_result.move != MoveNone && ab_result.reached_d > 0) {
        mcts_search.adjust_root_q(ab_result.move, ab_result.score);
    }

    info.best_move = mcts_search.get_best_move();

    // Collect potential ponder moves before save_to_cache() nulls the root.
    const Move pre_reconcile_best = info.best_move;
    const Move mcts_ponder = mcts_search.get_ponder_move_after(pre_reconcile_best);
    const Move ab_ponder   = (ab_result.move != MoveNone && ab_result.move != pre_reconcile_best)
        ? mcts_search.get_ponder_move_after(ab_result.move)
        : MoveNone;

    mcts_search.save_to_cache();  // preserve subtree for tree reuse next move

    // ----- AB-vs-MCTS reconciliation -----
    //
    // Three cases when AB suggests a different move than MCTS:
    //
    //  (1) AB's score is a mate. Mate is objective truth (within AB's depth),
    //      not subject to validation. Play AB's move unconditionally.
    //
    //  (2) Validation phase is on AND there's a non-trivial advantage AB
    //      claims. Carve out the reserved slice to run focused MCTS at the
    //      position AFTER AB's first move. The opponent moves at that
    //      position, so MCTS naturally explores their best refutations. If
    //      MCTS's verdict (negated to our POV) still beats what we'd get from
    //      MCTS's original choice, AB's tactic is real and we play it.
    //      Otherwise the opponent has a hidden refutation MCTS sees but AB
    //      missed — keep MCTS's pick.
    //
    //  (3) Validation off OR margin too thin. Just verify-log; no override.
    //
    if (ab_result.move != MoveNone && ab_result.move != info.best_move) {
        if (is_mate_score(ab_result.score)) {
            log_ab_outcome(ab_result, info.best_move, info.best_score, "mate-override");
            info.best_move  = ab_result.move;
            info.best_score = ab_result.score;
        } else if (do_validation_phase &&
                   ab_result.score >= info.best_score + info.override_margin) {
            // Run focused MCTS at the post-AB-move position. Opponent to move
            // there, so the returned best_score is from THEIR perspective.
            Position  validation_pos = pos;
            StateInfo vst;
            validation_pos.do_move(ab_result.move, vst);

            SearchInfo validation_info;
            validation_info.threads        = total_threads;  // all threads, focused
            validation_info.ab_threads     = 0;              // pure MCTS verdict
            validation_info.limits.time_ms = validation_budget;
            validation_info.start_time     = std::chrono::steady_clock::now();
            validation_info.silent         = true;           // don't emit info lines from opponent's POV

            mcts::MCTS validator(validation_pos, validation_info);
            validator.run();
            (void) validator.get_best_move();  // populates validation_info.best_score

            // validation_info.best_score is from the opponent's perspective
            // (validation_pos's STM). Negate to get the position's eval from
            // our root perspective.
            const Score validated_cp = static_cast<Score>(-validation_info.best_score);

            std::cout << "info string AB validation: ab " << ab_result.move.to_uci()
                      << " " << ab_result.score << "cp (d=" << ab_result.reached_d
                      << ")  mcts-verdict-after-move " << validated_cp << "cp"
                      << "  vs mcts-original " << info.best_score << "cp"
                      << std::endl;

            if (validated_cp >= info.best_score) {
                log_ab_outcome(ab_result, info.best_move, info.best_score, "override (validated)");
                info.best_move  = ab_result.move;
                info.best_score = validated_cp;
            } else {
                log_ab_outcome(ab_result, info.best_move, info.best_score, "refuted (kept mcts)");
            }
        } else {
            log_ab_outcome(ab_result, info.best_move, info.best_score, "verify");
        }
    } else if (ab_result.move != MoveNone) {
        log_ab_outcome(ab_result, info.best_move, info.best_score, "verify");
    }

    // Pick ponder move based on which move survived reconciliation.
    const Move ponder_move = (info.best_move == pre_reconcile_best)
        ? mcts_ponder
        : (info.best_move == ab_result.move ? ab_ponder : MoveNone);

    if (info.best_move == MoveNone) {
        std::cout << "bestmove 0000" << std::endl;
    } else if (ponder_move != MoveNone) {
        std::cout << "bestmove " << info.best_move.to_uci()
                  << " ponder " << ponder_move.to_uci() << std::endl;
    } else {
        std::cout << "bestmove " << info.best_move.to_uci() << std::endl;
    }

    return info.best_move;
}

}  // namespace eclipse
