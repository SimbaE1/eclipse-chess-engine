// SPDX-License-Identifier: GPL-3.0-or-later
#include "search.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <thread>
#include <utility>

#include "ab.hpp"
#include "bitboard.hpp"
#include "mcts.hpp"
#include "movegen.hpp"
#include "syzygy.hpp"
#include "thread_util.hpp"

namespace eclipse {

namespace {
// Depth ceiling on the AB verifier. Pegged high enough that the time
// budget is the binding constraint in any realistic position.
constexpr int kAbMaxDepth = 32;

// Sequential-mode AB budget: only used when Threads=1, so AB runs AFTER
// MCTS on the same thread. Set to 1/4 of the move so AB gets the same
// compute share it would in parallel mode (1 dedicated thread per 4 ==
// 25% of cores for the whole search). The old 5%/500ms-cap starved the
// tactical verifier at 1 thread relative to how it's funded at 4+. The
// total/2 guard at the use sites still prevents AB from ever taking the
// majority of the clock; kAbSeqBudgetMinMs keeps a usable floor at fast TCs.
constexpr std::int64_t kAbSeqBudgetPctNum = 1;
constexpr std::int64_t kAbSeqBudgetPctDen = 4;
constexpr std::int64_t kAbSeqBudgetMinMs  = 25;

// Validation budget: when AB disagrees with MCTS on a non-mate move, carve
// out this fraction of the move budget to run MCTS on the position AFTER
// AB's move. The verdict tells us whether MCTS, given a chance to look at
// the opponent's best replies, still agrees AB's tactic holds — instead of
// blindly trusting AB's depth-limited score. 1/8 of the budget leaves 7/8
// for the main MCTS+AB phase, which is enough headroom in any sane TC.
constexpr std::int64_t kValidationBudgetNum = 1;
constexpr std::int64_t kValidationBudgetDen = 8;
constexpr std::int64_t kValidationBudgetMinMs = 100;

// Tactic-trace budget: a dedicated AB iterative-deepening pass at the
// post-AB-move position, looking for the depth at which a refutation/tactic
// is first discovered (see ab::find_tactic_node). Separate from the warm-TT
// reuse the cross-check gets — this needs real depth, not just a quick read,
// so it gets its own modest slice rather than piggybacking on cached work.
constexpr std::int64_t kTacticTraceBudgetMs = 1500;
// Cheaper budget for the detection-side trace (search.cpp's disagreement
// check, run only when AB's own pick already differs from MCTS's but the
// fast cross-check didn't flag a problem) — a quick second opinion, not a
// thorough confirmation; the validation step's own trace gets the fuller
// budget above.
constexpr std::int64_t kDetectionTraceBudgetMs = 600;
// The detection-side trace runs AFTER the normal search budget is already
// spent -- it's extra wall-clock on top of whatever movetime was requested,
// not carved out of it. Fine in rapid/classical; risky in blitz/bullet
// where that overhead could itself cause time trouble. Skip it below this
// per-move budget so it only ever fires when there's real room for it.
constexpr std::int64_t kMinTimeForDeepDetectionMs = 3000;
// Virtual visits backed into a seeded tactic node: large enough to bias
// PUCT meaningfully on first encounter, small enough that a handful of real
// visits can override it if deeper search there disagrees with AB's read.
constexpr int kTacticSeedVisits = 8;

// Mate scores always override regardless of margin. They are objective truth
// (within AB's search depth) and validation cannot disprove them.
bool is_mate_score(Score s) noexcept {
    return s >= kMateInMaxPly || s <= -kMateInMaxPly;
}

// Cross-check budget: cheap because it reuses the just-finished AB search's
// warm TT — most of the relevant subtree is already resolved, this just
// asks for an honest score on a SPECIFIC move that AB's own root loop may
// only have given a cheap alpha-beta bound to (see ab::score_move).
constexpr std::int64_t kCrossCheckBudgetMs = 150;

// Reconciliation extension: if AB and MCTS both disagree with each other's
// pick even after validation, and there's slack left in the clock (see
// SearchLimits::extra_budget_ms), spend a bounded amount of it deepening
// both searchers instead of guessing. Capped so a genuinely murky position
// can't eat the whole remaining clock on one move.
constexpr std::int64_t kMinExtensionMs = 300;
constexpr int          kMaxExtensionRounds = 2;

// Eval-instability detection. A move can look fine at shallow depth and erode
// the deeper a searcher looks -- the Ba4-style slow trap (+98cp at d7, +46cp
// at d13, -280cp by SF d22). When AB's root score for its pick is still
// drifting at the deepest completed depth, the shallow read isn't trustworthy;
// if there's clock slack we deepen both searchers and re-check rather than
// committing. Tuned conservatively against AB's per-depth trajectory.
constexpr int   kMinDepthsForTrend = 6;   // need this many completed depths to judge
constexpr Score kUnsettledStepCp   = 45;  // a jump this big at the last ply = not settled
constexpr int   kUnsettledWindow   = 5;   // # of depth-to-depth transitions to scan
constexpr Score kUnsettledDriftCp  = 40;  // net drift over the window to count as unsettled
constexpr Score kStillMovingCp     = 12;  // recent spread below this = converged, leave it

// True if `ds` (root scores per completed depth) hasn't converged. Real AB
// trajectories are noisy (aspiration re-searches spike single plies), so we
// don't require strict monotonicity -- we look for a substantial NET drift
// over the recent window that is STILL moving at the deepest plies. That
// catches the slow-burn trap (Ba4: 179@d1 -> 84@d8 -> ~50@d11, despite a
// noisy d6 spike) while leaving genuinely settled evals alone. Mate/TB tails
// are treated as settled -- a deeper look has nothing to add.
bool eval_unsettled(const std::vector<Score>& ds) {
    const int n = static_cast<int>(ds.size());
    if (n < kMinDepthsForTrend) return false;
    if (ds[static_cast<std::size_t>(n - 1)] >= kMateInMaxPly ||
        ds[static_cast<std::size_t>(n - 1)] <= -kMateInMaxPly) return false;

    // (1) Big single-ply move at the deepest completed depth: clearly unsettled.
    if (std::abs(static_cast<int>(ds[static_cast<std::size_t>(n - 1)]) -
                 static_cast<int>(ds[static_cast<std::size_t>(n - 2)])) >= kUnsettledStepCp)
        return true;

    // (2) Net drift over the window AND still moving (recent plies not in a
    //     tight band). Both conditions are needed: net drift alone fires on a
    //     move that rose then settled; the "still moving" spread gates that out.
    const int w   = std::min(n - 1, kUnsettledWindow);
    const int net = static_cast<int>(ds[static_cast<std::size_t>(n - 1)]) -
                    static_cast<int>(ds[static_cast<std::size_t>(n - 1 - w)]);
    int lo = ds[static_cast<std::size_t>(n - 1)], hi = lo;
    for (int i = std::max(0, n - 3); i < n; ++i) {
        lo = std::min(lo, static_cast<int>(ds[static_cast<std::size_t>(i)]));
        hi = std::max(hi, static_cast<int>(ds[static_cast<std::size_t>(i)]));
    }
    const bool still_moving = (hi - lo) >= kStillMovingCp;
    return std::abs(net) >= kUnsettledDriftCp && still_moving;
}

void log_ab_outcome(const ab::Result& ab, Move mcts_move, Score mcts_cp, const char* tag) {
    const auto ab_nps = ab.nodes * 1000 / std::max<std::int64_t>(1, ab.elapsed_ms);
    std::cout << "info string AB " << tag
              << ": mcts " << mcts_move.to_uci() << " " << mcts_cp << "cp"
              << " vs ab " << ab.move.to_uci()    << " " << ab.score << "cp"
              << " (d=" << ab.reached_d << ", nodes=" << ab.nodes << ", nps=" << ab_nps << ")"
              << std::endl;
}
}  // namespace

bool SearchInfo::time_up() const noexcept {
    // External abort (parent search's stop, for internal sub-searches). Checked
    // first so a ponder miss / `stop` / `quit` aborts the sub-search before any
    // budget arithmetic — it has no clock of its own to fall back on.
    if (ext_stop && ext_stop->load(std::memory_order_relaxed)) return true;
    if (limits.nodes > 0 && static_cast<std::uint64_t>(nodes_searched.load(std::memory_order_relaxed)) >= limits.nodes) return true;
    if (limits.depth > 0 && nodes_searched.load(std::memory_order_relaxed) >= limits.depth) return true;
    // Absolute hard ceiling: the SUM of all phases must never cross this,
    // whatever per-phase budget is currently active (extensions reset
    // start_time, so a start_time-relative check alone would not catch an
    // overrun). Checked first and unconditionally — it must override even the
    // ponder path so a ponderhit that arrives near the deadline can't restart
    // a full think we don't have time for.
    if (hard_deadline.time_since_epoch().count() != 0 &&
        std::chrono::steady_clock::now() >= hard_deadline) return true;
    if (limits.ponder) {
        // ponderhit_at_ms is the only cross-thread ponder signal (see
        // search.hpp) — never touch start_time or limits.ponder here. The hard
        // ceiling for a ponder search is measured from the ponderhit instant
        // (not from when pondering began), so it isn't expressed via the
        // absolute hard_deadline above — which stays unset for ponder searches.
        const auto ph = ponderhit_at_ms.load(std::memory_order_acquire);
        if (ph < 0) return false;  // still pondering, no ponderhit yet
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        const auto since_hit = now_ms - ph;
        if (since_hit >= limits.time_ms) return true;
        return limits.hard_limit_ms > 0 && since_hit >= limits.hard_limit_ms;
    }
    if (limits.infinite || limits.time_ms <= 0) return false;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();
    return elapsed >= limits.time_ms;
}

std::int64_t SearchInfo::ms_until_hard_deadline() const noexcept {
    constexpr std::int64_t kUnbounded = std::numeric_limits<std::int64_t>::max() / 4;

    // A pending stop (ponder miss, `stop`, or `quit`) means "abort now": report
    // zero time left so every deadline-clamped phase below — the reconciliation
    // loop, the AB cross-check/tactic probes, and try_extend — bails out instead
    // of running expensive work whose result will be discarded. This is critical
    // for the ponder path: a ponder search that never received a ponderhit
    // otherwise reports kUnbounded (its ceiling is measured from the hit), so a
    // stopped ponder would keep deepening; and because the next `go` joins this
    // thread before starting, that wasted work is billed to the real move's
    // clock. That is exactly how a won game flagged on 2026-06-18.
    if (stop.load(std::memory_order_relaxed)) return 0;

    // Ponder searches carry the ceiling relative to the ponderhit instant
    // rather than as an absolute deadline (see time_up()): unbounded while
    // still pondering, then hard_limit_ms minus time since the hit.
    if (limits.ponder) {
        if (limits.hard_limit_ms <= 0) return kUnbounded;
        const auto ph = ponderhit_at_ms.load(std::memory_order_acquire);
        if (ph < 0) return kUnbounded;  // still pondering
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        const auto left = limits.hard_limit_ms - (now_ms - ph);
        return left > 0 ? left : 0;
    }

    // No hard deadline configured: return a large value so callers can min()
    // their fixed budgets against it without changing behaviour.
    if (hard_deadline.time_since_epoch().count() == 0) return kUnbounded;
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
        hard_deadline - std::chrono::steady_clock::now()).count();
    return left > 0 ? left : 0;
}

Move search(Position& pos, SearchInfo& info) {
    info.nodes_searched.store(0, std::memory_order_relaxed);
    info.start_time     = std::chrono::steady_clock::now();
    // Turn the hard ceiling into a single absolute deadline. Every phase below
    // clamps its budget to info.ms_until_hard_deadline(), and time_up() trips
    // once we pass it, so the total wall time for the move can never exceed
    // limits.hard_limit_ms regardless of how many phases/extensions run. Left
    // at epoch (no deadline) for movetime/depth/nodes/infinite searches.
    // Ponder searches are excluded here: their ceiling is measured from the
    // ponderhit instant inside time_up()/ms_until_hard_deadline(), not from
    // now, so the absolute deadline stays unset for them.
    info.hard_deadline = (info.limits.hard_limit_ms > 0 && !info.limits.ponder)
        ? info.start_time + std::chrono::milliseconds(info.limits.hard_limit_ms)
        : std::chrono::steady_clock::time_point{};
    info.ponderhit_at_ms.store(-1, std::memory_order_relaxed);
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
    // entirely and return the DTZ-optimal move immediately. Runs for ponder
    // searches too (a TB position is solved instantly), but a ponder search
    // holds the move until ponderhit — see below — so it can't emit bestmove
    // mid-ponder. Previously this was skipped entirely when pondering, so on
    // every pondered move the engine fell back to search and could fail to
    // convert a won tablebase position.
    if (syzygy::is_enabled() && !info.limits.infinite &&
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
                // A ponder search must not emit bestmove until the ponderhit
                // arrives (or we're told to stop) — UCI forbids it mid-ponder.
                // The TB move is already known, so just wait for the signal and
                // release it instantly: full tablebase play, no wasted search.
                if (info.limits.ponder) {
                    while (info.ponderhit_at_ms.load(std::memory_order_acquire) < 0 &&
                           !info.stop.load(std::memory_order_relaxed)) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
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
    // BigThread (16 MB stack), NOT std::thread: this worker runs ab::find_best_move,
    // whose negamax/qsearch recursion overflows macOS's 512 KB default secondary
    // stack on sharp mid-game positions -> SIGBUS (exit -10). See thread_util.hpp.
    BigThread   ab_thread;

    const int          orig_threads = info.threads;
    const std::int64_t orig_time_ms = info.limits.time_ms;

    if (parallel_ab) {
        // AB takes dedicated threads; MCTS gets the remaining threads. Both run
        // concurrently for the main phase (7/8 of the budget); the remaining
        // 1/8 is reserved for validation if AB disagrees with MCTS.
        info.threads        = total_threads - info.ab_threads;
        info.limits.time_ms = main_phase_time;
        const int ab_phase_threads = info.ab_threads;  // AB's reserved share
        const std::atomic<bool>* ab_stop = &info.stop;
        // This worker is launched at `go ponder` (when pondering begins). Make
        // its budget ponder-aware so it spends main_phase_time AFTER the
        // ponderhit — matching the (already ponder-aware) MCTS phase — instead
        // of exhausting it during the opponent's free think and leaving the
        // post-hit verify with no AB read. Null for a non-ponder search => plain
        // start-relative budget.
        const std::atomic<std::int64_t>* ab_ponder =
            info.limits.ponder ? &info.ponderhit_at_ms : nullptr;
        ab_thread = BigThread([&ab_result, &ab_pos, main_phase_time, ab_phase_threads, ab_stop, ab_ponder]() {
            ab_result = ab::find_best_move(ab_pos, kAbMaxDepth, main_phase_time, ab_phase_threads, ab_stop, ab_ponder);
        });
    } else if (total_time_ms > 0 && info.ab_threads > 0) {
        // Sequential mode: reserve 1/4 of the move for the post-MCTS AB run
        // (see kAbSeqBudget* above), plus the validation slice if enabled.
        std::int64_t ab_budget = total_time_ms * kAbSeqBudgetPctNum / kAbSeqBudgetPctDen;
        if (ab_budget < kAbSeqBudgetMinMs) ab_budget = kAbSeqBudgetMinMs;
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

    // Stats for MCTS's OWN phase, kept separate from AB's (printed below once
    // AB completes) so the two searchers' depth/nodes/nps are never conflated
    // -- they run concurrently (parallel_ab) or over different slices
    // (sequential), so a single combined number would be meaningless.
    if (!info.silent) {
        const auto mcts_elapsed_ms = std::max<std::int64_t>(1,
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - info.start_time).count());
        const auto mcts_nodes = info.nodes_searched.load(std::memory_order_relaxed);
        std::cout << "info string MCTS stats: depth=" << mcts_search.reached_depth()
                  << " nodes=" << mcts_nodes
                  << " nps=" << (mcts_nodes * 1000 / mcts_elapsed_ms)
                  << std::endl;
    }

    // ----- AB phase (completion) -----
    if (parallel_ab) {
        ab_thread.join();
    } else if (total_time_ms > 0 && info.ab_threads > 0) {
        // Run the reserved 1/4 slice now, on this thread. MCTS has finished,
        // so AB can use every thread (Lazy SMP) for the deepest read possible.
        std::int64_t ab_budget = total_time_ms * kAbSeqBudgetPctNum / kAbSeqBudgetPctDen;
        if (ab_budget < kAbSeqBudgetMinMs) ab_budget = kAbSeqBudgetMinMs;
        if (ab_budget > total_time_ms / 2) ab_budget = total_time_ms / 2;
        if (ab_budget < 1) ab_budget = 1;
        ab_result = ab::find_best_move(pos, kAbMaxDepth, ab_budget, total_threads, &info.stop);
    }

    info.limits.time_ms = orig_time_ms;  // restore for caller introspection
    info.threads        = orig_threads;

    if (ab_result.move != MoveNone && ab_result.reached_d > 0) {
        mcts_search.adjust_root_q(ab_result.move, ab_result.score);
    }

    info.best_move = mcts_search.get_best_move();

    // ----- AB-vs-MCTS reconciliation -----
    //
    // AB's own root loop only gives an exact score to its top-ordered move;
    // every other root move (MCTS's pick, when they disagree, is rarely
    // AB's top-ordered move) only gets a cheap alpha-beta bound. So instead
    // of just comparing "is AB's own favorite better than MCTS's", we ask AB
    // directly what it thinks of the move MCTS actually wants to play
    // (ab::score_move, reusing AB's now-warm TT — cheap). That catches AB
    // quietly knowing a move is bad even when AB's own alternative doesn't
    // look dramatically better in cp terms.
    //
    // Cases:
    //  (1) AB's score for MCTS's move is a mate (we get mated). Objective
    //      truth within AB's depth — take AB's own pick unconditionally if
    //      it disagrees, no validation needed.
    //  (2) AB's score for MCTS's move is far worse than MCTS's own claim
    //      (or AB has a different pick at all that beats the margin):
    //      validate AB's alternative via focused MCTS at the position after
    //      it, same as before. Validated -> override. Refuted -> see (3).
    //  (3) Both refuted (AB's view of MCTS's move says "bad", but MCTS's
    //      counter-check says AB's fix doesn't hold either): if there's
    //      still slack in the clock (extra_budget_ms), spend a bounded slice
    //      deepening both searchers and re-check, instead of guessing.
    //      Otherwise keep MCTS's move and log loudly — this is a real
    //      "neither searcher trusts the position" signal worth tracking.
    //  (4) No disagreement at all: log the cross-check anyway (always-on
    //      telemetry on how often/where the two searchers diverge, useful
    //      for prioritizing the underlying eval-quality fix later).
    Position ab_pos_cc = pos;  // scratch copy for score_move (mutates internally but restores)
    int extension_rounds = 0;

    // The AB cross-check / tactic-trace probes below run on their OWN clock
    // (ab::SearchCtx), so they can't see info.hard_deadline. Clamp every probe
    // budget to the time actually left before the deadline so they can't push
    // the move past it. Always >= 1 (a 0 budget means "unlimited" to ab.cpp).
    auto probe_budget = [&](std::int64_t want) -> std::int64_t {
        return std::max<std::int64_t>(1, std::min(want, info.ms_until_hard_deadline()));
    };

    // Spend a bounded slice of clock slack deepening BOTH searchers, then let
    // the loop re-check. Shared by the "both sides refuted each other" case and
    // the "they agree but the eval is still drifting" case. Returns false (no
    // extension performed) when the round cap or slack budget is exhausted.
    auto try_extend = [&](const char* reason) -> bool {
        // Never extend past the hard deadline: cap the slice at what's left of
        // the move's total budget. With nothing left to spend, don't extend —
        // returning false makes the caller commit to the current pick.
        const std::int64_t time_left = info.ms_until_hard_deadline();
        if (extension_rounds >= kMaxExtensionRounds ||
            info.limits.extra_budget_ms < kMinExtensionMs ||
            time_left < kMinExtensionMs) {
            return false;
        }
        const std::int64_t chunk = std::min({info.limits.extra_budget_ms,
                                             std::max<std::int64_t>(orig_time_ms, kMinExtensionMs),
                                             time_left});
        info.limits.extra_budget_ms -= chunk;
        ++extension_rounds;
        std::cout << "info string AB/MCTS reconciliation: extending by " << chunk
                  << "ms (" << reason << ", round " << extension_rounds << ", "
                  << info.limits.extra_budget_ms << "ms slack left)" << std::endl;

        // Extend MCTS on the SAME tree (more visits on the same position).
        info.limits.time_ms = chunk / 2;
        info.start_time     = std::chrono::steady_clock::now();
        mcts_search.run(/*keep_existing_root=*/true);
        info.best_move = mcts_search.get_best_move();

        // Extend AB. MCTS has just finished its slice above and is idle, so AB
        // gets ALL threads here (Lazy SMP) -- this focused deeper re-check is
        // exactly where the otherwise-idle cores buy depth, the whole point of
        // extending. Its global TT is already warm, so early depths resolve
        // almost instantly and it picks up roughly where it left off.
        ab_result = ab::find_best_move(pos, kAbMaxDepth, chunk - chunk / 2, total_threads, &info.stop);
        return true;
    };

    bool resolved = false;
    while (!resolved) {
        resolved = true;  // unless the extension branch below says otherwise

        if (ab_result.move == MoveNone) break;  // AB not running at all (ab_threads==0)

        // Aborting (ponder miss / stop / quit): commit to the current best move
        // immediately. Reconciliation results would be discarded, and dawdling
        // here delays the join that gates the next real search.
        if (info.stop.load(std::memory_order_relaxed)) break;

        // Out of time for reconciliation: commit to MCTS's move rather than
        // start probes/validation we can't afford. The hard deadline already
        // reserves a latency margin, so stopping here keeps us from flagging.
        // (ms_until_hard_deadline() also returns 0 once stop is set.)
        if (info.ms_until_hard_deadline() <= 0) break;

        // Before trusting AB's read at all -- whether to confirm MCTS's move or
        // to override with AB's own pick -- make sure that read has actually
        // settled. If AB's score is still drifting with depth and there's clock
        // slack, deepen both searchers and re-check first. A shallow read here
        // is exactly what commits to a slow-burn trap (the Ba4 case: AB's own
        // d8 pick is Ba4 at +84cp, but it erodes to ~+50 by d11 and AB switches
        // to Nf5). Runs whether or not AB and MCTS currently agree.
        if (eval_unsettled(ab_result.depth_scores) && try_extend("eval unsettled")) {
            resolved = false;
            continue;
        }

        const Score ab_view_of_mcts_move =
            ab::score_move(ab_pos_cc, info.best_move, kAbMaxDepth, probe_budget(kCrossCheckBudgetMs));

        // "Bad" means AB sees a problem with MCTS's specific move — a
        // confirmed loss, or a score well below what MCTS itself claims.
        // A winning mate on MCTS's own move is NOT a disagreement; there's
        // nothing to fix.
        bool ab_views_mcts_move_as_bad =
            (is_mate_score(ab_view_of_mcts_move) && ab_view_of_mcts_move < 0) ||
            ab_view_of_mcts_move <= info.best_score - info.override_margin;
        const bool ab_has_better_alternative =
            ab_result.score >= info.best_score + info.override_margin;

        // A tactic found in the line AFTER MCTS's move = a refutation of that
        // move (the opponent wins). Distinct from a tactic after AB's move,
        // which would mean AB's move is the improvement. We track this one
        // separately so the validation phase can judge AB's alternative against
        // what MCTS's move is *actually* worth once refuted -- rather than
        // assuming the disagreement means AB's move is better.
        ab::TacticNode mcts_refutation;

        // The cheap cross-check above (kCrossCheckBudgetMs, warm-TT reuse)
        // can miss a refutation that's genuinely several plies deep -- it's
        // built for speed, not depth. Before concluding "no disagreement",
        // if AB already has SOME independent opinion here (its own pick
        // differs) but the cheap check didn't catch why, give it a second,
        // deeper look specifically for a stable "aha" a quick read would
        // miss. Gated on ab_result.move != info.best_move so this never
        // runs in the common case where AB and MCTS already agree.
        if (!ab_views_mcts_move_as_bad && ab_result.move != info.best_move &&
            orig_time_ms >= kMinTimeForDeepDetectionMs) {
            Position deeper_pos = ab_pos_cc;
            StateInfo deeper_st;
            deeper_pos.do_move(info.best_move, deeper_st);
            mcts_refutation =
                ab::find_tactic_node(deeper_pos, kAbMaxDepth, probe_budget(kDetectionTraceBudgetMs));
            // root_score_cp is from deeper_pos's own STM (the opponent,
            // since we just played info.best_move) -- negate once to get it
            // back to our/root perspective, matching info.best_score's
            // convention, before comparing against the same margin the
            // cheap check uses.
            const Score our_perspective_cp = static_cast<Score>(-mcts_refutation.root_score_cp);
            if (mcts_refutation.found && our_perspective_cp <= info.best_score - info.override_margin) {
                ab_views_mcts_move_as_bad = true;
                std::cout << "info string AB deep cross-check: mcts " << info.best_move.to_uci()
                          << " hides a refutation at depth " << mcts_refutation.aha_depth
                          << " (" << our_perspective_cp << "cp vs claimed " << info.best_score << "cp), path=";
                for (const Move m : mcts_refutation.path) std::cout << m.to_uci() << " ";
                std::cout << std::endl;
            }
        }

        const bool ab_disagrees =
            ab_result.move != info.best_move &&
            (ab_views_mcts_move_as_bad || ab_has_better_alternative);

        const auto ab_nps = ab_result.nodes * 1000 /
            std::max<std::int64_t>(1, ab_result.elapsed_ms);
        std::cout << "info string AB cross-check: mcts " << info.best_move.to_uci()
                  << " " << info.best_score << "cp"
                  << " (ab's view " << ab_view_of_mcts_move << "cp)"
                  << "  ab's own pick " << ab_result.move.to_uci()
                  << " " << ab_result.score << "cp"
                  << " (d=" << ab_result.reached_d << ", nodes=" << ab_result.nodes
                  << ", nps=" << ab_nps << ")"
                  << std::endl;

        if (!ab_disagrees) {
            log_ab_outcome(ab_result, info.best_move, info.best_score, "verify");
            break;
        }

        if (is_mate_score(ab_view_of_mcts_move) && ab_view_of_mcts_move < 0) {
            log_ab_outcome(ab_result, info.best_move, info.best_score, "mate-override");
            info.best_move  = ab_result.move;
            info.best_score = ab_result.score;
            break;
        }

        if (do_validation_phase) {
            // Run focused MCTS at the post-AB-move position. Opponent to move
            // there, so the returned best_score is from THEIR perspective.
            Position  validation_pos = pos;
            StateInfo vst;
            validation_pos.do_move(ab_result.move, vst);

            SearchInfo validation_info;
            validation_info.threads        = total_threads;  // all threads, focused
            validation_info.ab_threads     = 0;              // pure MCTS verdict
            // Cap the validation MCTS at whatever is left before the deadline.
            const std::int64_t remaining_ms = info.ms_until_hard_deadline();
            validation_info.limits.time_ms = std::min<std::int64_t>(validation_budget, remaining_ms);
            validation_info.start_time     = std::chrono::steady_clock::now();
            // Bind an ABSOLUTE deadline for the validator, derived from the time
            // actually left on the move — do NOT just copy info.hard_deadline.
            // For a ponder search that field is epoch (unset; the ponder ceiling
            // lives in time_up()'s ponder branch, which a fresh SearchInfo can't
            // use), and the validator's own time_ms can be 0 (budget already
            // spent), which time_up() reads as "no limit" — so the validator
            // would spin until the MCTS node pool fills, overrunning the move by
            // 100s+ and flagging won games (the recurring ponder-path bug).
            // ms_until_hard_deadline() already folds in the ponder math and
            // returns 0 once stop is set, so this works for ponder and non-
            // ponder alike and collapses to ~now (immediate stop) when nothing
            // is left. Clamp to hard_limit_ms so the kUnbounded sentinel (untimed
            // analysis) can't overflow the time_point; left epoch when untimed.
            if (info.limits.hard_limit_ms > 0) {
                const std::int64_t left = std::clamp<std::int64_t>(remaining_ms, 0,
                                                                   info.limits.hard_limit_ms);
                validation_info.hard_deadline = validation_info.start_time +
                                                std::chrono::milliseconds(left);
            }
            // Also honour the parent's stop directly: a ponder miss sets
            // info.stop, but the validator runs on its own SearchInfo, so
            // without this it would keep going (and block the join) after the
            // move is already over.
            validation_info.ext_stop       = &info.stop;
            validation_info.silent         = true;           // don't emit info lines from opponent's POV

            mcts::MCTS validator(validation_pos, validation_info);

            // Look for AB's "aha moment": the specific node, several plies
            // into validation_pos, where iterative deepening first found a
            // stable refutation/tactic. If found, seed MCTS's evaluation
            // there directly (instead of leaving it to NNUE judgment + visit
            // allocation that may never reach it) and bias exploration along
            // the path so MCTS is actually likely to walk down to it.
            Position tactic_trace_pos = validation_pos;
            const ab::TacticNode tactic = (orig_time_ms >= kMinTimeForDeepDetectionMs)
                ? ab::find_tactic_node(tactic_trace_pos, kAbMaxDepth, probe_budget(kTacticTraceBudgetMs))
                : ab::TacticNode{};
            if (tactic.found) {
                Position key_pos = validation_pos;
                StateInfo key_st;
                for (const Move m : tactic.path) key_pos.do_move(m, key_st);

                std::cout << "info string AB tactic-trace: found at depth " << tactic.aha_depth
                          << ", path=";
                for (const Move m : tactic.path) std::cout << m.to_uci() << " ";
                std::cout << " seed_q=" << tactic.seed_q << std::endl;

                validator.set_bias_path(tactic.path);
                validator.set_value_seed(key_pos.key(), tactic.seed_q, kTacticSeedVisits);
            }

            validator.run();
            (void) validator.get_best_move();  // populates validation_info.best_score

            // validation_info.best_score is from the opponent's perspective
            // (validation_pos's STM). Negate to get the position's eval from
            // our root perspective.
            const Score validated_cp = static_cast<Score>(-validation_info.best_score);

            // The bar AB's move must clear is what MCTS's move is *actually*
            // worth, not MCTS's own claim for it. info.best_score assumes MCTS
            // gets to follow its move's optimistic line; if AB sees the move as
            // worse (ab_view_of_mcts_move) or has a stable refutation of it
            // (mcts_refutation), that optimistic figure is an overestimate and
            // judging AB's alternative against it is unfair. Take the tightest
            // (lowest) honest estimate. This is the "tactic makes MCTS's move
            // worse" half of the picture -- the seed below is the "tactic makes
            // AB's move better" half; we don't presume which one is in play.
            Score mcts_move_bar = std::min(info.best_score, ab_view_of_mcts_move);
            if (mcts_refutation.found) {
                mcts_move_bar = std::min(mcts_move_bar,
                                         static_cast<Score>(-mcts_refutation.root_score_cp));
            }

            std::cout << "info string AB validation: ab " << ab_result.move.to_uci()
                      << " " << ab_result.score << "cp (d=" << ab_result.reached_d
                      << ")  mcts-verdict-after-move " << validated_cp << "cp"
                      << "  vs mcts-claim " << info.best_score << "cp"
                      << " (bar " << mcts_move_bar << "cp"
                      << (mcts_refutation.found ? ", mcts-move refuted)" : ")")
                      << std::endl;

            if (validated_cp >= mcts_move_bar) {
                log_ab_outcome(ab_result, info.best_move, info.best_score, "override (validated)");
                info.best_move  = ab_result.move;
                info.best_score = validated_cp;
                break;
            }
            log_ab_outcome(ab_result, info.best_move, info.best_score, "refuted");
        } else {
            log_ab_outcome(ab_result, info.best_move, info.best_score, "refuted (no validation phase)");
        }

        // Both sides refuted each other. If there's clock slack and we haven't
        // burned through the extension budget, deepen both searchers and
        // re-check instead of guessing.
        if (!try_extend("double-disagreement")) {
            log_ab_outcome(ab_result, info.best_move, info.best_score,
                           "unresolved double-disagreement (kept mcts)");
            break;
        }
        resolved = false;  // loop again to re-check with the deepened results
    }
    info.limits.time_ms = orig_time_ms;  // restore again; the extension branch mutated it

    // Collect ponder move from whatever survived reconciliation, before
    // save_to_cache() nulls the root.
    const Move ponder_move = mcts_search.get_ponder_move_after(info.best_move);

    mcts_search.save_to_cache();  // preserve subtree for tree reuse next move

    if (info.best_move == MoveNone) {
        // Should only happen if the root has no legal moves (game already
        // over). Fall back to the first legal move rather than emitting
        // `bestmove 0000`, which GUIs/cutechess treat as an illegal move.
        if (moves.size > 0) info.best_move = moves[0];
        std::cout << "bestmove "
                   << (info.best_move != MoveNone ? info.best_move.to_uci() : "0000")
                   << std::endl;
    } else if (ponder_move != MoveNone) {
        std::cout << "bestmove " << info.best_move.to_uci()
                  << " ponder " << ponder_move.to_uci() << std::endl;
    } else {
        std::cout << "bestmove " << info.best_move.to_uci() << std::endl;
    }

    return info.best_move;
}

}  // namespace eclipse
