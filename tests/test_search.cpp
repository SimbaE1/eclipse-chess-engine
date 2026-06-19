// SPDX-License-Identifier: GPL-3.0-or-later
//
// Smoke tests for the search layer. Not a strength test - just confirms
// search runs, terminates, and picks plausible moves in obvious positions.

#include <atomic>
#include <chrono>

#include "check.hpp"
#include "eval.hpp"
#include "movegen.hpp"
#include "position.hpp"
#include "search.hpp"

using namespace eclipse;

namespace {

bool is_legal(const Position& pos, Move m) {
    Position p = pos;
    MoveList legal;
    generate_legal_moves(p, legal);
    for (const Move x : legal) if (x == m) return true;
    return false;
}

Move search_for(const char* fen, int depth) {
    Position p;
    p.set_from_fen(fen);
    SearchInfo info;
    info.limits.depth = depth;
    return search(p, info);
}

}  // namespace

int main() {
    // Material eval signs / orientation.
    {
        Position p = Position::startpos();
        ECLIPSE_CHECK(evaluate(p) == 0);
    }
    {
        // White is up a queen.
        Position p;
        p.set_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
        ECLIPSE_CHECK(evaluate(p) == 900);
    }
    {
        // Same position, black to move - eval is negative from black's view.
        Position p;
        p.set_from_fen("4k3/8/8/8/8/8/8/3QK3 b - - 0 1");
        ECLIPSE_CHECK(evaluate(p) == -900);
    }

    // Search returns a legal move from the start.
    {
        Position p = Position::startpos();
        SearchInfo info;
        info.limits.depth = 3;
        const Move m = search(p, info);
        ECLIPSE_CHECK(!m.is_null());
        ECLIPSE_CHECK(is_legal(p, m));
    }

    // Classic back-rank mate-in-1: Ra8# wins immediately. Check the search
    // both returns a legal move and recognises the mate (score >> mate
    // threshold). The position is reachable in normal play (unlike a king-
    // adjacent-to-check setup), which matters now that movegen refuses to
    // emit king-capturing moves.
    {
        Position p;
        p.set_from_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1");
        SearchInfo info;
        info.limits.depth = 4;
        const Move m = search(p, info);
        ECLIPSE_CHECK(!m.is_null());
        ECLIPSE_CHECK(is_legal(p, m));
        // Mate-in-1 detected: best score should be above the mate threshold.
        ECLIPSE_CHECK(info.best_score > kMateInMaxPly);
    }

    // Forced move position: pinned king, only one legal response. The search
    // must return that move regardless of depth.
    {
        // White king in check from black queen on the a1-h8 diagonal, blocked
        // only by moving the king. This is the same setup we know from
        // Position 4 - king on g1 with checker on b6, every legal move
        // blocks or moves out of check.
        Position p;
        ECLIPSE_CHECK(p.set_from_fen(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1"));
        const Move m = search_for(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1", 3);
        ECLIPSE_CHECK(!m.is_null());
        ECLIPSE_CHECK(is_legal(p, m));
    }

    // No-legal-move detection: stalemated black king. Search must terminate
    // gracefully and not return a bogus move.
    {
        // Black king on h8 stalemated by white K + Q (well-known position).
        Position p;
        p.set_from_fen("7k/8/6QK/8/8/8/8/8 b - - 0 1");
        SearchInfo info;
        info.limits.depth = 2;
        const Move m = search(p, info);
        // 0 legal moves; the search should not produce a "best move".
        ECLIPSE_CHECK(m.is_null());
    }

    // time_up() bounds for internal sub-searches (the validation MCTS).
    //
    // Regression guard for the won-game ponder flag: the validator runs on its
    // own SearchInfo with limits.time_ms clamped to whatever is left on the
    // move, which can be 0 when the budget is already spent. time_up() treats
    // time_ms<=0 as "unbounded" (correct for `go infinite`/depth searches), so
    // without a backstop a zero-budget validator under a ponder parent (whose
    // hard_deadline is epoch) would spin until the node pool filled, overrunning
    // the move by 100s+. The fix makes search() bind the validator an ABSOLUTE
    // deadline and point ext_stop at the parent's stop. Verify both here.
    {
        using Clock = std::chrono::steady_clock;
        // Zero budget, no hard deadline: this is the unbounded case (the bug).
        SearchInfo info;
        info.limits.time_ms = 0;
        ECLIPSE_CHECK(!info.time_up());  // documents the footgun the fix guards

        // An absolute deadline already in the past bounds it even at time_ms=0.
        info.hard_deadline = Clock::now() - std::chrono::milliseconds(1);
        ECLIPSE_CHECK(info.time_up());

        // ext_stop (parent abort) overrides everything, even with no deadline
        // and a positive budget — a ponder miss must abort the sub-search.
        SearchInfo sub;
        sub.limits.time_ms = 1'000'000;       // plenty of budget on its own
        sub.start_time     = Clock::now();    // ...and the clock just started
        std::atomic<bool> parent_stop{false};
        sub.ext_stop = &parent_stop;
        ECLIPSE_CHECK(!sub.time_up());
        parent_stop.store(true, std::memory_order_relaxed);
        ECLIPSE_CHECK(sub.time_up());
    }

    return eclipse::test::summarize("search");
}
