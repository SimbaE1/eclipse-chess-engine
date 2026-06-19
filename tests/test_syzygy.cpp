// SPDX-License-Identifier: GPL-3.0-or-later
//
// Syzygy tablebase tests. Guarded by ECLIPSE_SYZYGY_PATH (and a successful
// load), so a machine without tablebases is SKIPPED, never failed — same
// pattern as test_nnue.
//
// With the 3-4-5 WDL+DTZ set present these guard the won-endgame-draw bug
// (game hYkAdJFB, a drawn K+Q vs K):
//   1. the root probe must use DTZ, so it reports a real distance and a move
//      that converges to mate (WDL alone ranks every winning move equally);
//   2. the probe must fire on PONDERED moves too (it was skipped pre-fix);
//   3. K+Q vs K must actually reach checkmate, not shuffle into a draw.
//
// Run with:  ECLIPSE_SYZYGY_PATH=/path/to/syzygy ctest -R test_syzygy

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "check.hpp"
#include "movegen.hpp"
#include "position.hpp"
#include "search.hpp"
#include "syzygy.hpp"

using namespace eclipse;

namespace {
bool is_legal(Position pos, Move m) {
    MoveList legal;
    generate_legal_moves(pos, legal);
    for (const Move x : legal) if (x == m) return true;
    return false;
}
constexpr const char* kKQvK = "8/8/8/3k4/8/8/8/KQ6 w - - 0 1";  // White K+Q vs lone K
}  // namespace

int main() {
    const char* tb = std::getenv("ECLIPSE_SYZYGY_PATH");
    if (!tb || !*tb) {
        std::printf("SKIP  syzygy (set ECLIPSE_SYZYGY_PATH to run)\n");
        return 0;
    }
    syzygy::init(tb);
    if (!syzygy::is_enabled()) {
        std::printf("SKIP  syzygy (no tablebases found at %s)\n", tb);
        return 0;
    }

    // ── DTZ root probe: K+Q vs K is a win with a real, positive distance ──
    // The bug returned dtz=0 (WDL probe) and a non-progressing move.
    {
        Position p;
        ECLIPSE_CHECK(p.set_from_fen(kKQvK));
        const syzygy::RootBest rb = syzygy::probe_root_best(p);
        ECLIPSE_CHECK(rb.wdl == syzygy::kTbWin);
        ECLIPSE_CHECK(rb.dtz > 0);  // genuine distance-to-zero, not the old 0
        const Move m(static_cast<Square>(rb.from), static_cast<Square>(rb.to));
        ECLIPSE_CHECK(is_legal(p, m));
    }

    // ── interior WDL probe: KQvK is a win; bare kings are a draw ──────────
    {
        Position win;
        win.set_from_fen(kKQvK);
        ECLIPSE_CHECK(syzygy::probe_wdl(win) == syzygy::kTbWin);
        Position draw;
        draw.set_from_fen("8/8/8/3k4/8/8/8/K7 w - - 0 1");
        ECLIPSE_CHECK(syzygy::probe_wdl(draw) == syzygy::kTbDraw);
    }

    // ── search() short-circuits to the tablebase move (non-ponder) ───────
    {
        Position p;
        p.set_from_fen(kKQvK);
        SearchInfo info;
        info.limits.time_ms = 1000;
        const Move m = search(p, info);
        ECLIPSE_CHECK(!m.is_null());
        ECLIPSE_CHECK(is_legal(p, m));
    }

    // ── regression: the root probe must NOT be skipped on a ponder search ─
    // Pre-fix, `!info.limits.ponder` in the guard meant every pondered move
    // bypassed the tablebase. Fire a ponderhit from another thread; search must
    // still return the (instant) TB move. The store happens well after search
    // has reset ponderhit_at_ms to -1 at its start, so there is no race.
    {
        Position p;
        p.set_from_fen(kKQvK);
        SearchInfo info;
        info.limits.ponder        = true;
        info.limits.time_ms       = 1000;
        info.limits.hard_limit_ms = 1000;
        std::thread hit([&info] {
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            info.ponderhit_at_ms.store(ms, std::memory_order_release);
        });
        const Move m = search(p, info);
        hit.join();
        ECLIPSE_CHECK(!m.is_null());
        ECLIPSE_CHECK(is_legal(p, m));
    }

    // ── full conversion: K+Q vs K reaches checkmate (no shuffle draw) ────
    // Engine plays BOTH sides; with DTZ it mates well within the 50-move
    // window. Pre-fix it drew by threefold repetition.
    {
        Position p;
        p.set_from_fen(kKQvK);
        bool mated = false;
        int ply = 0;
        for (; ply < 50; ++ply) {
            MoveList legal;
            generate_legal_moves(p, legal);
            if (legal.size == 0) {
                mated = p.in_check();  // checkmate vs stalemate
                break;
            }
            SearchInfo info;
            info.limits.time_ms = 500;
            const Move m = search(p, info);
            ECLIPSE_CHECK(!m.is_null() && is_legal(p, m));
            if (m.is_null()) break;
            StateInfo st;
            p.do_move(m, st);
        }
        ECLIPSE_CHECK(mated);       // reached checkmate, not a draw/stalemate
        ECLIPSE_CHECK(ply <= 40);   // KQvK mates in well under 20 full moves
    }

    return eclipse::test::summarize("syzygy");
}
