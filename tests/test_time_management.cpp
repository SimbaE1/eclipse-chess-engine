// SPDX-License-Identifier: GPL-3.0-or-later
//
// Time-management regression tests.
//
// These guard the won-game time-flag class of bugs (the long saga in
// search.cpp / uci.cpp): time_up() must trip on the hard deadline, the soft
// budget, ext_stop, and the ponder ceiling; and ms_until_hard_deadline() must
// report 0 once the deadline or stop is hit so every post-MCTS phase (the
// validation MCTS, AB cross-check/tactic probes, extensions) bails instead of
// overrunning the clock. All deterministic — no real sleeping required.

#include <atomic>
#include <chrono>
#include <cstdint>

#include "check.hpp"
#include "search.hpp"

using namespace eclipse;
using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

namespace {
std::int64_t now_epoch_ms() {
    return std::chrono::duration_cast<milliseconds>(
        Clock::now().time_since_epoch()).count();
}
constexpr std::int64_t kBig = 1'000'000;  // "effectively unbounded" threshold
}  // namespace

int main() {
    // ── non-ponder soft budget ────────────────────────────────────────────
    {
        SearchInfo info;
        info.limits.time_ms = 50;
        info.start_time = Clock::now();
        ECLIPSE_CHECK(!info.time_up());                      // just started
        info.start_time = Clock::now() - milliseconds(100);  // 100ms elapsed
        ECLIPSE_CHECK(info.time_up());                       // 100 >= 50
    }

    // ── time_ms <= 0 is "unbounded" for non-ponder (go depth / go infinite) ─
    {
        SearchInfo info;
        info.limits.time_ms = 0;
        info.start_time = Clock::now() - milliseconds(10'000);
        ECLIPSE_CHECK(!info.time_up());
        info.limits.infinite = true;
        ECLIPSE_CHECK(!info.time_up());
    }

    // ── absolute hard deadline overrides even a huge soft budget ───────────
    // This is the core guarantee: whatever phase is running, once the deadline
    // passes the move is over.
    {
        SearchInfo info;
        info.limits.time_ms = kBig;
        info.start_time = Clock::now();
        info.hard_deadline = Clock::now() + milliseconds(50);
        ECLIPSE_CHECK(!info.time_up());
        info.hard_deadline = Clock::now() - milliseconds(1);
        ECLIPSE_CHECK(info.time_up());
    }

    // ── nodes / depth limits trip time_up ──────────────────────────────────
    {
        SearchInfo info;
        info.limits.nodes = 100;
        info.nodes_searched.store(100, std::memory_order_relaxed);
        ECLIPSE_CHECK(info.time_up());
        info.nodes_searched.store(99, std::memory_order_relaxed);
        ECLIPSE_CHECK(!info.time_up());
    }
    {
        SearchInfo info;
        info.limits.depth = 5;  // depth uses nodes_searched as the proxy counter
        info.nodes_searched.store(5, std::memory_order_relaxed);
        ECLIPSE_CHECK(info.time_up());
    }

    // ── ext_stop (parent abort) overrides everything ──────────────────────
    // A ponder miss / stop must abort an internal sub-search even with a fresh
    // clock and a huge budget. (This is the validation-MCTS backstop.)
    {
        SearchInfo info;
        info.limits.time_ms = kBig;
        info.start_time = Clock::now();
        std::atomic<bool> parent_stop{false};
        info.ext_stop = &parent_stop;
        ECLIPSE_CHECK(!info.time_up());
        parent_stop.store(true, std::memory_order_relaxed);
        ECLIPSE_CHECK(info.time_up());
    }

    // ── ponder: unbounded while still pondering (no ponderhit yet) ─────────
    // Even after a long ponder the search must not self-stop on its budget —
    // the ceiling is measured from the (not-yet-arrived) hit.
    {
        SearchInfo info;
        info.limits.ponder = true;
        info.limits.time_ms = 50;
        info.limits.hard_limit_ms = 100;
        info.start_time = Clock::now() - milliseconds(10'000);
        info.ponderhit_at_ms.store(-1, std::memory_order_relaxed);
        ECLIPSE_CHECK(!info.time_up());
        ECLIPSE_CHECK(info.ms_until_hard_deadline() > kBig);  // unbounded sentinel
    }

    // ── ponder: trips at since_hit >= time_ms after the hit ───────────────
    {
        SearchInfo info;
        info.limits.ponder = true;
        info.limits.time_ms = 40;
        info.limits.hard_limit_ms = 1000;
        info.ponderhit_at_ms.store(now_epoch_ms() - 100, std::memory_order_relaxed);
        ECLIPSE_CHECK(info.time_up());  // 100 >= 40
    }

    // ── ponder: hard_limit also bounds it even if the soft budget is huge ──
    {
        SearchInfo info;
        info.limits.ponder = true;
        info.limits.time_ms = kBig;
        info.limits.hard_limit_ms = 30;
        info.ponderhit_at_ms.store(now_epoch_ms() - 50, std::memory_order_relaxed);
        ECLIPSE_CHECK(info.time_up());  // 50 >= 30
    }

    // ── ponder: not yet up immediately after the hit ──────────────────────
    {
        SearchInfo info;
        info.limits.ponder = true;
        info.limits.time_ms = 10'000;
        info.limits.hard_limit_ms = 10'000;
        info.ponderhit_at_ms.store(now_epoch_ms(), std::memory_order_relaxed);
        ECLIPSE_CHECK(!info.time_up());
    }

    // ── ms_until_hard_deadline: no deadline -> large; stop -> 0 ────────────
    {
        SearchInfo info;
        ECLIPSE_CHECK(info.ms_until_hard_deadline() > kBig);
        info.stop.store(true, std::memory_order_relaxed);
        ECLIPSE_CHECK(info.ms_until_hard_deadline() == 0);  // every phase bails
    }

    // ── ms_until_hard_deadline: future deadline positive, past -> 0 ───────
    {
        SearchInfo info;
        info.hard_deadline = Clock::now() + milliseconds(500);
        const auto left = info.ms_until_hard_deadline();
        ECLIPSE_CHECK(left > 0 && left <= 500);
        info.hard_deadline = Clock::now() - milliseconds(1);
        ECLIPSE_CHECK(info.ms_until_hard_deadline() == 0);
    }

    // ── ms_until_hard_deadline: ponder = hard_limit - since_hit, clamped ──
    {
        SearchInfo info;
        info.limits.ponder = true;
        info.limits.hard_limit_ms = 200;
        info.ponderhit_at_ms.store(now_epoch_ms() - 50, std::memory_order_relaxed);
        const auto left = info.ms_until_hard_deadline();
        ECLIPSE_CHECK(left > 0 && left <= 200);
        // well past the ceiling -> 0
        info.ponderhit_at_ms.store(now_epoch_ms() - 10'000, std::memory_order_relaxed);
        ECLIPSE_CHECK(info.ms_until_hard_deadline() == 0);
        // stop wins over the ponder math
        info.stop.store(true, std::memory_order_relaxed);
        info.ponderhit_at_ms.store(now_epoch_ms(), std::memory_order_relaxed);
        ECLIPSE_CHECK(info.ms_until_hard_deadline() == 0);
    }

    return eclipse::test::summarize("time_management");
}
