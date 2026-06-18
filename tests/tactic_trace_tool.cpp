// SPDX-License-Identifier: GPL-3.0-or-later
// Manual debug tool (not a ctest): runs ab::find_tactic_node on a FEN passed
// as argv[1] and prints what it found. Used to verify the tactic-node
// detection in isolation, independent of whether the AB/MCTS reconciliation
// gate happens to trigger in any particular live search.
//
// IMPORTANT: pass the same SyzygyPath the live engine uses (argv[4] or the
// SYZYGY_PATH env var). negamax probes tablebases at interior nodes, which
// completely changes the score landscape in endgames -- without it this tool
// does NOT reproduce in-engine behavior and you'll chase phantom mismatches.
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "ab.hpp"
#include "position.hpp"
#include "zobrist.hpp"
#include "attacks.hpp"
#include "tt.hpp"
#include "syzygy.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: tactic_trace_tool <fen> [max_depth] [budget_ms] [syzygy_path]\n"
                     "       (syzygy path also read from $SYZYGY_PATH)\n";
        return 1;
    }
    eclipse::zobrist::init();
    eclipse::init_attacks();
    eclipse::g_tt.resize(256);  // match the engine's configured Hash size

    const char* tb_path = argc > 4 ? argv[4] : std::getenv("SYZYGY_PATH");
    if (tb_path && tb_path[0]) {
        eclipse::syzygy::init(tb_path);
        std::cout << "syzygy: " << (eclipse::syzygy::is_enabled()
                      ? "enabled (max " + std::to_string(eclipse::syzygy::max_pieces()) + " pieces)"
                      : "init failed / not enabled")
                  << " path=" << tb_path << "\n";
    } else {
        std::cout << "syzygy: DISABLED (no path) -- will NOT match the live engine in endgames\n";
    }

    eclipse::Position pos;
    pos.set_from_fen(argv[1]);
    const int max_depth = argc > 2 ? std::atoi(argv[2]) : 32;
    const std::int64_t budget_ms = argc > 3 ? std::atoll(argv[3]) : 8000;

    const auto result = eclipse::ab::find_tactic_node(pos, max_depth, budget_ms);
    if (!result.found) {
        std::cout << "no stable tactic found\n";
        return 0;
    }
    std::cout << "found at depth " << result.aha_depth << "\n";
    std::cout << "path: ";
    for (const auto m : result.path) std::cout << m.to_uci() << " ";
    std::cout << "\nseed_q: " << result.seed_q << "\n";
    std::cout << "root_score_cp: " << result.root_score_cp << "\n";
    return 0;
}
