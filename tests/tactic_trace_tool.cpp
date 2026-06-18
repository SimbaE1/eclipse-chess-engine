// SPDX-License-Identifier: GPL-3.0-or-later
// Manual debug tool (not a ctest): runs ab::find_tactic_node on a FEN passed
// as argv[1] and prints what it found. Used to verify the tactic-node
// detection in isolation, independent of whether the AB/MCTS reconciliation
// gate happens to trigger in any particular live search.
#include <cstdint>
#include <iostream>

#include "ab.hpp"
#include "position.hpp"
#include "zobrist.hpp"
#include "attacks.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: tactic_trace_tool <fen> [max_depth] [budget_ms]\n";
        return 1;
    }
    eclipse::zobrist::init();
    eclipse::init_attacks();

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
    return 0;
}
