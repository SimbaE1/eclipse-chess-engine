// SPDX-License-Identifier: GPL-3.0-or-later
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#include "attacks.hpp"
#include "eval.hpp"
#include "nnue.hpp"
#include "position.hpp"
#include "zobrist.hpp"

using namespace eclipse;

struct EvalCase {
    std::string name;
    std::string fen;
};

int main() {
    zobrist::init();
    init_attacks();

    const char* path = std::getenv("ECLIPSE_NNUE_PATH");
    if (path == nullptr) {
        std::printf("SKIP  eval_comparison (ECLIPSE_NNUE_PATH not set)\n");
        return 0;
    }

    std::vector<EvalCase> cases = {
        {"Startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
        {"K+Q vs K", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"},
        {"K+R vs K", "4k3/8/8/8/8/8/8/3RK3 w - - 0 1"},
        {"K+P vs K", "4k3/8/8/8/8/4P3/8/4K3 w - - 0 1"},
        {"Sicilian Defense", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"},
        {"Ruy Lopez", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 3"},
        {"Late Endgame", "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1"}
    };

    std::printf("%-20s | %-12s | %-12s | %-12s\n", "Position", "Material", "NNUE", "Difference");
    std::printf("---------------------|--------------|--------------|--------------\n");

    for (const auto& c : cases) {
        Position p;
        p.set_from_fen(c.fen);

        // Load NNUE once
        if (!nnue::is_loaded()) {
            if (!nnue::load(path)) {
                return 1;
            }
        }

        Score mat    = material_evaluate(p);
        Score n_eval = nnue::evaluate(p);

        std::printf("%-20s | %10d cp | %10d cp | %10d cp\n", 
                   c.name.c_str(), mat, n_eval, n_eval - mat);
    }

    return 0;
}
