// SPDX-License-Identifier: GPL-3.0-or-later
//
// Ad-hoc tool (not wired into CMake): reads FEN lines from stdin, prints the
// NNUE static eval (cp, relative to side-to-move) for each. Used to drive a
// "greedy 1-ply NNUE" move picker from Python without per-position process
// spawn overhead.
//
//   build/tests/nnue_eval_server <nnue_path>

#include <cstdio>
#include <iostream>
#include <string>

#include "attacks.hpp"
#include "nnue.hpp"
#include "position.hpp"
#include "zobrist.hpp"

using namespace eclipse;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <nnue_path>\n", argv[0]);
        return 1;
    }
    zobrist::init();
    init_attacks();

    if (!nnue::load(argv[1])) {
        std::fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        Position p;
        if (!p.set_from_fen(line)) {
            std::cout << "ERR" << std::endl;
            continue;
        }
        std::cout << nnue::evaluate(p) << std::endl;
    }
    return 0;
}
