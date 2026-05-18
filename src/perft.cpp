// SPDX-License-Identifier: GPL-3.0-or-later
#include "perft.hpp"

#include "movegen.hpp"

namespace eclipse {

std::uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1;

    MoveList moves;
    generate_legal_moves(pos, moves);

    // Bulk-counting at depth 1: skip the per-leaf do/undo cycle. This is the
    // standard perft micro-optimisation and matches what most engines do.
    if (depth == 1) return static_cast<std::uint64_t>(moves.size);

    std::uint64_t total = 0;
    StateInfo st;
    for (const Move m : moves) {
        pos.do_move(m, st);
        total += perft(pos, depth - 1);
        pos.undo_move(m, st);
    }
    return total;
}

std::vector<PerftSplit> perft_divided(Position& pos, int depth) {
    std::vector<PerftSplit> out;
    if (depth <= 0) return out;

    MoveList moves;
    generate_legal_moves(pos, moves);
    out.reserve(static_cast<std::size_t>(moves.size));

    StateInfo st;
    for (const Move m : moves) {
        pos.do_move(m, st);
        const std::uint64_t n = depth == 1 ? 1ULL : perft(pos, depth - 1);
        pos.undo_move(m, st);
        out.push_back({m, n});
    }
    return out;
}

}  // namespace eclipse
