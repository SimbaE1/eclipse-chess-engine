// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <vector>

#include "move.hpp"
#include "position.hpp"

namespace eclipse {

// Count legal leaf nodes at the given depth from `pos`. Mutates `pos` during
// the recursion but restores it before returning.
std::uint64_t perft(Position& pos, int depth);

// Per-root-move breakdown - the format printed by other engines as
//     "e2e4: 20\n  d2d4: 20\n..." followed by the total.
// Useful for diffing against another engine when perft totals disagree.
struct PerftSplit {
    Move          move;
    std::uint64_t nodes;
};

std::vector<PerftSplit> perft_divided(Position& pos, int depth);

}  // namespace eclipse
