// SPDX-License-Identifier: GPL-3.0-or-later
#include "tt.hpp"

#include <algorithm>

namespace eclipse {

TranspositionTable g_tt;

TranspositionTable::TranspositionTable(std::size_t mb_size) {
    resize(mb_size);
}

void TranspositionTable::resize(std::size_t mb_size) {
    std::size_t num_entries = (mb_size * 1024 * 1024) / sizeof(TTEntry);
    
    // Power of two for fast masking
    std::size_t p2 = 1;
    while (p2 * 2 <= num_entries) p2 *= 2;
    
    table_.assign(p2, TTEntry{0, MoveNone, 0, 0, TT_NONE});
    mask_ = p2 - 1;
}

void TranspositionTable::clear() {
    std::fill(table_.begin(), table_.end(), TTEntry{0, MoveNone, 0, 0, TT_NONE});
}

bool TranspositionTable::probe(std::uint64_t key, TTEntry& out) const {
    const TTEntry& e = table_[key & mask_];
    if (e.key == key) {
        out = e;
        return true;
    }
    return false;
}

void TranspositionTable::store(std::uint64_t key, Move move, Score score, int depth, TTFlag flag, int ply) {
    TTEntry& e = table_[key & mask_];
    
    // Replacement strategy: depth-preferred.
    // If the new entry is deeper or from a different position, overwrite.
    if (e.key != key || depth >= e.depth) {
        e.save(key, move, score, depth, flag, ply);
    }
}

}  // namespace eclipse
