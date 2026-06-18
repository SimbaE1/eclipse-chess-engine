// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <vector>

#include "eval.hpp"
#include "move.hpp"

namespace eclipse {

enum TTFlag : std::uint8_t {
    TT_NONE,
    TT_EXACT,
    TT_UPPERBOUND,
    TT_LOWERBOUND
};

struct TTEntry {
    std::uint64_t key;
    Move          move;
    Score         score;
    std::int16_t  depth;
    TTFlag        flag;
    // Search generation that wrote this entry. Lets find_tactic_node ignore
    // entries left by prior searches for depth-cutoffs (so its depth-by-depth
    // swing detection sees genuine depth-limited scores) without wiping the
    // whole table. Defaulted so existing aggregate inits stay valid.
    std::uint8_t  generation = 0;

    static constexpr Score kTTMateScore = kMateScore - 256;

    Score score_to_tt(Score s, int ply) {
        if (s >= kTTMateScore) return s + ply;
        if (s <= -kTTMateScore) return s - ply;
        return s;
    }

    Score score_from_tt(Score s, int ply) {
        if (s >= kTTMateScore) return s - ply;
        if (s <= -kTTMateScore) return s + ply;
        return s;
    }

    void save(std::uint64_t k, Move m, Score s, int d, TTFlag f, int ply,
              std::uint8_t gen) {
        key        = k;
        move       = m;
        score      = score_to_tt(s, ply);
        depth      = static_cast<std::int16_t>(d);
        flag       = f;
        generation = gen;
    }
};

class TranspositionTable {
public:
    TranspositionTable(std::size_t mb_size = 16);

    void resize(std::size_t mb_size);
    void clear();

    bool probe(std::uint64_t key, TTEntry& out) const;
    void store(std::uint64_t key, Move move, Score score, int depth, TTFlag flag, int ply);

    // Bump the current write-generation. find_tactic_node calls this so its
    // own entries are distinguishable from prior searches' without a clear().
    // Returns the new generation.
    std::uint8_t new_generation() { return ++generation_; }
    std::uint8_t generation() const { return generation_; }

private:
    std::vector<TTEntry> table_;
    std::size_t          mask_;
    std::uint8_t         generation_ = 0;
};

extern TranspositionTable g_tt;

}  // namespace eclipse
