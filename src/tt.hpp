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

    void save(std::uint64_t k, Move m, Score s, int d, TTFlag f, int ply) {
        key   = k;
        move  = m;
        score = score_to_tt(s, ply);
        depth = static_cast<std::int16_t>(d);
        flag  = f;
    }
};

class TranspositionTable {
public:
    TranspositionTable(std::size_t mb_size = 16);

    void resize(std::size_t mb_size);
    void clear();

    bool probe(std::uint64_t key, TTEntry& out) const;
    void store(std::uint64_t key, Move move, Score score, int depth, TTFlag flag, int ply);

private:
    std::vector<TTEntry> table_;
    std::size_t          mask_;
};

extern TranspositionTable g_tt;

}  // namespace eclipse
