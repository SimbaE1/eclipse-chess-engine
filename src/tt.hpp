// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "eval.hpp"
#include "move.hpp"

namespace eclipse {

enum TTFlag : std::uint8_t {
    TT_NONE,
    TT_EXACT,
    TT_UPPERBOUND,
    TT_LOWERBOUND
};

// Unpacked view of a table entry. This is what probe() hands back and what the
// search reads; the table itself stores a packed, race-safe 16-byte form (see
// TTSlot in tt.cpp). Keeping the two apart lets the storage layout change
// without touching any call site in ab.cpp.
struct TTEntry {
    std::uint64_t key;
    Move          move;
    Score         score;
    std::int16_t  depth;
    TTFlag        flag;
    // Search generation that wrote this entry, 0-63. Lets find_tactic_node
    // ignore entries left by prior searches for depth-cutoffs (so its
    // depth-by-depth swing detection sees genuine depth-limited scores)
    // without wiping the whole table.
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
};

class TranspositionTable {
public:
    // Generation counter is 6 bits (packed alongside the bound flag), so it
    // wraps at 64. new_generation() returns the already-masked value, which is
    // what SearchCtx::cutoff_gen compares against.
    static constexpr std::uint8_t kGenCycle = 64;

    TranspositionTable(std::size_t mb_size = 16);
    ~TranspositionTable();

    TranspositionTable(const TranspositionTable&)            = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;

    void resize(std::size_t mb_size);
    void clear();

    bool probe(std::uint64_t key, TTEntry& out) const;
    void store(std::uint64_t key, Move move, Score score, int depth, TTFlag flag, int ply);

    // Bump the current write-generation. find_tactic_node calls this so its own
    // entries are distinguishable from prior searches' without a clear(), and
    // the bucket replacement policy uses the distance from it as an age.
    std::uint8_t new_generation() {
        generation_ = static_cast<std::uint8_t>((generation_ + 1) % kGenCycle);
        return generation_;
    }
    std::uint8_t generation() const { return generation_; }

private:
    struct Cluster;

    std::size_t index(std::uint64_t key) const noexcept { return key & mask_; }

    std::unique_ptr<Cluster[]> clusters_;
    std::size_t                count_ = 0;   // number of clusters (power of two)
    std::size_t                mask_  = 0;
    std::uint8_t               generation_ = 0;
};

extern TranspositionTable g_tt;

}  // namespace eclipse
