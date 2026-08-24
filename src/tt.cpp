// SPDX-License-Identifier: GPL-3.0-or-later
#include "tt.hpp"

#include <algorithm>
#include <cstring>
#include <new>

namespace eclipse {

namespace {

// Packed slot: 16 bytes, four to a 64-byte cache line.
//
// `key` holds (zobrist ^ data), the Hyatt lockless-hashing trick. Lazy SMP
// writes entries without any lock, so a reader can observe one 64-bit word from
// a new entry paired with the other word from the entry it replaced. The old
// code stored the raw key and only compared that, so such a torn read passed
// validation and handed the search another position's move and score — a source
// of illegal-move aborts and silent corruption that gets worse with every extra
// AB thread. XOR-pairing makes a torn pair fail the check (the two halves no
// longer reconstruct the probed key), at the cost of one XOR per access.
//
// Both words are atomics loaded/stored relaxed: that guarantees each 64-bit
// half is read and written without tearing (which plain uint64 does not
// formally promise under a data race), while the XOR handles consistency
// *between* the halves. No ordering is needed — a stale-but-consistent entry is
// always a legal search outcome.
struct TTSlot {
    std::atomic<std::uint64_t> key{0};
    std::atomic<std::uint64_t> data{0};
};

constexpr int kClusterSize = 4;

// data word layout:
//   [ 0:16)  move       (raw 16-bit Move)
//   [16:32)  score      (int16)
//   [32:40)  depth + 128 (uint8, so qsearch's negative depths survive)
//   [40:42)  bound flag
//   [42:48)  generation (0-63)
//   [48:64)  reserved (static eval will live here)
static_assert(sizeof(Move) == 2, "TT packing assumes a 16-bit Move");

std::uint64_t pack(Move m, Score score, int depth, TTFlag flag, std::uint8_t gen) {
    std::uint16_t raw_move;
    std::memcpy(&raw_move, &m, sizeof(raw_move));
    const auto d8 = static_cast<std::uint8_t>(std::clamp(depth, -127, 127) + 128);
    return static_cast<std::uint64_t>(raw_move)
         | (static_cast<std::uint64_t>(static_cast<std::uint16_t>(
                static_cast<std::int16_t>(score))) << 16)
         | (static_cast<std::uint64_t>(d8) << 32)
         | (static_cast<std::uint64_t>(flag & 0x3) << 40)
         | (static_cast<std::uint64_t>(gen & 0x3F) << 42);
}

Move unpack_move(std::uint64_t data) {
    const auto raw = static_cast<std::uint16_t>(data & 0xFFFF);
    Move m;
    std::memcpy(&m, &raw, sizeof(raw));
    return m;
}

Score        unpack_score(std::uint64_t d) { return static_cast<std::int16_t>((d >> 16) & 0xFFFF); }
int          unpack_depth(std::uint64_t d) { return static_cast<int>((d >> 32) & 0xFF) - 128; }
TTFlag       unpack_flag (std::uint64_t d) { return static_cast<TTFlag>((d >> 40) & 0x3); }
std::uint8_t unpack_gen  (std::uint64_t d) { return static_cast<std::uint8_t>((d >> 42) & 0x3F); }

}  // namespace

struct alignas(64) TranspositionTable::Cluster {
    TTSlot slot[kClusterSize];
};

TranspositionTable g_tt;

TranspositionTable::TranspositionTable(std::size_t mb_size) {
    resize(mb_size);
}

TranspositionTable::~TranspositionTable() = default;

void TranspositionTable::resize(std::size_t mb_size) {
    // Cluster-granular sizing. The old code divided the byte budget by a
    // 24-byte entry and rounded the *entry* count down to a power of two, which
    // discarded up to half the configured Hash (256 MB really allocated 201).
    // A 64-byte cluster divides any power-of-two MB value exactly, so the
    // rounding below is a no-op for every realistic Hash setting.
    std::size_t num_clusters = (std::max<std::size_t>(1, mb_size) * 1024 * 1024) / sizeof(Cluster);

    std::size_t p2 = 1;
    while (p2 * 2 <= num_clusters) p2 *= 2;

    clusters_ = std::make_unique<Cluster[]>(p2);   // value-initialised => all slots zero
    count_    = p2;
    mask_     = p2 - 1;
}

void TranspositionTable::clear() {
    for (std::size_t i = 0; i < count_; ++i)
        for (int j = 0; j < kClusterSize; ++j) {
            clusters_[i].slot[j].key.store(0, std::memory_order_relaxed);
            clusters_[i].slot[j].data.store(0, std::memory_order_relaxed);
        }
}

bool TranspositionTable::probe(std::uint64_t key, TTEntry& out) const {
    const Cluster& c = clusters_[index(key)];
    for (int i = 0; i < kClusterSize; ++i) {
        const std::uint64_t data = c.slot[i].data.load(std::memory_order_relaxed);
        if (data == 0) continue;                       // never written
        const std::uint64_t k = c.slot[i].key.load(std::memory_order_relaxed);
        if ((k ^ data) != key) continue;               // miss, or a torn read

        out.key        = key;
        out.move       = unpack_move(data);
        out.score      = unpack_score(data);
        out.depth      = static_cast<std::int16_t>(unpack_depth(data));
        out.flag       = unpack_flag(data);
        out.generation = unpack_gen(data);
        return true;
    }
    return false;
}

void TranspositionTable::store(std::uint64_t key, Move move, Score score, int depth,
                               TTFlag flag, int ply) {
    Cluster& c = clusters_[index(key)];

    TTEntry tmp;
    const Score tt_score = tmp.score_to_tt(score, ply);

    // An exact mate proof should never be discarded in favour of a deeper
    // non-mate or a weaker-bound mate entry — the faster mate distance is
    // always the most useful information regardless of search depth.
    const bool new_is_exact_mate = (flag == TT_EXACT)
                                && (tt_score >=  TTEntry::kTTMateScore
                                 || tt_score <= -TTEntry::kTTMateScore);

    int victim         = 0;
    int victim_quality = 0;

    for (int i = 0; i < kClusterSize; ++i) {
        const std::uint64_t data = c.slot[i].data.load(std::memory_order_relaxed);

        if (data == 0) {                                // empty slot: take it
            victim = i;
            break;
        }

        const std::uint64_t k = c.slot[i].key.load(std::memory_order_relaxed);
        if ((k ^ data) == key) {
            // Same position. Refresh unless the existing entry is meaningfully
            // deeper: a shallower re-search still carries a newer generation
            // (which protects it from eviction) and usually a better move.
            const int stored_depth = unpack_depth(data);
            if (!new_is_exact_mate && flag != TT_EXACT && depth + 2 < stored_depth)
                return;
            // Preserve a real move when the new entry has none — an all-node
            // fail-low stores MoveNone, and throwing away the previously known
            // best move here costs move ordering on the next visit.
            if (move == MoveNone) move = unpack_move(data);
            victim = i;
            break;
        }

        // Replacement score: prefer to evict shallow and stale. Age is the
        // distance back from the current generation, wrapped over the 6-bit
        // cycle. The old table had a single slot with no aging at all, so one
        // deep entry from three moves ago blocked that index permanently.
        const int age = (kGenCycle + generation_ - unpack_gen(data)) % kGenCycle;
        const int quality = unpack_depth(data) - 4 * age;
        if (i == 0 || quality < victim_quality) {
            victim = i;
            victim_quality = quality;
        }
    }

    const std::uint64_t data = pack(move, tt_score, depth, flag, generation_);
    c.slot[victim].key.store(key ^ data, std::memory_order_relaxed);
    c.slot[victim].data.store(data, std::memory_order_relaxed);
}

}  // namespace eclipse
