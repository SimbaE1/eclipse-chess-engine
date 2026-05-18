// SPDX-License-Identifier: GPL-3.0-or-later
#include "attacks.hpp"

#include <cstring>

namespace eclipse {

Bitboard pawn_attacks_table[ColorNB][SquareNB];
Bitboard knight_attacks_table[SquareNB];
Bitboard king_attacks_table[SquareNB];

namespace {

// Per-square slider attack tables. Sized to the worst case for each piece
// (12 relevant bits for a rook at a corner, 9 for a bishop near the centre)
// so that indexing is identical for every square. Wastes ~1 MB total
// versus a packed layout - acceptable for now; the layout simplifies the
// magic-search code and the unused entries never get probed.
constexpr int kRookTableSize   = 1 << 12;  // 4096
constexpr int kBishopTableSize = 1 <<  9;  // 512

Bitboard rook_table  [SquareNB][kRookTableSize];
Bitboard bishop_table[SquareNB][kBishopTableSize];

struct Magic {
    Bitboard        mask;
    Bitboard        magic;
    unsigned        shift;
    const Bitboard* table;
};

Magic rook_magics  [SquareNB];
Magic bishop_magics[SquareNB];

bool initialised = false;

constexpr int kRookDirs  [4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
constexpr int kBishopDirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

// Ray walk from `sq` in each direction. Stops at the first occupied square
// (which is included - that's the "first blocker" semantics sliders need).
Bitboard ray_attacks(Square sq, Bitboard occ, const int (*dirs)[2]) noexcept {
    Bitboard out = 0;
    const int sf = file_of(sq);
    const int sr = rank_of(sq);
    for (int d = 0; d < 4; ++d) {
        int df = dirs[d][0];
        int dr = dirs[d][1];
        int f  = sf + df;
        int r  = sr + dr;
        while (f >= 0 && f <= 7 && r >= 0 && r <= 7) {
            Square s = make_square(File(f), Rank(r));
            out |= square_bb(s);
            if (occ & square_bb(s)) break;
            f += df;
            r += dr;
        }
    }
    return out;
}

// The magic-bitboard "relevant occupancy" mask: ray squares strictly inside
// the board. Edge squares are excluded because they can never be a blocker
// for further attacks - the ray already terminates there.
Bitboard sliding_mask(Square sq, const int (*dirs)[2]) noexcept {
    Bitboard out = 0;
    const int sf = file_of(sq);
    const int sr = rank_of(sq);
    for (int d = 0; d < 4; ++d) {
        int df = dirs[d][0];
        int dr = dirs[d][1];
        int f  = sf + df;
        int r  = sr + dr;
        while (f + df >= 0 && f + df <= 7 && r + dr >= 0 && r + dr <= 7) {
            out |= square_bb(make_square(File(f), Rank(r)));
            f += df;
            r += dr;
        }
    }
    return out;
}

// xorshift64* with a multiplicative scramble. Same generator the Zobrist init
// uses; rolling our own keeps the engine deterministic across libstdc++ /
// libc++ random implementations.
struct Prng {
    std::uint64_t state;
    std::uint64_t next() noexcept {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 2685821657736338717ULL;
    }
    // ANDing three draws yields a "sparse" 64-bit value (~8 set bits on avg).
    // Sparse magics tend to satisfy the perfect-hash property faster.
    std::uint64_t sparse() noexcept { return next() & next() & next(); }
};

void init_slider(Square sq,
                 const int (*dirs)[2],
                 Magic& m,
                 Bitboard* table,
                 int table_size,
                 Prng& rng) {
    m.mask  = sliding_mask(sq, dirs);
    const int bits = popcount(m.mask);
    m.shift = static_cast<unsigned>(64 - bits);
    m.table = table;

    const int n_subsets = 1 << bits;

    Bitboard subsets[kRookTableSize];
    Bitboard correct[kRookTableSize];

    // Carry-Rippler iterates every subset of `mask` exactly once. Start at 0;
    // each step adds the next subset in lexicographic order; we return to 0
    // after 2^bits steps, which is our termination signal.
    int i = 0;
    Bitboard occ = 0;
    do {
        subsets[i] = occ;
        correct[i] = ray_attacks(sq, occ, dirs);
        ++i;
        occ = (occ - m.mask) & m.mask;
    } while (occ != 0);

    // The popcount heuristic asks for >= 6 set bits in the top byte of
    // (mask * magic). It works well for high-bit-count masks, but for the
    // 5-6-bit masks of corner bishops, sparse magics rarely meet the
    // threshold within a reasonable budget. Adapt it down for thin masks.
    const int top_byte_threshold = bits >= 8 ? 6 : (bits >= 6 ? 4 : 2);

    for (int tries = 0; tries < 10'000'000; ++tries) {
        const Bitboard magic = rng.sparse();
        if (popcount((m.mask * magic) >> 56) < top_byte_threshold) continue;

        std::memset(table, 0, static_cast<std::size_t>(table_size) * sizeof(Bitboard));

        bool fail = false;
        for (int j = 0; j < n_subsets; ++j) {
            const unsigned idx = static_cast<unsigned>(
                (subsets[j] * magic) >> m.shift);
            if (table[idx] == 0) {
                table[idx] = correct[j];
            } else if (table[idx] != correct[j]) {
                fail = true;
                break;
            }
        }
        if (!fail) {
            m.magic = magic;
            return;
        }
    }
    // Magic search has never failed in practice within this budget. If it
    // ever does in CI, the engine should fail loudly rather than serve up
    // silently wrong attacks - hence the bald std::abort.
    std::abort();
}

void init_leapers() {
    for (int s = 0; s < SquareNB; ++s) {
        const Bitboard bb = square_bb(Square(s));

        Bitboard knight = 0;
        knight |= (bb & ~file_bb(FileH))                              << 17;  // NNE
        knight |= (bb & ~file_bb(FileA))                              << 15;  // NNW
        knight |= (bb & ~(file_bb(FileH) | file_bb(FileG)))           << 10;  // ENE
        knight |= (bb & ~(file_bb(FileA) | file_bb(FileB)))           <<  6;  // WNW
        knight |= (bb & ~file_bb(FileH))                              >> 15;  // SSE
        knight |= (bb & ~file_bb(FileA))                              >> 17;  // SSW
        knight |= (bb & ~(file_bb(FileH) | file_bb(FileG)))           >>  6;  // ESE
        knight |= (bb & ~(file_bb(FileA) | file_bb(FileB)))           >> 10;  // WSW
        knight_attacks_table[s] = knight;

        Bitboard k = bb;
        k |= shift_east (k) | shift_west (k);
        k |= shift_north(k) | shift_south(k);
        k &= ~bb;
        king_attacks_table[s] = k;

        pawn_attacks_table[White][s] = shift_ne(bb) | shift_nw(bb);
        pawn_attacks_table[Black][s] = shift_se(bb) | shift_sw(bb);
    }
}

}  // namespace

Bitboard rook_attacks(Square s, Bitboard occ) noexcept {
    const Magic& m = rook_magics[s];
    return m.table[((occ & m.mask) * m.magic) >> m.shift];
}

Bitboard bishop_attacks(Square s, Bitboard occ) noexcept {
    const Magic& m = bishop_magics[s];
    return m.table[((occ & m.mask) * m.magic) >> m.shift];
}

Bitboard piece_attacks(PieceType pt, Square s, Bitboard occ) noexcept {
    switch (pt) {
        case Knight: return knight_attacks(s);
        case Bishop: return bishop_attacks(s, occ);
        case Rook:   return rook_attacks(s, occ);
        case Queen:  return queen_attacks(s, occ);
        case King:   return king_attacks(s);
        default:     return 0;
    }
}

void init_attacks() {
    if (initialised) return;

    init_leapers();

    Prng rng{0xDEADBEEFCAFEF00DULL};
    for (int s = 0; s < SquareNB; ++s) {
        init_slider(Square(s), kRookDirs,
                    rook_magics[s],   rook_table[s],   kRookTableSize,   rng);
        init_slider(Square(s), kBishopDirs,
                    bishop_magics[s], bishop_table[s], kBishopTableSize, rng);
    }

    initialised = true;
}

}  // namespace eclipse
