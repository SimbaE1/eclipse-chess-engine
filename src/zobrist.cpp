// SPDX-License-Identifier: GPL-3.0-or-later
#include "zobrist.hpp"

namespace eclipse::zobrist {

std::uint64_t piece_square[PieceNB][SquareNB];
std::uint64_t castling[CastlingRightsNB];
std::uint64_t en_passant_file[FileNB];
std::uint64_t side_to_move;

namespace {

// Standard xorshift64* with a multiplicative scramble. Deterministic across
// runs - the same Position FEN must produce the same key for transposition
// tables and opening-book lookup to be meaningful.
struct Xorshift64 {
    std::uint64_t state;
    std::uint64_t next() noexcept {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 2685821657736338717ULL;
    }
};

bool initialised = false;

}  // namespace

void init() {
    if (initialised) return;
    Xorshift64 rng{0x9E3779B97F4A7C15ULL};

    for (int p = 0; p < PieceNB; ++p)
        for (int s = 0; s < SquareNB; ++s)
            piece_square[p][s] = rng.next();

    for (int c = 0; c < CastlingRightsNB; ++c)
        castling[c] = rng.next();

    for (int f = 0; f < FileNB; ++f)
        en_passant_file[f] = rng.next();

    side_to_move = rng.next();

    // NoPiece is conventionally not XORed into a position key. Zeroing the
    // row prevents an accidental XOR from corrupting the hash.
    for (int s = 0; s < SquareNB; ++s)
        piece_square[NoPiece][s] = 0;

    initialised = true;
}

}  // namespace eclipse::zobrist
