// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "bitboard.hpp"
#include "types.hpp"

namespace eclipse {

// Leaper attack tables. Filled by init_attacks() and read by inlined accessors
// below so callers see them as compile-time-shaped lookups.
extern Bitboard pawn_attacks_table[ColorNB][SquareNB];
extern Bitboard knight_attacks_table[SquareNB];
extern Bitboard king_attacks_table[SquareNB];

// Idempotent. Must be called once before any of the accessors below are used.
// The Position constructor and the UCI loop both call it on first use.
void init_attacks();

inline Bitboard pawn_attacks(Color c, Square s) noexcept {
    return pawn_attacks_table[c][s];
}
inline Bitboard knight_attacks(Square s) noexcept {
    return knight_attacks_table[s];
}
inline Bitboard king_attacks(Square s) noexcept {
    return king_attacks_table[s];
}

// Slider attacks are looked up through magic bitboards. `occ` is the full
// occupancy bitboard; the magic-mask narrowing happens inside.
Bitboard rook_attacks  (Square s, Bitboard occ) noexcept;
Bitboard bishop_attacks(Square s, Bitboard occ) noexcept;

inline Bitboard queen_attacks(Square s, Bitboard occ) noexcept {
    return rook_attacks(s, occ) | bishop_attacks(s, occ);
}

// Convenience: attacks of `pt` (a non-pawn piece) standing on `s` with the
// given occupancy. Pawn captures use pawn_attacks(color, sq) since they
// depend on color.
Bitboard piece_attacks(PieceType pt, Square s, Bitboard occ) noexcept;

}  // namespace eclipse
