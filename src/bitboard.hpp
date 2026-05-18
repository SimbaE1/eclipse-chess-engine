#pragma once

#include <bit>
#include <cstdint>
#include <string>

#include "types.hpp"

namespace eclipse {

using Bitboard = std::uint64_t;

constexpr Bitboard kEmptyBB = 0;
constexpr Bitboard kFullBB  = ~Bitboard{0};

constexpr Bitboard square_bb(Square s) noexcept {
    return Bitboard{1} << s;
}

constexpr Bitboard file_bb(File f) noexcept {
    return 0x0101010101010101ULL << f;
}

constexpr Bitboard rank_bb(Rank r) noexcept {
    return 0xFFULL << (r * 8);
}

constexpr int popcount(Bitboard b) noexcept {
    return std::popcount(b);
}

// Precondition: b != 0. std::countr_zero(0) is defined (returns 64) but the
// resulting Square would be SquareNone, which crashes any caller that uses it
// as a board index.
constexpr Square lsb(Bitboard b) noexcept {
    return Square(std::countr_zero(b));
}

constexpr Square msb(Bitboard b) noexcept {
    return Square(63 - std::countl_zero(b));
}

// Returns the least-significant set bit and clears it. The standard idiom for
// iterating the set bits of a bitboard.
constexpr Square pop_lsb(Bitboard& b) noexcept {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

// East/west variants mask the wrapping file first so off-board bits become
// zero instead of reappearing on the opposite side of the board.
constexpr Bitboard shift_north(Bitboard b) noexcept { return b << 8; }
constexpr Bitboard shift_south(Bitboard b) noexcept { return b >> 8; }
constexpr Bitboard shift_east (Bitboard b) noexcept { return (b & ~file_bb(FileH)) << 1; }
constexpr Bitboard shift_west (Bitboard b) noexcept { return (b & ~file_bb(FileA)) >> 1; }
constexpr Bitboard shift_ne   (Bitboard b) noexcept { return (b & ~file_bb(FileH)) << 9; }
constexpr Bitboard shift_nw   (Bitboard b) noexcept { return (b & ~file_bb(FileA)) << 7; }
constexpr Bitboard shift_se   (Bitboard b) noexcept { return (b & ~file_bb(FileH)) >> 7; }
constexpr Bitboard shift_sw   (Bitboard b) noexcept { return (b & ~file_bb(FileA)) >> 9; }

template <Direction D>
constexpr Bitboard shift(Bitboard b) noexcept {
    if constexpr      (D == North)     return shift_north(b);
    else if constexpr (D == South)     return shift_south(b);
    else if constexpr (D == East)      return shift_east(b);
    else if constexpr (D == West)      return shift_west(b);
    else if constexpr (D == NorthEast) return shift_ne(b);
    else if constexpr (D == NorthWest) return shift_nw(b);
    else if constexpr (D == SouthEast) return shift_se(b);
    else if constexpr (D == SouthWest) return shift_sw(b);
    else static_assert(D == North, "shift<>: unsupported direction");
}

// ASCII rendering, used in tests and ad-hoc debugging.
std::string bb_to_string(Bitboard b);

}  // namespace eclipse
