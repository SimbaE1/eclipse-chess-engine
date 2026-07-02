// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cassert>
#include <cstdint>
#include <string>

#include "types.hpp"

namespace eclipse {

// 16-bit move encoding:
//   bits  0- 5: from-square
//   bits  6-11: to-square
//   bits 12-13: promotion piece (0=Knight, 1=Bishop, 2=Rook, 3=Queen)
//   bits 14-15: move type tag
//
// Mirrors the encoding used by most modern engines so it can be slotted into a
// transposition-table entry / packed into a `score+move` u32 later on without
// reshuffling.
class Move {
public:
    enum Type : std::uint16_t {
        Normal    = 0u << 14,
        Promotion = 1u << 14,
        EnPassant = 2u << 14,
        Castling  = 3u << 14,
    };

    constexpr Move() noexcept : data_(0) {}

    constexpr Move(Square from, Square to) noexcept
        : data_(static_cast<std::uint16_t>(
              static_cast<unsigned>(from) | (static_cast<unsigned>(to) << 6))) {}

    constexpr Move(Square from, Square to, Type t) noexcept
        : data_(static_cast<std::uint16_t>(
              static_cast<unsigned>(from)
              | (static_cast<unsigned>(to) << 6)
              | static_cast<unsigned>(t))) {}

    static constexpr Move make_promotion(Square from, Square to, PieceType pt) noexcept {
        const unsigned promo_bits = static_cast<unsigned>(pt - Knight) << 12;
        return Move(static_cast<std::uint16_t>(
            static_cast<unsigned>(from)
            | (static_cast<unsigned>(to) << 6)
            | promo_bits
            | static_cast<unsigned>(Promotion)));
    }

    static constexpr Move make_en_passant(Square from, Square to) noexcept {
        return Move(from, to, EnPassant);
    }

    static constexpr Move make_castling(Square king_from, Square king_to) noexcept {
        return Move(king_from, king_to, Castling);
    }

    constexpr Square from() const noexcept {
        return Square(data_ & 0x3Fu);
    }
    constexpr Square to() const noexcept {
        return Square((data_ >> 6) & 0x3Fu);
    }
    constexpr Type type() const noexcept {
        return Type(data_ & (0x3u << 14));
    }
    constexpr PieceType promotion_piece() const noexcept {
        assert(type() == Promotion && "promotion_piece() called on a non-promotion move");
        return PieceType(((data_ >> 12) & 0x3u) + Knight);
    }

    constexpr bool is_null() const noexcept { return data_ == 0; }
    constexpr std::uint16_t raw() const noexcept { return data_; }

    constexpr bool operator<(const Move& other) const noexcept {
        return data_ < other.data_;
    }

    std::string to_uci() const;

    friend constexpr bool operator==(Move a, Move b) noexcept = default;

private:
    constexpr explicit Move(std::uint16_t d) noexcept : data_(d) {}

    std::uint16_t data_;
};

inline constexpr Move MoveNone{};

// Compact list with a fixed cap. 218 is the verified upper bound on legal moves
// in any reachable chess position; rounding up to 256 keeps the struct cleanly
// aligned and gives headroom for pseudo-legal generation that may transiently
// hold a few extras.
struct MoveList {
    Move  moves[256];
    int   size = 0;

    constexpr void push(Move m) noexcept { moves[size++] = m; }
    constexpr void clear() noexcept { size = 0; }

    constexpr const Move* begin() const noexcept { return moves; }
    constexpr const Move* end()   const noexcept { return moves + size; }
    constexpr Move*       begin()       noexcept { return moves; }
    constexpr Move*       end()         noexcept { return moves + size; }

    constexpr const Move& operator[](int i) const noexcept { return moves[i]; }
    constexpr Move&       operator[](int i)       noexcept { return moves[i]; }
};

std::string square_to_string(Square s);
Square      square_from_string(const char* s);

// When true, castling moves are output as king-to-rook (Chess960 UCI format).
// When false (default), castling moves are output as king-to-destination (e1g1/e1c1).
void set_chess960_mode(bool v) noexcept;
bool get_chess960_mode() noexcept;

}  // namespace eclipse
