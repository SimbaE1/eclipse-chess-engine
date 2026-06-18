#pragma once

#include <cstdint>

namespace eclipse {

enum Color : std::uint8_t {
    White   = 0,
    Black   = 1,
    ColorNB = 2,
};

constexpr Color operator~(Color c) noexcept {
    return Color(c ^ 1);
}

struct SearchLimits {
    int           depth    = 0;
    std::uint64_t nodes    = 0;
    std::int64_t  time_ms  = 0;
    bool          infinite = false;
    bool          ponder   = false;

    // Extra time still safely spendable on THIS move beyond `time_ms`, per
    // the remain/3 safety cap already enforced in uci.cpp's cmd_go (the
    // allocated time_ms is usually well under that cap). 0 when there's no
    // clock to borrow from (movetime/depth/nodes-limited searches). Used by
    // search.cpp's AB/MCTS reconciliation to extend search on a genuine
    // disagreement instead of always falling back immediately.
    std::int64_t  extra_budget_ms = 0;

    // Absolute hard ceiling on TOTAL wall-clock time for this move, in ms from
    // search start. Unlike time_ms (the soft target for the main MCTS phase)
    // and extra_budget_ms (slack the reconciliation may borrow), this bounds
    // the SUM of every phase the search runs: main MCTS+AB, the validation
    // MCTS, the AB cross-check/tactic-trace probes, and any extension rounds.
    // search() turns it into a single absolute deadline that time_up() and
    // every sub-phase budget respect, so the engine physically cannot overrun
    // the clock and flag. Set by uci.cpp with a latency margin already
    // subtracted. 0 = no hard ceiling (movetime/depth/nodes/infinite/ponder).
    std::int64_t  hard_limit_ms = 0;
};

// Piece-type indexing leaves slot 0 free as NoPieceType so a Piece can encode
// the empty square as 0 and pack into 4 bits (color << 3 | piecetype).
enum PieceType : std::uint8_t {
    NoPieceType = 0,
    Pawn        = 1,
    Knight      = 2,
    Bishop      = 3,
    Rook        = 4,
    Queen       = 5,
    King        = 6,
    PieceTypeNB = 8,
};

enum Piece : std::uint8_t {
    NoPiece  = 0,
    WPawn    = 1, WKnight, WBishop, WRook, WQueen, WKing,
    BPawn    = 9, BKnight, BBishop, BRook, BQueen, BKing,
    PieceNB  = 16,
};

constexpr Piece make_piece(Color c, PieceType pt) noexcept {
    return Piece((c << 3) | pt);
}

constexpr Color color_of(Piece p) noexcept {
    return Color(p >> 3);
}

constexpr PieceType type_of(Piece p) noexcept {
    return PieceType(p & 7);
}

enum File : std::uint8_t {
    FileA, FileB, FileC, FileD, FileE, FileF, FileG, FileH, FileNB,
};

enum Rank : std::uint8_t {
    Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, RankNB,
};

// Square layout: A1 = 0, B1 = 1, ..., H1 = 7, A2 = 8, ..., H8 = 63.
enum Square : std::uint8_t {
    A1 = 0, B1, C1, D1, E1, F1, G1, H1,
    A2,     B2, C2, D2, E2, F2, G2, H2,
    A3,     B3, C3, D3, E3, F3, G3, H3,
    A4,     B4, C4, D4, E4, F4, G4, H4,
    A5,     B5, C5, D5, E5, F5, G5, H5,
    A6,     B6, C6, D6, E6, F6, G6, H6,
    A7,     B7, C7, D7, E7, F7, G7, H7,
    A8,     B8, C8, D8, E8, F8, G8, H8,
    SquareNB   = 64,
    SquareNone = 64,
};

constexpr Square make_square(File f, Rank r) noexcept {
    return Square((r << 3) | f);
}

constexpr File file_of(Square s) noexcept { return File(s & 7); }
constexpr Rank rank_of(Square s) noexcept { return Rank(s >> 3); }

constexpr Square flip_rank(Square s) noexcept { return Square(s ^ 56); }
constexpr Square flip_file(Square s) noexcept { return Square(s ^ 7); }

constexpr bool is_valid(Square s) noexcept { return s < SquareNB; }

// Offsets used both as integer deltas for ray walks and as bit-shift amounts
// for whole-bitboard shifts. Kept as a signed type because southward deltas
// are negative.
enum Direction : int {
    North     =  8,
    South     = -8,
    East      =  1,
    West      = -1,
    NorthEast = North + East,
    NorthWest = North + West,
    SouthEast = South + East,
    SouthWest = South + West,
};

// Castling rights packed as a 4-bit mask. Indexable as an array key.
enum CastlingRights : std::uint8_t {
    NoCastling       = 0,
    WhiteKingside    = 1,
    WhiteQueenside   = 2,
    BlackKingside    = 4,
    BlackQueenside   = 8,
    WhiteCastling    = WhiteKingside | WhiteQueenside,
    BlackCastling    = BlackKingside | BlackQueenside,
    AnyCastling      = WhiteCastling | BlackCastling,
    CastlingRightsNB = 16,
};

constexpr CastlingRights operator|(CastlingRights a, CastlingRights b) noexcept {
    return CastlingRights(std::uint8_t(std::uint8_t(a) | std::uint8_t(b)));
}
constexpr CastlingRights operator&(CastlingRights a, CastlingRights b) noexcept {
    return CastlingRights(std::uint8_t(std::uint8_t(a) & std::uint8_t(b)));
}
constexpr CastlingRights operator~(CastlingRights a) noexcept {
    return CastlingRights(std::uint8_t(std::uint8_t(AnyCastling) & ~std::uint8_t(a)));
}
constexpr CastlingRights& operator|=(CastlingRights& a, CastlingRights b) noexcept {
    a = a | b;
    return a;
}
constexpr CastlingRights& operator&=(CastlingRights& a, CastlingRights b) noexcept {
    a = a & b;
    return a;
}

// Centipawn values for piece types, used for SEE and move ordering.
constexpr int kPieceValue[PieceTypeNB] = {
    0,     // NoPieceType
    100,   // Pawn
    320,   // Knight
    330,   // Bishop
    500,   // Rook
    900,   // Queen
    20000, // King
    0,
};

using Score = int;
constexpr Score kInfinite       = 30000;
constexpr Score kMateScore      = 29000;
constexpr Score kMateInMaxPly   = kMateScore - 1000;
constexpr Score kDraw           = 0;

}  // namespace eclipse
