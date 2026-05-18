// SPDX-License-Identifier: GPL-3.0-or-later
#include "attacks.hpp"
#include "bitboard.hpp"
#include "check.hpp"

using namespace eclipse;

namespace {

Bitboard sqs(std::initializer_list<Square> list) {
    Bitboard b = 0;
    for (Square s : list) b |= square_bb(s);
    return b;
}

}  // namespace

int main() {
    init_attacks();

    // Knight on E5 reaches the standard 8 squares.
    ECLIPSE_CHECK(knight_attacks(E5)
                  == sqs({C4, D3, F3, G4, G6, F7, D7, C6}));

    // Corner knight on A1 has just two moves.
    ECLIPSE_CHECK(knight_attacks(A1) == sqs({B3, C2}));

    // Edge knight on H4: F3, F5, G2, G6.
    ECLIPSE_CHECK(knight_attacks(H4) == sqs({F3, F5, G2, G6}));

    // King on E5 attacks the 8 neighbours.
    ECLIPSE_CHECK(king_attacks(E5)
                  == sqs({D4, E4, F4, D5, F5, D6, E6, F6}));
    // Corner king on A1 attacks A2, B1, B2.
    ECLIPSE_CHECK(king_attacks(A1) == sqs({A2, B1, B2}));

    // Pawn attacks point forward-diagonal for the colour.
    ECLIPSE_CHECK(pawn_attacks(White, E2) == sqs({D3, F3}));
    ECLIPSE_CHECK(pawn_attacks(Black, E7) == sqs({D6, F6}));
    // Edge pawns only attack one diagonal.
    ECLIPSE_CHECK(pawn_attacks(White, A2) == sqs({B3}));
    ECLIPSE_CHECK(pawn_attacks(White, H2) == sqs({G3}));
    ECLIPSE_CHECK(pawn_attacks(Black, A7) == sqs({B6}));

    // Rook on D4, empty board: full rank-4 + full file-D minus D4 itself.
    {
        const Bitboard expected = (rank_bb(Rank4) | file_bb(FileD)) & ~square_bb(D4);
        ECLIPSE_CHECK(rook_attacks(D4, 0) == expected);
    }

    // Rook on D4 with friend on D6 stops the north ray at D6 (still "attacked"
    // -- legality of capturing own piece is filtered by movegen, not by attacks).
    // The south, east, and west rays are unobstructed.
    {
        const Bitboard occ = square_bb(D6) | square_bb(D4);
        const Bitboard expected = sqs({D1, D2, D3, D5, D6,
                                       A4, B4, C4, E4, F4, G4, H4});
        ECLIPSE_CHECK(rook_attacks(D4, occ) == expected);
    }

    // Bishop on D4, empty board: both diagonals through D4 minus the square.
    {
        const Bitboard expected = sqs({A1, B2, C3, E5, F6, G7, H8,
                                       A7, B6, C5, E3, F2, G1});
        ECLIPSE_CHECK(bishop_attacks(D4, 0) == expected);
    }

    // Bishop on D4 with blocker on F6 trims the NE ray at F6.
    {
        const Bitboard occ = square_bb(F6) | square_bb(D4);
        const Bitboard expected = sqs({A1, B2, C3, E5, F6,
                                       A7, B6, C5, E3, F2, G1});
        ECLIPSE_CHECK(bishop_attacks(D4, occ) == expected);
    }

    // Queen = rook | bishop.
    {
        const Bitboard occ = square_bb(D6) | square_bb(F6) | square_bb(D4);
        ECLIPSE_CHECK(queen_attacks(D4, occ)
                      == (rook_attacks(D4, occ) | bishop_attacks(D4, occ)));
    }

    // piece_attacks() dispatch matches direct calls.
    {
        const Bitboard occ = sqs({A2, H7, E5, D4, C3});
        ECLIPSE_CHECK(piece_attacks(Knight, E5, occ) == knight_attacks(E5));
        ECLIPSE_CHECK(piece_attacks(King,   E5, occ) == king_attacks(E5));
        ECLIPSE_CHECK(piece_attacks(Bishop, E5, occ) == bishop_attacks(E5, occ));
        ECLIPSE_CHECK(piece_attacks(Rook,   E5, occ) == rook_attacks  (E5, occ));
        ECLIPSE_CHECK(piece_attacks(Queen,  E5, occ) == queen_attacks (E5, occ));
    }

    return eclipse::test::summarize("attacks");
}
