#include "check.hpp"
#include "types.hpp"

using namespace eclipse;

int main() {
    ECLIPSE_CHECK(~White == Black);
    ECLIPSE_CHECK(~Black == White);

    ECLIPSE_CHECK(make_piece(White, Pawn)  == WPawn);
    ECLIPSE_CHECK(make_piece(White, King)  == WKing);
    ECLIPSE_CHECK(make_piece(Black, Queen) == BQueen);

    ECLIPSE_CHECK(color_of(WPawn) == White);
    ECLIPSE_CHECK(color_of(BKing) == Black);
    ECLIPSE_CHECK(type_of(WKnight) == Knight);
    ECLIPSE_CHECK(type_of(BRook)   == Rook);

    ECLIPSE_CHECK(make_square(FileA, Rank1) == A1);
    ECLIPSE_CHECK(make_square(FileH, Rank8) == H8);
    ECLIPSE_CHECK(make_square(FileE, Rank4) == E4);

    ECLIPSE_CHECK(file_of(E4) == FileE);
    ECLIPSE_CHECK(rank_of(E4) == Rank4);
    ECLIPSE_CHECK(file_of(A1) == FileA);
    ECLIPSE_CHECK(rank_of(H8) == Rank8);

    ECLIPSE_CHECK(flip_rank(A1) == A8);
    ECLIPSE_CHECK(flip_rank(E2) == E7);
    ECLIPSE_CHECK(flip_file(A1) == H1);
    ECLIPSE_CHECK(flip_file(B4) == G4);

    ECLIPSE_CHECK(is_valid(A1));
    ECLIPSE_CHECK(is_valid(H8));
    ECLIPSE_CHECK(!is_valid(SquareNone));

    return eclipse::test::summarize("types");
}
