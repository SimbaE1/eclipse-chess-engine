#include "bitboard.hpp"
#include "check.hpp"

using namespace eclipse;

int main() {
    // Single-square bitboards
    ECLIPSE_CHECK(square_bb(A1) == 1ULL);
    ECLIPSE_CHECK(square_bb(H8) == (1ULL << 63));
    ECLIPSE_CHECK(popcount(square_bb(D4)) == 1);

    // File and rank masks
    ECLIPSE_CHECK(popcount(file_bb(FileA)) == 8);
    ECLIPSE_CHECK(popcount(rank_bb(Rank1)) == 8);
    ECLIPSE_CHECK((file_bb(FileA) & file_bb(FileH)) == 0);
    ECLIPSE_CHECK((file_bb(FileA) & rank_bb(Rank1)) == square_bb(A1));
    ECLIPSE_CHECK((file_bb(FileH) & rank_bb(Rank8)) == square_bb(H8));
    ECLIPSE_CHECK(file_bb(FileD) == (file_bb(FileA) << 3));
    ECLIPSE_CHECK(rank_bb(Rank5) == (rank_bb(Rank1) << 32));

    // lsb / msb / pop_lsb
    Bitboard c = square_bb(C3) | square_bb(F6);
    ECLIPSE_CHECK(lsb(c) == C3);
    ECLIPSE_CHECK(msb(c) == F6);
    ECLIPSE_CHECK(popcount(c) == 2);
    Square first = pop_lsb(c);
    ECLIPSE_CHECK(first == C3);
    ECLIPSE_CHECK(popcount(c) == 1);
    ECLIPSE_CHECK(lsb(c) == F6);
    Square second = pop_lsb(c);
    ECLIPSE_CHECK(second == F6);
    ECLIPSE_CHECK(c == 0);

    // Orthogonal shifts
    ECLIPSE_CHECK(shift_north(square_bb(E4)) == square_bb(E5));
    ECLIPSE_CHECK(shift_south(square_bb(E4)) == square_bb(E3));
    ECLIPSE_CHECK(shift_east (square_bb(E4)) == square_bb(F4));
    ECLIPSE_CHECK(shift_west (square_bb(E4)) == square_bb(D4));

    // Diagonal shifts
    ECLIPSE_CHECK(shift_ne(square_bb(E4)) == square_bb(F5));
    ECLIPSE_CHECK(shift_nw(square_bb(E4)) == square_bb(D5));
    ECLIPSE_CHECK(shift_se(square_bb(E4)) == square_bb(F3));
    ECLIPSE_CHECK(shift_sw(square_bb(E4)) == square_bb(D3));

    // Edge masking: shifts off the board produce zero, not wrap
    ECLIPSE_CHECK(shift_east(square_bb(H4)) == 0);
    ECLIPSE_CHECK(shift_west(square_bb(A4)) == 0);
    ECLIPSE_CHECK(shift_ne(square_bb(H4))   == 0);
    ECLIPSE_CHECK(shift_nw(square_bb(A4))   == 0);
    ECLIPSE_CHECK(shift_se(square_bb(H1))   == 0);
    ECLIPSE_CHECK(shift_sw(square_bb(A1))   == 0);
    ECLIPSE_CHECK(shift_north(rank_bb(Rank8)) == 0);
    ECLIPSE_CHECK(shift_south(rank_bb(Rank1)) == 0);

    // Whole-rank shift preserves popcount when not crossing an edge
    ECLIPSE_CHECK(popcount(shift_north(rank_bb(Rank4))) == 8);
    ECLIPSE_CHECK(shift_north(rank_bb(Rank4)) == rank_bb(Rank5));

    // Template dispatch matches the named free functions
    ECLIPSE_CHECK(shift<North>    (square_bb(E4)) == shift_north(square_bb(E4)));
    ECLIPSE_CHECK(shift<South>    (square_bb(E4)) == shift_south(square_bb(E4)));
    ECLIPSE_CHECK(shift<East>     (square_bb(E4)) == shift_east (square_bb(E4)));
    ECLIPSE_CHECK(shift<West>     (square_bb(E4)) == shift_west (square_bb(E4)));
    ECLIPSE_CHECK(shift<NorthEast>(square_bb(E4)) == shift_ne   (square_bb(E4)));
    ECLIPSE_CHECK(shift<NorthWest>(square_bb(E4)) == shift_nw   (square_bb(E4)));
    ECLIPSE_CHECK(shift<SouthEast>(square_bb(E4)) == shift_se   (square_bb(E4)));
    ECLIPSE_CHECK(shift<SouthWest>(square_bb(E4)) == shift_sw   (square_bb(E4)));

    // Iterate all 64 squares via pop_lsb
    Bitboard all   = kFullBB;
    int      seen  = 0;
    bool     order = true;
    Square   prev  = A1;
    while (all) {
        Square s = pop_lsb(all);
        if (seen > 0 && !(s > prev)) order = false;
        prev = s;
        ++seen;
    }
    ECLIPSE_CHECK(seen == 64);
    ECLIPSE_CHECK(order);

    return eclipse::test::summarize("bitboard");
}
