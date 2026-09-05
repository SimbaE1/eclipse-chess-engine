// SPDX-License-Identifier: GPL-3.0-or-later
//
// Cross-checks see() against a brute-force reference: the exchange on the
// target square evaluated by plain recursive minimax, each side free to stand
// pat, with captured pieces actually removed from the attacker set.
//
// There was no SEE test before, which is how two independent bugs survived in
// a routine that decides capture pruning and move ordering:
//
//   1. attackers_to() builds its result from the pos.pieces() bitboards, which
//      do not change as the swap loop removes pieces -- only the local `occ`
//      does. Without masking by occ, a piece that had already captured was
//      handed back as an attacker and re-selected. Knights, kings and pawns
//      never escape this, since their attack tables ignore occ entirely.
//   2. The textbook swap-algorithm cutoff was applied to a loop whose gain[]
//      is one ply out of alignment with the textbook's, so it discarded an
//      entry the backpropagation needed.
//
// Together they made 33% of all captures score wrong.

#include <algorithm>
#include <cstdio>
#include <random>

#include "check.hpp"
#include "attacks.hpp"
#include "movegen.hpp"
#include "position.hpp"
#include "see.hpp"

using namespace eclipse;

namespace {

// Least-valuable attacker of `stm` among pieces still present in `occ`.
Square least_valuable(const Position& pos, Color stm, Bitboard occ, Square to) {
    const Bitboard att = pos.attackers_to(to, occ) & occ;
    for (PieceType pt : {Pawn, Knight, Bishop, Rook, Queen, King}) {
        const Bitboard s = att & pos.pieces(stm, pt);
        if (s) return lsb(s);
    }
    return SquareNone;
}

// What `stm` can gain by continuing the exchange on `to`, given the piece
// sitting there is worth `on_square`. Declining the capture is worth 0.
int reference_gain(const Position& pos, Square to, Color stm, Bitboard occ,
                   int on_square, int depth) {
    if (depth > 32) return 0;
    const Square from = least_valuable(pos, stm, occ, to);
    if (from == SquareNone) return 0;
    const int moving = kPieceValue[type_of(pos.piece_on(from))];
    return std::max(0, on_square - reference_gain(pos, to, ~stm,
                                                  occ ^ (1ULL << from),
                                                  moving, depth + 1));
}

int reference_see(const Position& pos, Move m) {
    const int captured = kPieceValue[type_of(pos.piece_on(m.to()))];
    const int moving   = kPieceValue[type_of(pos.piece_on(m.from()))];
    return captured - reference_gain(pos, m.to(), ~pos.side_to_move(),
                                     pos.occupied() ^ (1ULL << m.from()),
                                     moving, 0);
}

void check_position(const char* fen, Square from, Square to, int expect) {
    Position pos;
    ECLIPSE_CHECK(pos.set_from_fen(fen));
    const int got = see(pos, Move(from, to));
    if (got != expect)
        std::fprintf(stderr, "  see=%d expect=%d on %s\n", got, expect, fen);
    ECLIPSE_CHECK(got == expect);
}

// Hand-built cases covering each bug directly. Knights attacking d5 sit on
// b4, b6, c3, c7, e3, e7, f4 and f6.
void test_known_positions() {
    // Bug 1. One white knight takes a pawn defended by two black knights and
    // has no follow-up: 100 - 320. The unmasked version re-selected the very
    // knight it had just lost and scored this a free pawn (+100).
    check_position("4k3/8/1n3n2/3p4/8/2N5/8/4K3 w - - 0 1", C3, D5, -220);
    check_position("4k3/8/1n3n2/3p4/8/2N1N3/8/4K3 w - - 0 1", C3, D5, -220);
    check_position("4k3/8/1n6/3p4/8/2N5/8/4K3 w - - 0 1", C3, D5, -220);

    // Bug 2. Rook battery on the d-file against a pawn defended once:
    // RxP (+100), NxR (-500), RxN (+320) = -80. The cutoff stopped before the
    // second rook could recapture and returned -400.
    check_position("4k3/8/1n6/3p4/8/8/3R4/3RK3 w - - 0 1", D2, D5, -80);

    // Controls that were already correct and must stay so.
    check_position("4k3/8/8/3p4/8/2N5/8/4K3 w - - 0 1", C3, D5, 100);
    check_position("4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1", C3, D5, 900);
}

// Every plain capture reachable along random games, against the reference.
void test_against_reference() {
    std::mt19937 rng(20260905u);
    long tested = 0, wrong = 0;

    for (int game = 0; game < 60; ++game) {
        Position pos;
        ECLIPSE_CHECK(pos.set_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
        for (int ply = 0; ply < 140; ++ply) {
            MoveList ml;
            generate_legal_moves(pos, ml);
            if (ml.size == 0) break;

            for (const Move m : ml) {
                // The reference models a plain capture only; en passant and
                // promotion carry extra terms see() handles separately.
                if (m.type() != Move::Normal) continue;
                if (pos.piece_on(m.to()) == NoPiece) continue;
                ++tested;
                const int got = see(pos, m), want = reference_see(pos, m);
                if (got != want) {
                    ++wrong;
                    if (wrong <= 5)
                        std::fprintf(stderr, "  see=%d ref=%d  %s  %d->%d\n",
                                     got, want, pos.fen().c_str(),
                                     int(m.from()), int(m.to()));
                }
            }

            StateInfo st;
            pos.do_move(ml[int(rng() % unsigned(ml.size))], st, false, false);
        }
    }

    std::printf("  see vs reference: %ld captures, %ld wrong\n", tested, wrong);
    ECLIPSE_CHECK(tested > 20000);
    ECLIPSE_CHECK(wrong == 0);
}

}  // namespace

int main() {
    test_known_positions();
    test_against_reference();
    return eclipse::test::summarize("test_see");
}
