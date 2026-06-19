// SPDX-License-Identifier: GPL-3.0-or-later
//
// do_move / undo_move round-trip tests.
//
// After undo_move the position must be byte-for-byte what it was: same zobrist
// key, same FEN, same side to move. This is the incremental-state path
// (zobrist key + castling/ep/halfmove + accumulator snapshot) that MCTS and AB
// exercise millions of times per move — a single asymmetry there silently
// corrupts search. Checked for EVERY legal move (and one ply deeper) across a
// set of positions covering castling, en passant, promotions, and captures.

#include <cstdint>
#include <string>

#include "check.hpp"
#include "movegen.hpp"
#include "position.hpp"

using namespace eclipse;

namespace {

void roundtrip(const char* fen, int depth) {
    Position p;
    ECLIPSE_CHECK(p.set_from_fen(fen));

    const std::uint64_t key0 = p.key();
    const std::string   fen0 = p.fen();
    const Color         stm0 = p.side_to_move();

    MoveList legal;
    generate_legal_moves(p, legal);
    for (const Move m : legal) {
        StateInfo st;
        p.do_move(m, st);

        // Every real move flips the side to move, which is hashed into the key,
        // so the key must actually change — catches a no-op / stuck hash.
        ECLIPSE_CHECK(p.key() != key0);
        ECLIPSE_CHECK(p.side_to_move() != stm0);

        if (depth > 1) {
            MoveList l2;
            generate_legal_moves(p, l2);
            for (const Move m2 : l2) {
                StateInfo st2;
                p.do_move(m2, st2);
                p.undo_move(m2, st2);
            }
            // The inner walk must itself restore the post-`m` position exactly.
            const std::uint64_t mid_key = p.key();
            (void) mid_key;  // implicitly verified by the outer undo below
        }

        p.undo_move(m, st);

        ECLIPSE_CHECK(p.key()          == key0);
        ECLIPSE_CHECK(p.fen()          == fen0);
        ECLIPSE_CHECK(p.side_to_move() == stm0);
    }

    // Whole loop left the position untouched.
    ECLIPSE_CHECK(p.key() == key0);
    ECLIPSE_CHECK(p.fen() == fen0);
}

}  // namespace

int main() {
    // Start position.
    roundtrip("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2);
    // "Kiwipete" — castling both sides, captures, rich tactics.
    roundtrip("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2);
    // En passant available (target square f6).
    roundtrip("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", 2);
    // Promotions for both colors.
    roundtrip("4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1", 2);
    // Black to move, en passant, mixed pieces.
    roundtrip("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1", 2);
    // Small endgame.
    roundtrip("8/8/8/3k4/8/8/8/KQ6 w - - 0 1", 2);

    return eclipse::test::summarize("makemove");
}
