// SPDX-License-Identifier: GPL-3.0-or-later
#include "check.hpp"
#include "movegen.hpp"
#include "position.hpp"

using namespace eclipse;

namespace {

// FEN round-trip: parse, regenerate, must match.
void check_fen_roundtrip(const char* fen) {
    Position p;
    ECLIPSE_CHECK(p.set_from_fen(fen));
    ECLIPSE_CHECK(p.fen() == std::string(fen));
}

// do_move + undo_move must restore the FEN and the Zobrist key exactly.
void check_do_undo(const char* fen) {
    Position p;
    p.set_from_fen(fen);

    const std::string before_fen = p.fen();
    const std::uint64_t before_key = p.key();

    MoveList moves;
    generate_legal_moves(p, moves);

    StateInfo st;
    for (const Move m : moves) {
        p.do_move(m, st);
        p.undo_move(m, st);
        ECLIPSE_CHECK(p.fen() == before_fen);
        ECLIPSE_CHECK(p.key() == before_key);
    }
}

}  // namespace

int main() {
    check_fen_roundtrip("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    check_fen_roundtrip("8/8/8/8/8/8/8/8 w - - 0 1");
    check_fen_roundtrip("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    check_fen_roundtrip("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    check_fen_roundtrip("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1");

    // Side to move serialised correctly.
    {
        Position p;
        p.set_from_fen("8/8/8/8/8/8/4k3/4K3 b - - 0 1");
        ECLIPSE_CHECK(p.side_to_move() == Black);
        ECLIPSE_CHECK(p.king_square(White) == E1);
        ECLIPSE_CHECK(p.king_square(Black) == E2);
    }

    // En-passant target round-trips.
    {
        Position p;
        p.set_from_fen("rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq d6 0 2");
        ECLIPSE_CHECK(p.ep_square() == D6);
        ECLIPSE_CHECK(p.fen()
                      == "rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq d6 0 2");
    }

    // Castling-rights mask survives a round-trip.
    {
        Position p;
        p.set_from_fen("r3k2r/8/8/8/8/8/8/R3K2R w Qk - 0 1");
        ECLIPSE_CHECK(p.castling_rights()
                      == (WhiteQueenside | BlackKingside));
    }

    // do_move/undo_move is exactly reversible on every legal first move from
    // a battery of representative positions (startpos, Kiwipete, endgame,
    // promotion-heavy, complex middlegame).
    check_do_undo("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    check_do_undo("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    check_do_undo("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    check_do_undo("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1");
    check_do_undo("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");

    // From startpos, the side to move has 20 legal moves.
    {
        Position p = Position::startpos();
        MoveList moves;
        generate_legal_moves(p, moves);
        ECLIPSE_CHECK(moves.size == 20);
    }

    // Kiwipete has 48 legal moves.
    {
        Position p;
        p.set_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        MoveList moves;
        generate_legal_moves(p, moves);
        ECLIPSE_CHECK(moves.size == 48);
    }

    return eclipse::test::summarize("position");
}
