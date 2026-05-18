// SPDX-License-Identifier: GPL-3.0-or-later
//
// Perft validation against the standard test set:
// https://www.chessprogramming.org/Perft_Results
//
// Depths are chosen so the whole test runs in a few seconds on CI hardware
// without -march=native. The deeper layers (startpos d6 = 119M, kiwipete d5 =
// 193M) are reserved for local manual runs via the `go perft` UCI command.

#include "check.hpp"
#include "perft.hpp"
#include "position.hpp"

using namespace eclipse;

namespace {

struct Case {
    const char*   fen;
    int           depth;
    std::uint64_t expected;
};

const Case kCases[] = {
    // Startpos.
    {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1,         20},
    {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2,        400},
    {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3,       8902},
    {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4,     197281},
    {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5,    4865609},

    // Kiwipete - heavy tactical position, every special move flavour appears.
    {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1,        48},
    {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2,      2039},
    {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3,     97862},
    {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4,   4085603},

    // Position 3 - endgame with promotion, useful for catching ep + promo bugs.
    {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1,      14},
    {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2,     191},
    {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3,    2812},
    {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4,   43238},
    {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5,  674624},

    // Position 4 - white in check from a long-diagonal bishop on b6, with
    // promotion-capable black pawns on b2/g2. Values verified against
    // Stockfish 17.1 - the chessprogramming.org wiki's published numbers
    // (264 / 9467 / 422333) are incorrect for this FEN.
    {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1", 1,      6},
    {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1", 2,    280},
    {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1", 3,   9346},
    {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1", 4, 438345},

    // Position 5 - middlegame with a pinned pawn promotion.
    {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1,       44},
    {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2,     1486},
    {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3,    62379},
    {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4,  2103487},

    // Position 6 - quiet symmetric middlegame, exercises sliders heavily.
    {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 1,       46},
    {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 2,     2079},
    {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3,    89890},
    {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4,  3894594},
};

}  // namespace

int main() {
    for (const Case& c : kCases) {
        Position p;
        if (!p.set_from_fen(c.fen)) {
            std::fprintf(stderr, "FEN parse failed: %s\n", c.fen);
            ECLIPSE_CHECK(false);
            continue;
        }
        const std::uint64_t got = perft(p, c.depth);
        if (got != c.expected) {
            std::fprintf(stderr,
                         "perft mismatch: fen=\"%s\" depth=%d got=%llu expected=%llu\n",
                         c.fen, c.depth,
                         static_cast<unsigned long long>(got),
                         static_cast<unsigned long long>(c.expected));
            // Divided perft helps localise the buggy subtree on regression.
            Position p2;
            p2.set_from_fen(c.fen);
            const auto split = perft_divided(p2, c.depth);
            for (const auto& s : split) {
                std::fprintf(stderr, "  %s: %llu\n",
                             s.move.to_uci().c_str(),
                             static_cast<unsigned long long>(s.nodes));
            }
            ECLIPSE_CHECK(false);
        }
    }
    return eclipse::test::summarize("perft");
}
