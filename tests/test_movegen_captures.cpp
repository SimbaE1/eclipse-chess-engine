// SPDX-License-Identifier: GPL-3.0-or-later
//
// Cross-checks generate_legal_captures() against the reference definition:
// generate_legal_moves() filtered to captures / en passant / promotions.
// Walks every node of a bounded DFS from the perft test positions so the
// generator is exercised across castling, ep, promotion, pin, and check
// geometries — the same coverage perft gives the full generator.

#include <algorithm>
#include <cstdio>
#include <vector>

#include "check.hpp"
#include "movegen.hpp"
#include "position.hpp"

using namespace eclipse;

namespace {

// Reference: full legal movegen filtered to tactical moves. Castling must be
// excluded explicitly — it's encoded king-to-rook, so the naive
// piece_on(m.to()) != NoPiece test sees the (own) rook and misreads it as a
// capture. ab.cpp's old filter had exactly that bug, sending castling moves
// through qsearch as "captures".
void reference_captures(Position& pos, MoveList& out) {
    MoveList all;
    generate_legal_moves(pos, all);
    for (const Move m : all) {
        if (m.type() == Move::Castling) continue;
        const bool is_cap = pos.piece_on(m.to()) != NoPiece
                         || m.type() == Move::EnPassant
                         || m.type() == Move::Promotion;
        if (is_cap) out.push(m);
    }
}

std::vector<Move> sorted(const MoveList& ml) {
    std::vector<Move> v(ml.begin(), ml.end());
    std::sort(v.begin(), v.end(),
              [](Move a, Move b) { return a.raw() < b.raw(); });
    return v;
}

int g_nodes_checked = 0;

// DFS to `depth`, comparing the two capture generators at every node.
// `path` is the move sequence from the root FEN, printed on mismatch.
bool check_tree(Position& pos, int depth, std::vector<Move>& path) {
    MoveList fast;
    generate_legal_captures(pos, fast);
    MoveList ref;
    reference_captures(pos, ref);
    ++g_nodes_checked;

    if (sorted(fast) != sorted(ref)) {
        std::fprintf(stderr, "capture mismatch after moves:");
        for (const Move m : path) std::fprintf(stderr, " %s", m.to_uci().c_str());
        std::fprintf(stderr, "\n  fast:");
        for (const Move m : fast) std::fprintf(stderr, " %s", m.to_uci().c_str());
        std::fprintf(stderr, "\n  ref: ");
        for (const Move m : ref)  std::fprintf(stderr, " %s", m.to_uci().c_str());
        std::fprintf(stderr, "\n");
        return false;
    }

    if (depth == 0) return true;
    MoveList moves;
    generate_legal_moves(pos, moves);
    for (const Move m : moves) {
        StateInfo st;
        pos.do_move(m, st, /*snapshot_acc=*/false, /*update_acc=*/false);
        path.push_back(m);
        const bool ok = check_tree(pos, depth - 1, path);
        path.pop_back();
        pos.undo_move(m, st, /*restore_acc=*/false);
        if (!ok) return false;
    }
    return true;
}

struct Case {
    const char* fen;
    int         depth;
};

// Same positions as test_perft, at depths that keep the run to ~1s.
const Case kCases[] = {
    {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",                  3},
    {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",      3},
    {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                                 4},
    {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1",          3},
    {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",                 3},
    {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",  3},
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
        std::vector<Move> path;
        ECLIPSE_CHECK(check_tree(p, c.depth, path));
    }
    std::fprintf(stderr, "checked %d nodes\n", g_nodes_checked);
    return eclipse::test::summarize("movegen_captures");
}
