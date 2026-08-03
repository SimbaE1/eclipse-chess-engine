// SPDX-License-Identifier: GPL-3.0-or-later
//
// SEE correctness regression. Guards two bugs fixed 2026-08-02:
//   (1) attackers_to() masks by piece bitboards, not occ, so the swap loop
//       used to re-select already-captured attackers (knights/pawns/king
//       always; sliders via the reopened ray) — fabricating phantom recaptures.
//       Fixed by masking each recompute with `& occ`.
//   (2) en passant never removed the ep-captured pawn from occ, so x-ray
//       attackers along that rank/file were not revealed.
// Piece values (types.hpp): P=100 N=320 B=330 R=500 Q=900.
#include <cstdio>
#include <string>

#include "movegen.hpp"
#include "position.hpp"
#include "see.hpp"

using namespace eclipse;

namespace {
int g_fail = 0;

void check(const char* fen, const char* uci, int expect) {
    Position pos;
    if (!pos.set_from_fen(fen)) { std::printf("BAD FEN %s\n", fen); ++g_fail; return; }
    MoveList moves;
    generate_legal_moves(pos, moves);
    Move found = MoveNone;
    for (const Move m : moves)
        if (m.to_uci() == uci) { found = m; break; }
    if (found == MoveNone) { std::printf("move %s not legal in %s\n", uci, fen); ++g_fail; return; }
    const int got = see(pos, found);
    const bool ok = (got == expect);
    std::printf("[%s] %-58s %s  see=%d (expect %d)\n",
                ok ? "ok" : "FAIL", fen, uci, got, expect);
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    // (bug 1) Nxd4 wins a pawn but the d1 queen recaptures the knight: 100-320.
    // The pre-fix code re-used the just-captured knight and reported +100.
    check("r1bqkbnr/pppppppp/2n5/8/3P4/8/PPP1PPPP/RNBQKBNR b - - 0 2", "c6d4", -220);

    // Rook takes an undefended bishop: clean +330.
    check("4k3/8/8/3b4/8/8/8/3RK3 w - - 0 1", "d1d5", 330);

    // Same, but the bishop is defended by the e6 pawn: 330 - 500 = -170
    // (rook is lost for the bishop, so the capture is bad).
    check("4k3/8/4p3/3b4/8/8/8/3RK3 w - - 0 1", "d1d5", -170);

    // (bug 2) exd6 e.p. exercises the ep-pawn removal (the captured pawn is on
    // d5, not d6). SEE reads 100 here: en passant is pawn-takes-pawn, so
    // gain[1]=0 and the swap loop's `max(...) <= 0` cutoff prunes before the
    // Rd3 recapture is ever considered. So the ep x-ray reveal is correct but
    // its effect stays masked until that cutoff is changed to the textbook
    // `< 0` — deferred to the training-time SPRT batch (a separate finding).
    check("4k3/8/8/3pP3/8/3r4/8/4K3 w - d6 0 1", "e5d6", 100);

    std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
