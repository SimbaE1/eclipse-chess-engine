// SPDX-License-Identifier: GPL-3.0-or-later
#include "syzygy.hpp"

#include <iostream>

// Fathom Syzygy TB probe library.
// tbprobe.h is a C header wrapped in extern "C".
#include <tbprobe.h>

namespace eclipse::syzygy {

namespace {
bool g_enabled = false;
}

void init(const std::string& path) {
    if (path.empty() || path == "<empty>") return;
    g_enabled = tb_init(path.c_str());
    if (g_enabled && TB_LARGEST > 0)
        std::cout << "info string Syzygy TBs loaded: up to "
                  << TB_LARGEST << " pieces" << std::endl;
    else
        std::cout << "info string Syzygy: no TB files found at " << path << std::endl;
}

bool     is_enabled()   { return g_enabled && TB_LARGEST > 0; }
unsigned max_pieces()   { return TB_LARGEST; }

static unsigned ep_arg(const Position& pos) {
    return pos.ep_square() == SquareNone ? 0u : static_cast<unsigned>(pos.ep_square());
}

// Use tb_probe_wdl_impl directly to bypass Fathom's conservative rule50 check.
// We want theoretical WDL for interior search nodes; the engine's existing
// 50-move-rule handling already covers the practical draw case.
unsigned probe_wdl(const Position& pos) {
    if (!g_enabled) return kTbFailed;
    // Guard: if the opponent's king is in check the position is illegal
    // (arose from a bad FEN). Fathom's move generator will crash on such
    // positions (it can generate king captures, then lsb() on empty bb).
    const Color opp = ~pos.side_to_move();
    if (pos.is_square_attacked(pos.king_square(opp), pos.side_to_move()))
        return kTbFailed;
    return static_cast<unsigned>(tb_probe_wdl_impl(
        pos.pieces(White),
        pos.pieces(Black),
        pos.pieces(King),
        pos.pieces(Queen),
        pos.pieces(Rook),
        pos.pieces(Bishop),
        pos.pieces(Knight),
        pos.pieces(Pawn),
        ep_arg(pos),
        pos.side_to_move() == White));
}

static unsigned castling_arg(const Position& pos) {
    unsigned c = 0u;
    const auto cr = pos.castling_rights();
    if (cr & WhiteKingside)  c |= 0x1u;  // TB_CASTLING_K
    if (cr & WhiteQueenside) c |= 0x2u;  // TB_CASTLING_Q
    if (cr & BlackKingside)  c |= 0x4u;  // TB_CASTLING_k
    if (cr & BlackQueenside) c |= 0x8u;  // TB_CASTLING_q
    return c;
}

unsigned probe_root(const Position& pos, unsigned* results) {
    if (!g_enabled) return kTbFailed;
    const Color opp2 = ~pos.side_to_move();
    if (pos.is_square_attacked(pos.king_square(opp2), pos.side_to_move()))
        return kTbFailed;
    return static_cast<unsigned>(tb_probe_root(
        pos.pieces(White),
        pos.pieces(Black),
        pos.pieces(King),
        pos.pieces(Queen),
        pos.pieces(Rook),
        pos.pieces(Bishop),
        pos.pieces(Knight),
        pos.pieces(Pawn),
        static_cast<unsigned>(pos.halfmove_clock()),
        castling_arg(pos),
        ep_arg(pos),
        pos.side_to_move() == White,
        results));
}

RootBest probe_root_best(const Position& pos) {
    if (!g_enabled) return {kTbFailed, 0, 0, 0, 0};
    if (pos.castling_rights() != NoCastling) return {kTbFailed, 0, 0, 0, 0};

    // TbRootMoves is ~100 KB — use a static to avoid stack overflow.
    // Safe: this function is only called from the main thread before any
    // search workers launch, and tb_probe_root_wdl is documented NOT thread-safe.
    static TbRootMoves s_rm;
    const int ok = tb_probe_root_wdl(
        pos.pieces(White),
        pos.pieces(Black),
        pos.pieces(King),
        pos.pieces(Queen),
        pos.pieces(Rook),
        pos.pieces(Bishop),
        pos.pieces(Knight),
        pos.pieces(Pawn),
        static_cast<unsigned>(pos.halfmove_clock()),
        0u,            // castling (guarded above)
        ep_arg(pos),
        pos.side_to_move() == White,
        true,          // useRule50
        &s_rm);

    if (!ok || s_rm.size == 0) return {kTbFailed, 0, 0, 0, 0};

    // Pick the move with the highest tbRank (1000=win, 899=cursed-win,
    // 0=draw, -899=blessed-loss, -1000=loss).
    const TbRootMove* best = &s_rm.moves[0];
    for (unsigned i = 1; i < s_rm.size; ++i) {
        if (s_rm.moves[i].tbRank > best->tbRank)
            best = &s_rm.moves[i];
    }

    unsigned wdl;
    if      (best->tbRank >= 1000) wdl = kTbWin;
    else if (best->tbRank >= 899)  wdl = kTbCursedWin;
    else if (best->tbRank >= 0)    wdl = kTbDraw;
    else if (best->tbRank >= -899) wdl = kTbBlessedLoss;
    else                           wdl = kTbLoss;

    return {
        wdl,
        static_cast<unsigned>(TB_MOVE_FROM(best->move)),
        static_cast<unsigned>(TB_MOVE_TO(best->move)),
        static_cast<unsigned>(TB_MOVE_PROMOTES(best->move)),
        0u   // DTZ not available without .rtbz files
    };
}

}  // namespace eclipse::syzygy
