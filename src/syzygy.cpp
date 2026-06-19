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
    // Illegal to probe if the side NOT to move is in check.
    const Color opp = ~pos.side_to_move();
    if (pos.is_square_attacked(pos.king_square(opp), pos.side_to_move()))
        return {kTbFailed, 0, 0, 0, 0};

    // DTZ-based root probe (was tb_probe_root_wdl). The WDL root probe ranks
    // every winning move equally, so the engine had no progress gradient and
    // would shuffle a won position into a 50-move / repetition draw (e.g. a
    // drawn K+Q vs K). tb_probe_root ranks by distance-to-zero, so it returns a
    // move that genuinely converges to mate AND respects the halfmove counter.
    // It needs the .rtbz tables — returns TB_RESULT_FAILED if they're absent, so
    // we fall back to search. Main-thread only (not thread-safe); called before
    // any search workers launch.
    const unsigned res = tb_probe_root(
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
        nullptr);      // don't need the per-move results array

    if (res == TB_RESULT_FAILED || res == TB_RESULT_CHECKMATE ||
        res == TB_RESULT_STALEMATE)
        return {kTbFailed, 0, 0, 0, 0};

    unsigned wdl;
    switch (TB_GET_WDL(res)) {
        case TB_WIN:          wdl = kTbWin;         break;
        case TB_CURSED_WIN:   wdl = kTbCursedWin;   break;
        case TB_DRAW:         wdl = kTbDraw;        break;
        case TB_BLESSED_LOSS: wdl = kTbBlessedLoss; break;
        default:              wdl = kTbLoss;        break;
    }

    return {
        wdl,
        static_cast<unsigned>(TB_GET_FROM(res)),
        static_cast<unsigned>(TB_GET_TO(res)),
        static_cast<unsigned>(TB_GET_PROMOTES(res)),
        static_cast<unsigned>(TB_GET_DTZ(res)),
    };
}

}  // namespace eclipse::syzygy
