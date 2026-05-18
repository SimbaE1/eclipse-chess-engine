// SPDX-License-Identifier: GPL-3.0-or-later
#include "eval.hpp"

#include "bitboard.hpp"
#include "nnue.hpp"

namespace eclipse {

Score material_evaluate(const Position& pos) noexcept {
    int score = 0;
    for (int pt = Pawn; pt <= Queen; ++pt) {
        const PieceType p = PieceType(pt);
        score += kPieceValue[pt] *
                 (popcount(pos.pieces(White, p)) - popcount(pos.pieces(Black, p)));
    }
    return pos.side_to_move() == White ? score : -score;
}

Score evaluate(const Position& pos) noexcept {
    // The NNUE - even the throwaway one - encodes material plus positional
    // signal. Fall back to pure-material only when no weights are loaded.
    if (nnue::is_loaded()) return nnue::evaluate(pos);
    return material_evaluate(pos);
}

}  // namespace eclipse
