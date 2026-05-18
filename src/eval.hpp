// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "position.hpp"
#include "types.hpp"

namespace eclipse {

using Score = int;

constexpr Score kInfinite       = 30000;
constexpr Score kMateScore      = 29000;
constexpr Score kMateInMaxPly   = kMateScore - 1000;
constexpr Score kDraw           = 0;

// Phase 0 placeholder: pure material. The NNUE this engine will eventually
// load replaces evaluate() wholesale; the scaffolding stays so search code
// has something to call against in the meantime.
constexpr Score kPieceValue[PieceTypeNB] = {
    0,     // NoPieceType
    100,   // Pawn
    320,   // Knight
    330,   // Bishop
    500,   // Rook
    900,   // Queen
    0,     // King - never counted; trades that lose the king are handled by
           // search's checkmate scoring instead.
    0,
};

// Returns the side-to-move evaluation in centipawn units (positive = stm
// better). Used by qsearch leaves.
Score evaluate(const Position& pos) noexcept;

// Fallback pure-material evaluation, used when NNUE is not loaded or for
// testing/benchmarking.
Score material_evaluate(const Position& pos) noexcept;

}  // namespace eclipse
