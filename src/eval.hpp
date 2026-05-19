// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "position.hpp"
#include "types.hpp"

namespace eclipse {

// Phase 0 placeholder: pure material. The NNUE this engine will eventually
// load replaces evaluate() wholesale; the scaffolding stays so search code
// has something to call against in the meantime.

// Returns the side-to-move evaluation in centipawn units (positive = stm
// better). Used by qsearch leaves.
Score evaluate(const Position& pos) noexcept;

// Fallback pure-material evaluation, used when NNUE is not loaded or for
// testing/benchmarking.
Score material_evaluate(const Position& pos) noexcept;

}  // namespace eclipse
