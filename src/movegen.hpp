// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "move.hpp"
#include "position.hpp"

namespace eclipse {

// Generates all pseudo-legal moves from `pos`. "Pseudo-legal" means the move
// is shape-correct (piece moves the right way, captures are valid, etc.) but
// the resulting position may leave our king in check. Callers wanting strictly
// legal moves should use generate_legal_moves().
void generate_pseudo_legal_moves(const Position& pos, MoveList& out);

// Generates strictly legal moves. Internally walks pseudo-legal moves and
// filters by playing each on `pos` and checking our king's safety. `pos` is
// returned to its original state on exit; the helper must mutate it during the
// pseudo-legal check pass.
void generate_legal_moves(Position& pos, MoveList& out);

}  // namespace eclipse
