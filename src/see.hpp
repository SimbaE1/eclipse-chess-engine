// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "position.hpp"

namespace eclipse {

// Static Exchange Evaluation (SEE). Returns the estimated material balance 
// of a series of exchanges on the target square of move `m`, from the 
// perspective of the side to move. Positive means we win material.
int see(const Position& pos, Move m) noexcept;

// Returns true if the SEE value of move `m` is >= threshold.
// Faster than calling see() and comparing if we only care about the threshold.
bool see_ge(const Position& pos, Move m, int threshold) noexcept;

}  // namespace eclipse
