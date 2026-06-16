// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <string>

#include "position.hpp"

namespace eclipse::syzygy {

// WDL result constants (mirror Fathom's TB_* defines without pulling in tbprobe.h).
constexpr unsigned kTbLoss        = 0u;
constexpr unsigned kTbBlessedLoss = 1u;
constexpr unsigned kTbDraw        = 2u;
constexpr unsigned kTbCursedWin   = 3u;
constexpr unsigned kTbWin         = 4u;
constexpr unsigned kTbFailed      = 0xFFFFFFFFu;

void     init(const std::string& path);
bool     is_enabled();
unsigned max_pieces();

// WDL probe for interior search nodes. Ignores rule50 (theoretical WDL).
// Requires pos.castling_rights() == NoCastling (caller must check).
// Returns kTbWin/kTbCursedWin/kTbDraw/kTbBlessedLoss/kTbLoss or kTbFailed.
unsigned probe_wdl(const Position& pos);

// Root DTZ probe. Fills results[0..kTbMaxMoves-1], terminated by kTbFailed.
// Returns the best result for the side to move, or kTbFailed.
constexpr unsigned kTbMaxMoves = 193u;
unsigned probe_root(const Position& pos, unsigned* results);

// Best root move from the tablebase. Returns wdl==kTbFailed if the probe
// failed or the position is checkmate/stalemate. Squares use A1=0 encoding.
// promotes: 0=none, 1=Q, 2=R, 3=B, 4=N (Fathom TB_PROMOTES_* encoding).
struct RootBest {
    unsigned wdl;
    unsigned from;
    unsigned to;
    unsigned promotes;
    unsigned dtz;
};
RootBest probe_root_best(const Position& pos);

}  // namespace eclipse::syzygy
