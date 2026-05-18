// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>

#include "types.hpp"

namespace eclipse::zobrist {

// One 64-bit key per (piece, square). The NoPiece slot is initialised but
// callers should not XOR it in - we never "place" an empty piece. Indexing it
// produces a perfectly fine hash, so the discipline is the only guard.
extern std::uint64_t piece_square[PieceNB][SquareNB];

// Indexed by the 4-bit castling-rights mask (0..15).
extern std::uint64_t castling[CastlingRightsNB];

// File-only en-passant key. Rank is implied by the side to move, so only the
// file actually affects what captures are legal.
extern std::uint64_t en_passant_file[FileNB];

// XORed in when it is Black's turn to move.
extern std::uint64_t side_to_move;

// Idempotent. Call once at startup before constructing any Position.
void init();

}  // namespace eclipse::zobrist
