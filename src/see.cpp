// SPDX-License-Identifier: GPL-3.0-or-later
#include "see.hpp"
#include "attacks.hpp"
#include "types.hpp"

#include <algorithm>

namespace eclipse {

namespace {

// Helper to find the least valuable piece of color `stm` that attacks `sq`.
// Returns SquareNone if no attacker is found.
Square find_least_valuable(const Position& pos, Color stm, Bitboard attackers) noexcept {
    for (PieceType pt : {Pawn, Knight, Bishop, Rook, Queen, King}) {
        Bitboard subset = attackers & pos.pieces(stm, pt);
        if (subset) return lsb(subset);
    }
    return SquareNone;
}

}  // namespace

int see(const Position& pos, Move m) noexcept {
    Square from = m.from();
    Square to   = m.to();

    // The first "gain" is the piece captured on the target square.
    Piece captured = pos.piece_on(to);
    if (m.type() == Move::EnPassant) {
        captured = make_piece(~pos.side_to_move(), Pawn);
    } else if (m.type() == Move::Promotion) {
        // Promotion is treated as: we lose a pawn and gain the promotion piece.
        // But the first capture is still whatever was on `to`.
    }

    int gain[32];
    gain[0] = kPieceValue[type_of(captured)];
    
    // If it's a promotion, we also gain the difference between the promoted piece
    // and a pawn immediately.
    if (m.type() == Move::Promotion) {
        gain[0] += kPieceValue[m.promotion_piece()] - kPieceValue[Pawn];
    }

    Bitboard occ = pos.occupied();
    Bitboard attackers = pos.attackers_to(to, occ);
    
    // Side to move just played `m`, so they "lose" the piece that moved.
    int d = 0;
    Color stm = pos.side_to_move();
    PieceType active_piece = type_of(pos.piece_on(from));

    // Remove the moving piece from occupancy and update attackers.
    occ ^= (1ULL << from);
    attackers = pos.attackers_to(to, occ);

    while (true) {
        d++;
        // The current side to move "loses" their active piece to gain the previous gain.
        gain[d] = kPieceValue[active_piece] - gain[d - 1];
        
        // Pruning: if we can't even beat the previous gain by standing still, stop.
        if (std::max(-gain[d - 1], gain[d]) <= 0) break;

        stm = ~stm;
        Square next_from = find_least_valuable(pos, stm, attackers);
        if (next_from == SquareNone) break;

        active_piece = type_of(pos.piece_on(next_from));
        occ ^= (1ULL << next_from);
        
        // Refresh attackers to account for X-rays.
        attackers = pos.attackers_to(to, occ);
        
        if (d >= 31) break; // Safety cap
    }

    // Backpropagate the scores. Each side chooses the maximum of (not capturing) or (capturing).
    while (--d > 0) {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    }

    return gain[0];
}

bool see_ge(const Position& pos, Move m, int threshold) noexcept {
    // For now, simple wrapper. Could be optimized further.
    return see(pos, m) >= threshold;
}

}  // namespace eclipse
