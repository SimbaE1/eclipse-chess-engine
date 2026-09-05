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

    // Side to move just played `m`, so they "lose" the piece that moved.
    int d = 0;
    Color stm = pos.side_to_move();
    PieceType active_piece = type_of(pos.piece_on(from));

    // Remove the moving piece from occupancy and update attackers.
    //
    // The `& occ` is load-bearing. attackers_to() builds its result from the
    // pos.pieces() bitboards, which do NOT change as this loop removes pieces
    // -- only our local `occ` does. Without the mask, a piece that has already
    // captured is still a member of pos.pieces() and gets handed back as an
    // attacker again on the next refresh, so find_least_valuable re-selects
    // it. Sliders sometimes escape this by being blocked once occ changes, but
    // knights, kings and pawns never do: knight_attacks()/king_attacks() and
    // the pawn tables ignore occ entirely, so a knight that captured is
    // re-selected on every subsequent iteration until the d >= 31 safety cap.
    // Masking by occ drops anything no longer on the board.
    occ ^= (1ULL << from);
    Bitboard attackers = pos.attackers_to(to, occ) & occ;

    while (true) {
        d++;
        // The current side to move "loses" their active piece to gain the previous gain.
        gain[d] = kPieceValue[active_piece] - gain[d - 1];

        // No early cutoff here. The textbook swap algorithm prunes on
        //     max(-gain[d-1], gain[d]) < 0
        // but that test is only sound in the textbook's ply alignment, where
        // gain[d] is formed from the attacker about to capture. This loop
        // forms it from active_piece -- the piece already standing on the
        // square -- which is one ply earlier, so the invariant does not hold
        // and the break discards a gain[] entry the backpropagation below
        // still needs. Against a brute-force reference over 205227 captures
        // the cutoff was wrong on 11.4% of them with `< 0` and 23.7% with the
        // `<= 0` it shipped with; removing it entirely is exact.
        //
        // It also cost nothing to remove: the swap loop is bounded by the
        // number of attackers on one square, which is small.

        stm = ~stm;
        Square next_from = find_least_valuable(pos, stm, attackers);
        if (next_from == SquareNone) break;

        active_piece = type_of(pos.piece_on(next_from));
        occ ^= (1ULL << next_from);
        
        // Refresh attackers to account for X-rays, masking off pieces that
        // have already been captured out of `occ` (see above).
        attackers = pos.attackers_to(to, occ) & occ;
        
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
