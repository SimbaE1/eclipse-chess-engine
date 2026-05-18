// SPDX-License-Identifier: GPL-3.0-or-later
#include "movegen.hpp"

#include "attacks.hpp"
#include "bitboard.hpp"

namespace eclipse {

namespace {

template <Color Us>
void generate_pawn_moves(const Position& pos, MoveList& out) {
    constexpr Color Them = Us == White ? Black : White;
    constexpr Direction Up      = Us == White ? North     : South;
    constexpr Direction UpLeft  = Us == White ? NorthWest : SouthEast;
    constexpr Direction UpRight = Us == White ? NorthEast : SouthWest;
    constexpr Bitboard  PromoSourceRank = Us == White ? rank_bb(Rank7) : rank_bb(Rank2);
    constexpr Bitboard  DoublePushRank  = Us == White ? rank_bb(Rank3) : rank_bb(Rank6);

    const Bitboard pawns        = pos.pieces(Us, Pawn);
    const Bitboard pawns_promo  = pawns &  PromoSourceRank;
    const Bitboard pawns_no_pr  = pawns & ~PromoSourceRank;
    const Bitboard empty        = ~pos.occupied();
    // Kings are never legal capture targets - excluding them here keeps the
    // engine from producing positions with a missing king, which would crash
    // is_square_attacked downstream (king_square returns SquareNone).
    const Bitboard enemy        = pos.pieces(Them) & ~pos.pieces(Them, King);

    // Single pushes (no promotion).
    {
        Bitboard single = shift<Up>(pawns_no_pr) & empty;
        Bitboard b = single;
        while (b) {
            const Square to   = pop_lsb(b);
            const Square from = Square(static_cast<int>(to) - Up);
            out.push(Move(from, to));
        }
        // Double pushes piggyback off the single-push set: only pawns that
        // landed on rank 3/6 after one push can attempt a second.
        Bitboard doublep = shift<Up>(single & DoublePushRank) & empty;
        while (doublep) {
            const Square to   = pop_lsb(doublep);
            const Square from = Square(static_cast<int>(to) - 2 * Up);
            out.push(Move(from, to));
        }
    }

    // Captures (no promotion).
    {
        Bitboard lcap = shift<UpLeft>(pawns_no_pr)  & enemy;
        while (lcap) {
            const Square to   = pop_lsb(lcap);
            const Square from = Square(static_cast<int>(to) - UpLeft);
            out.push(Move(from, to));
        }
        Bitboard rcap = shift<UpRight>(pawns_no_pr) & enemy;
        while (rcap) {
            const Square to   = pop_lsb(rcap);
            const Square from = Square(static_cast<int>(to) - UpRight);
            out.push(Move(from, to));
        }
    }

    // Promotions: push + both captures.
    if (pawns_promo) {
        auto emit_promos = [&](Square from, Square to) {
            out.push(Move::make_promotion(from, to, Queen));
            out.push(Move::make_promotion(from, to, Rook));
            out.push(Move::make_promotion(from, to, Bishop));
            out.push(Move::make_promotion(from, to, Knight));
        };

        Bitboard push_pr = shift<Up>(pawns_promo) & empty;
        while (push_pr) {
            const Square to   = pop_lsb(push_pr);
            const Square from = Square(static_cast<int>(to) - Up);
            emit_promos(from, to);
        }
        Bitboard lcap_pr = shift<UpLeft>(pawns_promo)  & enemy;
        while (lcap_pr) {
            const Square to   = pop_lsb(lcap_pr);
            const Square from = Square(static_cast<int>(to) - UpLeft);
            emit_promos(from, to);
        }
        Bitboard rcap_pr = shift<UpRight>(pawns_promo) & enemy;
        while (rcap_pr) {
            const Square to   = pop_lsb(rcap_pr);
            const Square from = Square(static_cast<int>(to) - UpRight);
            emit_promos(from, to);
        }
    }

    // En passant. Any pawn of ours that attacks the ep square can play it.
    // pawn_attacks(Them, ep_sq) yields exactly the squares from which a pawn
    // of Us could capture on ep_sq (the relation is symmetric).
    if (pos.ep_square() != SquareNone) {
        const Square ep = pos.ep_square();
        Bitboard attackers = pawn_attacks(Them, ep) & pawns;
        while (attackers) {
            const Square from = pop_lsb(attackers);
            out.push(Move::make_en_passant(from, ep));
        }
    }
}

template <PieceType Pt>
void generate_piece_moves(const Position& pos, MoveList& out, Bitboard target) {
    const Color    us  = pos.side_to_move();
    const Bitboard occ = pos.occupied();
    Bitboard pieces = pos.pieces(us, Pt);
    while (pieces) {
        const Square from = pop_lsb(pieces);
        Bitboard attacks = 0;
        if constexpr (Pt == Knight) attacks = knight_attacks(from);
        if constexpr (Pt == Bishop) attacks = bishop_attacks(from, occ);
        if constexpr (Pt == Rook)   attacks = rook_attacks  (from, occ);
        if constexpr (Pt == Queen)  attacks = queen_attacks (from, occ);
        Bitboard moves = attacks & target;
        while (moves) {
            const Square to = pop_lsb(moves);
            out.push(Move(from, to));
        }
    }
}

template <Color Us>
void generate_king_moves(const Position& pos, MoveList& out) {
    constexpr Color Them = Us == White ? Black : White;
    constexpr Square KingHome      = Us == White ? E1 : E8;
    constexpr Square KingSideTo    = Us == White ? G1 : G8;
    constexpr Square KingSideMid   = Us == White ? F1 : F8;
    constexpr Square QueenSideTo   = Us == White ? C1 : C8;
    constexpr Square QueenSideMid  = Us == White ? D1 : D8;
    constexpr Square QueenSideKnt  = Us == White ? B1 : B8;
    constexpr CastlingRights KingSide  = Us == White ? WhiteKingside  : BlackKingside;
    constexpr CastlingRights QueenSide = Us == White ? WhiteQueenside : BlackQueenside;

    const Bitboard our_pieces = pos.pieces(Us);
    const Square   king_sq    = pos.king_square(Us);

    Bitboard targets = king_attacks(king_sq) & ~our_pieces & ~pos.pieces(Them, King);
    while (targets) {
        const Square to = pop_lsb(targets);
        out.push(Move(king_sq, to));
    }

    const Bitboard occ = pos.occupied();

    // Castling: rights present, path empty, king not in/through/into check.
    if (pos.castling_rights() & KingSide) {
        const Bitboard between = square_bb(KingSideMid) | square_bb(KingSideTo);
        if (!(occ & between)
            && !pos.is_square_attacked(KingHome,    Them)
            && !pos.is_square_attacked(KingSideMid, Them)
            && !pos.is_square_attacked(KingSideTo,  Them)) {
            out.push(Move::make_castling(KingHome, KingSideTo));
        }
    }
    if (pos.castling_rights() & QueenSide) {
        const Bitboard between = square_bb(QueenSideMid)
                               | square_bb(QueenSideTo)
                               | square_bb(QueenSideKnt);
        if (!(occ & between)
            && !pos.is_square_attacked(KingHome,     Them)
            && !pos.is_square_attacked(QueenSideMid, Them)
            && !pos.is_square_attacked(QueenSideTo,  Them)) {
            out.push(Move::make_castling(KingHome, QueenSideTo));
        }
    }
}

template <Color Us>
void generate_all(const Position& pos, MoveList& out) {
    constexpr Color Them = Us == White ? Black : White;
    // Targets = anywhere except our own pieces and the enemy king. See the
    // note in generate_pawn_moves about why kings are excluded.
    const Bitboard target = ~pos.pieces(Us) & ~pos.pieces(Them, King);
    generate_pawn_moves<Us>(pos, out);
    generate_piece_moves<Knight>(pos, out, target);
    generate_piece_moves<Bishop>(pos, out, target);
    generate_piece_moves<Rook>  (pos, out, target);
    generate_piece_moves<Queen> (pos, out, target);
    generate_king_moves<Us>(pos, out);
}

}  // namespace

void generate_pseudo_legal_moves(const Position& pos, MoveList& out) {
    if (pos.side_to_move() == White)
        generate_all<White>(pos, out);
    else
        generate_all<Black>(pos, out);
}

void generate_legal_moves(Position& pos, MoveList& out) {
    MoveList pseudo;
    generate_pseudo_legal_moves(pos, pseudo);

    StateInfo st;
    for (const Move m : pseudo) {
        pos.do_move(m, st);
        // After do_move, side_to_move has flipped. Our king is the king of
        // the side that just moved; it must not be attacked by the side now
        // on the move.
        const Color   us       = ~pos.side_to_move();
        const Square  king_sq  = pos.king_square(us);
        const bool    legal    = !pos.is_square_attacked(king_sq, ~us);
        pos.undo_move(m, st);
        if (legal) out.push(m);
    }
}

}  // namespace eclipse
