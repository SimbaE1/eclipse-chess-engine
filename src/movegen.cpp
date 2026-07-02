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
    constexpr Rank  HomeRank       = Us == White ? Rank1 : Rank8;
    constexpr Square KingSideDest  = Us == White ? G1 : G8;
    constexpr Square QueenSideDest = Us == White ? C1 : C8;
    constexpr Square KingSideRookDest  = Us == White ? F1 : F8;
    constexpr Square QueenSideRookDest = Us == White ? D1 : D8;
    constexpr CastlingRights KingSide  = Us == White ? WhiteKingside  : BlackKingside;
    constexpr CastlingRights QueenSide = Us == White ? WhiteQueenside : BlackQueenside;

    const Bitboard our_pieces = pos.pieces(Us);
    const Square   king_sq    = pos.king_square(Us);

    Bitboard targets = king_attacks(king_sq) & ~our_pieces & ~pos.pieces(Them, King);
    while (targets) {
        const Square to = pop_lsb(targets);
        out.push(Move(king_sq, to));
    }

    // Castling is only possible from the home rank.
    if (rank_of(king_sq) != HomeRank) return;

    const Bitboard occ = pos.occupied();

    // Build a bitboard of all squares between two squares on the same rank
    // (exclusive of endpoints). Works for any pair of squares on the same rank
    // because consecutive files within a rank are consecutive in the Square enum.
    auto between_excl = [](Square a, Square b) -> Bitboard {
        if (a == b) return Bitboard(0);
        if (a > b) { Square t = a; a = b; b = t; }
        Bitboard bb = 0;
        for (Square s = Square(a + 1); s < b; s = Square(s + 1))
            bb |= square_bb(s);
        return bb;
    };
    // Inclusive ray: all squares from a to b on the same rank (both included).
    auto ray_incl = [](Square a, Square b) -> Bitboard {
        if (a > b) { Square t = a; a = b; b = t; }
        Bitboard bb = 0;
        for (Square s = a; s <= b; s = Square(s + 1))
            bb |= square_bb(s);
        return bb;
    };

    // Castling: rights present, path empty (excluding king and castling rook
    // themselves), king not in/through/into check. Encoded as king-to-rook (KxR).
    if (pos.castling_rights() & KingSide) {
        const Square rook_sq = pos.castling_rook_sq(KingSide);
        if (rook_sq != SquareNone) {
            // Squares that must be vacant: union of king's travel path and rook's
            // travel path, minus the squares occupied by the king and rook themselves
            // (they are part of the move, not blockers).
            const Bitboard castlers     = square_bb(king_sq) | square_bb(rook_sq);
            const Bitboard must_empty   = (ray_incl(king_sq, KingSideDest)
                                         | ray_incl(rook_sq, KingSideRookDest)) & ~castlers;
            const Bitboard king_path    = ray_incl(king_sq, KingSideDest);
            const Bitboard occ_excl     = occ & ~castlers;
            if (!(occ_excl & must_empty)) {
                bool safe = true;
                Bitboard path = king_path;
                while (path) {
                    if (pos.is_square_attacked(pop_lsb(path), Them)) { safe = false; break; }
                }
                if (safe) out.push(Move::make_castling(king_sq, rook_sq));
            }
        }
    }
    if (pos.castling_rights() & QueenSide) {
        const Square rook_sq = pos.castling_rook_sq(QueenSide);
        if (rook_sq != SquareNone) {
            const Bitboard castlers     = square_bb(king_sq) | square_bb(rook_sq);
            const Bitboard must_empty   = (ray_incl(king_sq, QueenSideDest)
                                         | ray_incl(rook_sq, QueenSideRookDest)) & ~castlers;
            const Bitboard king_path    = ray_incl(king_sq, QueenSideDest);
            const Bitboard occ_excl     = occ & ~castlers;
            if (!(occ_excl & must_empty)) {
                bool safe = true;
                Bitboard path = king_path;
                while (path) {
                    if (pos.is_square_attacked(pop_lsb(path), Them)) { safe = false; break; }
                }
                if (safe) out.push(Move::make_castling(king_sq, rook_sq));
            }
        }
    }
    (void)between_excl;  // used only in future; suppress warning
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
        // Legality only inspects board geometry (king safety), so skip the NNUE
        // accumulator entirely: no snapshot, no incremental FT update, no
        // restore. Without this, every pseudo-legal move would do/undo three
        // 4 KB accumulator memcpys plus a full feature-transformer update — the
        // dominant cost of node expansion, all of it discarded.
        pos.do_move(m, st, /*snapshot_acc=*/false, /*update_acc=*/false);
        // After do_move, side_to_move has flipped. Our king is the king of
        // the side that just moved; it must not be attacked by the side now
        // on the move.
        const Color   us       = ~pos.side_to_move();
        const Square  king_sq  = pos.king_square(us);
        const bool    legal    = !pos.is_square_attacked(king_sq, ~us);
        pos.undo_move(m, st, /*restore_acc=*/false);
        if (legal) out.push(m);
    }
}

}  // namespace eclipse
