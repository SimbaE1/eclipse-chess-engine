// SPDX-License-Identifier: GPL-3.0-or-later
#include "position.hpp"

#include <array>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <sstream>

#include "attacks.hpp"
#include "nnue.hpp"
#include "zobrist.hpp"

namespace eclipse {

namespace {

constexpr std::string_view kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Bit-parallel inline popcount — counts set bits in a CastlingRights value.
// Used to detect Chess960 FEN: if more than one right maps to the same file
// we skip (handled per-character in the loop). Not security critical, just a
// small helper to avoid a full <bit> include for one call.
[[maybe_unused]] constexpr int count_bits(int v) noexcept {
    int c = 0;
    while (v) { c += v & 1; v >>= 1; }
    return c;
}

Piece piece_from_char(char c) noexcept {
    switch (c) {
        case 'P': return WPawn;   case 'N': return WKnight;
        case 'B': return WBishop; case 'R': return WRook;
        case 'Q': return WQueen;  case 'K': return WKing;
        case 'p': return BPawn;   case 'n': return BKnight;
        case 'b': return BBishop; case 'r': return BRook;
        case 'q': return BQueen;  case 'k': return BKing;
        default:  return NoPiece;
    }
}

char piece_to_char(Piece p) noexcept {
    constexpr char kChars[PieceNB] = {
        '.',
        'P', 'N', 'B', 'R', 'Q', 'K',
        '?',
        '?',
        'p', 'n', 'b', 'r', 'q', 'k',
        '?',
    };
    return kChars[p];
}

// The old compile-time revoke table is replaced by a per-position
// castling_revoke_[] built in set_from_fen to support Chess960 starting
// positions where kings and rooks may not be on E/A/H files.

}  // namespace

Position::Position() {
    set_from_fen(kStartFen);
}

Position Position::startpos() {
    Position p;
    p.set_from_fen(kStartFen);
    return p;
}

void Position::clear() {
    for (Bitboard& b : by_piece_) b = 0;
    for (Bitboard& b : by_color_) b = 0;
    for (Piece&    p : board_)    p = NoPiece;
    stm_              = White;
    castling_         = NoCastling;
    ep_               = SquareNone;
    halfmove_clock_   = 0;
    fullmove_number_  = 1;
    key_              = 0;
    for (Square& s : castling_rook_sq_)    s  = SquareNone;
    for (CastlingRights& c : castling_revoke_) c = NoCastling;
}

void Position::put_piece(Piece p, Square s) {
    board_[s] = p;
    by_piece_[p] |= square_bb(s);
    by_color_[color_of(p)] |= square_bb(s);
    key_ ^= zobrist::piece_square[p][s];
}

void Position::remove_piece(Square s) {
    const Piece p = board_[s];
    board_[s] = NoPiece;
    by_piece_[p] &= ~square_bb(s);
    by_color_[color_of(p)] &= ~square_bb(s);
    key_ ^= zobrist::piece_square[p][s];
}

void Position::move_piece(Square from, Square to) {
    const Piece p = board_[from];
    const Bitboard mask = square_bb(from) | square_bb(to);
    by_piece_[p] ^= mask;
    by_color_[color_of(p)] ^= mask;
    board_[from] = NoPiece;
    board_[to]   = p;
    key_ ^= zobrist::piece_square[p][from] ^ zobrist::piece_square[p][to];
}

void Position::raw_put_piece(Piece p, Square s) noexcept {
    board_[s] = p;
    by_piece_[p] |= square_bb(s);
    by_color_[color_of(p)] |= square_bb(s);
}

void Position::raw_remove_piece(Square s) noexcept {
    const Piece p = board_[s];
    board_[s] = NoPiece;
    by_piece_[p] &= ~square_bb(s);
    by_color_[color_of(p)] &= ~square_bb(s);
}

void Position::raw_move_piece(Square from, Square to) noexcept {
    const Piece p = board_[from];
    const Bitboard mask = square_bb(from) | square_bb(to);
    by_piece_[p] ^= mask;
    by_color_[color_of(p)] ^= mask;
    board_[from] = NoPiece;
    board_[to]   = p;
}

std::uint64_t Position::compute_key() const noexcept {
    std::uint64_t k = 0;
    for (int s = 0; s < SquareNB; ++s) {
        const Piece p = board_[s];
        if (p != NoPiece) k ^= zobrist::piece_square[p][s];
    }
    k ^= zobrist::castling[castling_];
    if (ep_ != SquareNone) k ^= zobrist::en_passant_file[file_of(ep_)];
    if (stm_ == Black)     k ^= zobrist::side_to_move;
    return k;
}

bool Position::set_from_fen(std::string_view fen) {
    zobrist::init();
    init_attacks();
    clear();

    // Tokenise on whitespace.
    std::array<std::string_view, 6> tok{};
    std::size_t pos = 0;
    int ntok = 0;
    while (pos < fen.size() && ntok < 6) {
        while (pos < fen.size() && std::isspace(static_cast<unsigned char>(fen[pos]))) ++pos;
        const std::size_t start = pos;
        while (pos < fen.size() && !std::isspace(static_cast<unsigned char>(fen[pos]))) ++pos;
        if (start < pos) tok[static_cast<std::size_t>(ntok++)] = fen.substr(start, pos - start);
    }
    if (ntok < 4) return false;

    // Field 1: piece placement.
    int rank = 7;
    int file = 0;
    for (char c : tok[0]) {
        if (c == '/') {
            if (file != 8 || rank == 0) return false;
            --rank;
            file = 0;
        } else if (c >= '1' && c <= '8') {
            file += (c - '0');
            if (file > 8) return false;
        } else {
            const Piece p = piece_from_char(c);
            if (p == NoPiece) return false;
            if (file >= 8 || rank < 0) return false;
            raw_put_piece(p, make_square(File(file), Rank(rank)));
            ++file;
        }
    }
    if (rank != 0 || file != 8) return false;

    // Field 2: side to move.
    if      (tok[1] == "w") stm_ = White;
    else if (tok[1] == "b") stm_ = Black;
    else return false;

    // Field 3: castling availability.
    // Supports both standard ("KQkq") and Chess960 file-letter notation ("AHah").
    castling_ = NoCastling;
    if (tok[2] != "-") {
        for (char c : tok[2]) {
            if (c == 'K') {
                castling_ |= WhiteKingside;
                castling_rook_sq_[WhiteKingside] = H1;
            } else if (c == 'Q') {
                castling_ |= WhiteQueenside;
                castling_rook_sq_[WhiteQueenside] = A1;
            } else if (c == 'k') {
                castling_ |= BlackKingside;
                castling_rook_sq_[BlackKingside] = H8;
            } else if (c == 'q') {
                castling_ |= BlackQueenside;
                castling_rook_sq_[BlackQueenside] = A8;
            } else if (c >= 'A' && c <= 'H') {
                // Chess960: white rook on that file. Determine side by comparing
                // rook file to the white king's file.
                const File   f       = File(c - 'A');
                const Square rook_sq = make_square(f, Rank1);
                const Square wk      = king_square(White);
                if (wk != SquareNone && f > file_of(wk)) {
                    castling_ |= WhiteKingside;
                    castling_rook_sq_[WhiteKingside] = rook_sq;
                } else if (wk != SquareNone) {
                    castling_ |= WhiteQueenside;
                    castling_rook_sq_[WhiteQueenside] = rook_sq;
                }
            } else if (c >= 'a' && c <= 'h') {
                // Chess960: black rook on that file.
                const File   f       = File(c - 'a');
                const Square rook_sq = make_square(f, Rank8);
                const Square bk      = king_square(Black);
                if (bk != SquareNone && f > file_of(bk)) {
                    castling_ |= BlackKingside;
                    castling_rook_sq_[BlackKingside] = rook_sq;
                } else if (bk != SquareNone) {
                    castling_ |= BlackQueenside;
                    castling_rook_sq_[BlackQueenside] = rook_sq;
                }
            } else {
                return false;
            }
        }
    }

    // Build the per-position castling-revoke table from the actual initial
    // king and rook squares (replaces the old compile-time kCastlingRevoke).
    {
        const Square wk = king_square(White);
        const Square bk = king_square(Black);
        if (wk != SquareNone && (castling_ & WhiteCastling))
            castling_revoke_[wk] |= WhiteCastling;
        if (bk != SquareNone && (castling_ & BlackCastling))
            castling_revoke_[bk] |= BlackCastling;
        for (const CastlingRights cr :
             {WhiteKingside, WhiteQueenside, BlackKingside, BlackQueenside}) {
            const Square rsq = castling_rook_sq_[cr];
            if (rsq != SquareNone) castling_revoke_[rsq] |= cr;
        }
    }

    // Field 4: en passant target square.
    if (tok[3] == "-") {
        ep_ = SquareNone;
    } else if (tok[3].size() == 2) {
        char file_c = tok[3][0];
        char rank_c = tok[3][1];
        if (file_c < 'a' || file_c > 'h' || rank_c < '1' || rank_c > '8') return false;
        ep_ = make_square(File(file_c - 'a'), Rank(rank_c - '1'));
    } else {
        return false;
    }

    // Field 5: halfmove clock (optional - some test FENs omit).
    halfmove_clock_ = 0;
    if (ntok >= 5) {
        halfmove_clock_ = std::atoi(std::string(tok[4]).c_str());
        if (halfmove_clock_ < 0) halfmove_clock_ = 0;
    }

    // Field 6: fullmove number (optional).
    fullmove_number_ = 1;
    if (ntok >= 6) {
        fullmove_number_ = std::atoi(std::string(tok[5]).c_str());
        if (fullmove_number_ < 1) fullmove_number_ = 1;
    }

    key_ = compute_key();

    // Refresh the NNUE accumulator to match the freshly-loaded position. If
    // the network is not loaded yet, refresh() leaves acc_.computed=false and
    // evaluate() will lazy-refresh on first call.
    nnue::refresh(*this, acc_);

    return true;
}

std::string Position::fen() const {
    std::ostringstream oss;

    // Field 1: pieces.
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            const Square s = make_square(File(f), Rank(r));
            const Piece p = board_[s];
            if (p == NoPiece) {
                ++empty;
            } else {
                if (empty > 0) { oss << empty; empty = 0; }
                oss << piece_to_char(p);
            }
        }
        if (empty > 0) oss << empty;
        if (r > 0) oss << '/';
    }

    // Field 2.
    oss << ' ' << (stm_ == White ? 'w' : 'b');

    // Field 3: castling. Use K/Q/k/q for standard positions (rooks on A/H files),
    // file letters for Chess960 positions.
    oss << ' ';
    if (castling_ == NoCastling) {
        oss << '-';
    } else {
        if (castling_ & WhiteKingside)
            oss << (castling_rook_sq_[WhiteKingside] == H1
                    ? 'K' : char('A' + file_of(castling_rook_sq_[WhiteKingside])));
        if (castling_ & WhiteQueenside)
            oss << (castling_rook_sq_[WhiteQueenside] == A1
                    ? 'Q' : char('A' + file_of(castling_rook_sq_[WhiteQueenside])));
        if (castling_ & BlackKingside)
            oss << (castling_rook_sq_[BlackKingside] == H8
                    ? 'k' : char('a' + file_of(castling_rook_sq_[BlackKingside])));
        if (castling_ & BlackQueenside)
            oss << (castling_rook_sq_[BlackQueenside] == A8
                    ? 'q' : char('a' + file_of(castling_rook_sq_[BlackQueenside])));
    }

    // Field 4.
    oss << ' ';
    if (ep_ == SquareNone) oss << '-';
    else                   oss << square_to_string(ep_);

    // Field 5 / 6.
    oss << ' ' << halfmove_clock_ << ' ' << fullmove_number_;

    return oss.str();
}

bool Position::is_square_attacked(Square s, Color by) const noexcept {
    const Bitboard occ = occupied();
    // A pawn of color `by` attacks `s` iff a pawn of color `by` sits on a
    // square from which it could capture toward `s`. By symmetry, those
    // squares are exactly pawn_attacks(~by, s).
    if (pawn_attacks  (~by, s) & pieces(by, Pawn))               return true;
    if (knight_attacks(s)      & pieces(by, Knight))             return true;
    if (king_attacks  (s)      & pieces(by, King))               return true;
    const Bitboard bishops_queens = pieces(by, Bishop) | pieces(by, Queen);
    if (bishop_attacks(s, occ) & bishops_queens)                 return true;
    const Bitboard rooks_queens   = pieces(by, Rook)   | pieces(by, Queen);
    if (rook_attacks  (s, occ) & rooks_queens)                   return true;
    return false;
}

Bitboard Position::attackers_to(Square s, Bitboard occ) const noexcept {
    return (pawn_attacks(Black, s) & pieces(White, Pawn))
         | (pawn_attacks(White, s) & pieces(Black, Pawn))
         | (knight_attacks(s)      & (pieces(White, Knight) | pieces(Black, Knight)))
         | (rook_attacks(s, occ)   & (pieces(White, Rook)   | pieces(Black, Rook)   | pieces(White, Queen) | pieces(Black, Queen)))
         | (bishop_attacks(s, occ) & (pieces(White, Bishop) | pieces(Black, Bishop) | pieces(White, Queen) | pieces(Black, Queen)))
         | (king_attacks(s)        & (pieces(White, King)   | pieces(Black, King)));
}

void Position::do_move(Move m, StateInfo& st, bool snapshot_acc, bool update_acc) {
    using namespace zobrist;

    const Square     from = m.from();
    const Square     to   = m.to();
    const Move::Type mt   = m.type();
    const Color      us   = stm_;
    const Color      them = ~us;
    const Piece      moving    = board_[from];
    const PieceType  moving_pt = type_of(moving);

    // Save undo state.
    st.prev_castling = castling_;
    st.prev_ep       = ep_;
    st.prev_halfmove = halfmove_clock_;
    st.prev_key      = key_;
    st.captured      = (mt == Move::EnPassant)
                         ? make_piece(them, Pawn)
                         : board_[to];
    // Snapshot the pre-move accumulator so undo_move can restore it without
    // a full refresh. Skipped when the caller guarantees it will never undo
    // (snapshot_acc=false, e.g. the MCTS descent) — that 4 KB copy per ply is
    // otherwise pure waste there.
    if (snapshot_acc) st.accumulator = acc_;

    // The castling-rights hash key uses the full 4-bit mask, so we XOR out
    // the old value here and XOR in the new one at the bottom of this fn.
    key_ ^= castling[castling_];

    // Clear any prior en-passant square (key XOR + reset). A fresh ep gets
    // set below if this move is a pawn double-push.
    if (ep_ != SquareNone) {
        key_ ^= en_passant_file[file_of(ep_)];
        ep_ = SquareNone;
    }

    // Halfmove clock: increment by default; pawn moves and captures reset.
    ++halfmove_clock_;
    if (us == Black) ++fullmove_number_;
    if (moving_pt == Pawn || st.captured != NoPiece) halfmove_clock_ = 0;

    if (mt == Move::Castling) {
        // Castling is encoded internally as (king_from, rook_from). The rook
        // file relative to the king determines kingside vs queenside; the king
        // always lands on G/F for kingside, C/D for queenside (same as standard
        // chess), regardless of where they started in Chess960.
        const Square rook_from = to;   // `to` stores the rook's initial square
        const bool   kingside  = (file_of(rook_from) > file_of(from));
        const Square king_to   = make_square(kingside ? FileG : FileC, rank_of(from));
        const Square rook_to   = make_square(kingside ? FileF : FileD, rank_of(from));
        // Remove both pieces before placing either, so king and rook can swap
        // or pass through each other's squares (valid Chess960 configurations).
        remove_piece(from);
        remove_piece(rook_from);
        put_piece(make_piece(us, King), king_to);
        put_piece(make_piece(us, Rook), rook_to);
    } else if (mt == Move::EnPassant) {
        // The captured pawn sits on the file of `to` and rank of `from`.
        const Square cap_sq = make_square(file_of(to), rank_of(from));
        remove_piece(cap_sq);
        move_piece(from, to);
    } else if (mt == Move::Promotion) {
        // The pawn vacates, then we drop the promoted piece on the target.
        if (st.captured != NoPiece) remove_piece(to);
        remove_piece(from);
        put_piece(make_piece(us, m.promotion_piece()), to);
    } else {
        // Normal move.
        if (st.captured != NoPiece) remove_piece(to);
        move_piece(from, to);
        // Pawn double push - the ep square sits between from and to.
        if (moving_pt == Pawn && std::abs(static_cast<int>(to) - static_cast<int>(from)) == 16) {
            ep_ = Square((static_cast<int>(from) + static_cast<int>(to)) / 2);
            key_ ^= en_passant_file[file_of(ep_)];
        }
    }

    // ---- NNUE accumulator: incremental update ----
    // HalfKP feature indices are king-relative, so any move that shifts a
    // king (explicit king moves and castling) invalidates every feature and
    // forces a full refresh. Everything else touches 2-3 columns at most.
    //
    // The helpers read king_square() off `*this`, which is fine because the
    // king has NOT moved in any branch other than the full-refresh ones.
    // We skip work entirely when acc_.computed is false - either the network
    // is not loaded or evaluate() will lazy-refresh on first call.
    if (update_acc && acc_.computed) {
        const bool king_moved = (moving_pt == King) || (mt == Move::Castling);
        if (king_moved) {
            // Only the moving side's perspective is reindexed by the new king
            // square, so it needs a full rebuild. The other perspective is
            // indexed by the stationary opposite king — there, the only changed
            // features are the moved king (and the rook on castling, plus any
            // captured piece), which we patch in incrementally. Halves the
            // king-move FT cost vs refreshing both sides.
            const int moved = (us == White) ? 0 : 1;
            const int other = moved ^ 1;
            nnue::refresh_perspective(*this, acc_, moved);
            if (mt == Move::Castling) {
                // `to` is the rook's initial square (KxR encoding).
                const Square rook_from = to;
                const bool   kingside  = (file_of(rook_from) > file_of(from));
                const Square king_to   = make_square(kingside ? FileG : FileC, rank_of(from));
                const Square rook_to   = make_square(kingside ? FileF : FileD, rank_of(from));
                nnue::remove_piece_one(acc_, *this, other, us, King, from);
                nnue::add_piece_one   (acc_, *this, other, us, King, king_to);
                nnue::remove_piece_one(acc_, *this, other, us, Rook, rook_from);
                nnue::add_piece_one   (acc_, *this, other, us, Rook, rook_to);
            } else {
                if (st.captured != NoPiece) {
                    nnue::remove_piece_one(acc_, *this, other, color_of(st.captured),
                                           type_of(st.captured), to);
                }
                nnue::remove_piece_one(acc_, *this, other, us, King, from);
                nnue::add_piece_one   (acc_, *this, other, us, King, to);
            }
        } else if (mt == Move::EnPassant) {
            const Square cap_sq = make_square(file_of(to), rank_of(from));
            nnue::remove_piece(acc_, *this, them, Pawn,      cap_sq);
            nnue::remove_piece(acc_, *this, us,   Pawn,      from);
            nnue::add_piece   (acc_, *this, us,   Pawn,      to);
        } else if (mt == Move::Promotion) {
            if (st.captured != NoPiece) {
                nnue::remove_piece(acc_, *this, color_of(st.captured),
                                   type_of(st.captured), to);
            }
            nnue::remove_piece(acc_, *this, us, Pawn,                  from);
            nnue::add_piece   (acc_, *this, us, m.promotion_piece(),   to);
        } else {
            // Normal quiet or capture move.
            if (st.captured != NoPiece) {
                nnue::remove_piece(acc_, *this, color_of(st.captured),
                                   type_of(st.captured), to);
            }
            nnue::remove_piece(acc_, *this, us, moving_pt, from);
            nnue::add_piece   (acc_, *this, us, moving_pt, to);
        }
    }

    // Castling-rights revocation: any move out of or into one of the keyed
    // squares revokes the affected rights. Uses the per-position table built
    // from the actual initial king/rook squares so Chess960 positions work.
    castling_ &= ~(castling_revoke_[from] | castling_revoke_[to]);
    key_ ^= castling[castling_];

    stm_ = them;
    key_ ^= zobrist::side_to_move;

#ifndef NDEBUG
    // Debug-only invariant: the incrementally-updated accumulator must match
    // a from-scratch refresh of the post-move position bit-for-bit. Catches
    // missed king-move refreshes, perspective desyncs, and feature-index
    // miscalculations immediately at the source of the error.
    if (update_acc && acc_.computed && nnue::is_loaded()) {
        nnue::Accumulator scratch;
        nnue::refresh(*this, scratch);
        assert(scratch.computed);
        for (int p = 0; p < 2; ++p) {
            for (int i = 0; i < nnue::kFtOutSize; ++i) {
                assert(acc_.v[p][i] == scratch.v[p][i]);
            }
        }
    }
#endif
}

void Position::undo_move(Move m, const StateInfo& st, bool restore_acc) {
    const Square     from = m.from();
    const Square     to   = m.to();
    const Move::Type mt   = m.type();

    stm_ = ~stm_;
    const Color us = stm_;

    if (mt == Move::Castling) {
        // `to` is the rook's initial square (KxR encoding). Reverse the move:
        // remove king and rook from their post-castle squares, restore originals.
        const Square rook_from = to;
        const bool   kingside  = (file_of(rook_from) > file_of(from));
        const Square king_to   = make_square(kingside ? FileG : FileC, rank_of(from));
        const Square rook_to   = make_square(kingside ? FileF : FileD, rank_of(from));
        raw_remove_piece(king_to);
        raw_remove_piece(rook_to);
        raw_put_piece(make_piece(us, King), from);
        raw_put_piece(make_piece(us, Rook), rook_from);
    } else if (mt == Move::EnPassant) {
        raw_move_piece(to, from);
        const Square cap_sq = make_square(file_of(to), rank_of(from));
        raw_put_piece(make_piece(~us, Pawn), cap_sq);
    } else if (mt == Move::Promotion) {
        raw_remove_piece(to);
        raw_put_piece(make_piece(us, Pawn), from);
        if (st.captured != NoPiece) raw_put_piece(st.captured, to);
    } else {
        raw_move_piece(to, from);
        if (st.captured != NoPiece) raw_put_piece(st.captured, to);
    }

    castling_         = st.prev_castling;
    ep_               = st.prev_ep;
    halfmove_clock_   = st.prev_halfmove;
    key_              = st.prev_key;
    if (us == Black) --fullmove_number_;

    // Restore the pre-do_move accumulator snapshot. Skipped when the matching
    // do_move ran with update_acc=false (legality testing): the accumulator was
    // never touched, so there is nothing to roll back and st.accumulator holds
    // no valid snapshot. Otherwise unconditional: if the network was not loaded
    // the snapshot is just an uncomputed zero accumulator, which is the correct
    // state to roll back to.
    if (restore_acc) acc_ = st.accumulator;
}

void Position::do_null_move(StateInfo& st) {
    st.prev_castling = castling_;
    st.prev_ep       = ep_;
    st.prev_halfmove = halfmove_clock_;
    st.prev_key      = key_;
    st.captured      = NoPiece;
    st.accumulator   = acc_;

    if (ep_ != SquareNone) {
        key_ ^= zobrist::en_passant_file[file_of(ep_)];
        ep_ = SquareNone;
    }

    stm_ = ~stm_;
    key_ ^= zobrist::side_to_move;

    if (stm_ == White) ++fullmove_number_;
    ++halfmove_clock_;
}

void Position::undo_null_move(const StateInfo& st) {
    stm_ = ~stm_;
    if (stm_ == Black) --fullmove_number_;

    castling_         = st.prev_castling;
    ep_               = st.prev_ep;
    halfmove_clock_   = st.prev_halfmove;
    key_              = st.prev_key;
    acc_              = st.accumulator;
}

std::string Position::ascii_board() const {
    std::ostringstream oss;
    for (int r = 7; r >= 0; --r) {
        oss << (r + 1) << "  ";
        for (int f = 0; f < 8; ++f) {
            const Square s = make_square(File(f), Rank(r));
            oss << piece_to_char(board_[s]) << ' ';
        }
        oss << '\n';
    }
    oss << "   a b c d e f g h\n";
    return oss.str();
}

}  // namespace eclipse
