// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "accumulator.hpp"
#include "bitboard.hpp"
#include "move.hpp"
#include "types.hpp"

namespace eclipse {

// Information that do_move() saves and undo_move() restores. The caller owns
// the StateInfo and is responsible for keeping the same instance through the
// matched do/undo pair. This mirrors Stockfish's approach and avoids embedding
// a history stack inside Position itself.
struct StateInfo {
    CastlingRights    prev_castling   = NoCastling;
    Square            prev_ep         = SquareNone;
    int               prev_halfmove   = 0;
    std::uint64_t     prev_key        = 0;
    Piece             captured        = NoPiece;
    // Pre-do_move accumulator snapshot, used by undo_move to restore the
    // Position's cached accumulator without a full refresh.
    nnue::Accumulator accumulator{};
};

class Position {
public:
    Position();

    // Replace this position's state with the FEN-described one. Throws nothing
    // on malformed input; instead returns false and leaves the position in an
    // unspecified but valid state. Use the standard FEN field order:
    //   <pieces> <stm> <castling> <ep> <halfmove> <fullmove>
    bool set_from_fen(std::string_view fen);
    std::string fen() const;

    static Position startpos();

    // ---- read-only accessors --------------------------------------------------
    Piece     piece_on    (Square s) const noexcept { return board_[s]; }
    Bitboard  occupied    ()        const noexcept { return by_color_[White] | by_color_[Black]; }
    Bitboard  pieces      (Color c) const noexcept { return by_color_[c]; }
    Bitboard  pieces      (PieceType pt) const noexcept {
        return by_piece_[make_piece(White, pt)] | by_piece_[make_piece(Black, pt)];
    }
    Bitboard  pieces      (Color c, PieceType pt) const noexcept {
        return by_piece_[make_piece(c, pt)];
    }

    Color           side_to_move    () const noexcept { return stm_; }
    Square          ep_square       () const noexcept { return ep_; }
    CastlingRights  castling_rights () const noexcept { return castling_; }
    int             halfmove_clock  () const noexcept { return halfmove_clock_; }
    int             fullmove_number () const noexcept { return fullmove_number_; }
    std::uint64_t   key             () const noexcept { return key_; }

    Square king_square(Color c) const noexcept {
        return lsb(by_piece_[make_piece(c, King)]);
    }

    // Up-to-date NNUE accumulator for this position. Maintained incrementally
    // by do_move / undo_move; falls back to a refresh on first access if the
    // network was loaded after the position was last set. Marked `mutable` /
    // returned by non-const reference so evaluate() (which takes
    // `const Position&`) can perform the lazy refresh.
    nnue::Accumulator& accumulator() const noexcept { return acc_; }

    // ---- queries --------------------------------------------------------------
    // True iff the given square is attacked by any piece of color `by`.
    bool is_square_attacked(Square s, Color by) const noexcept;

    // Returns a bitboard of all pieces (both colors) that attack square `s` given
    // the occupancy `occ`.
    Bitboard attackers_to(Square s, Bitboard occ) const noexcept;

    // True iff the side to move's king is in check.
    bool in_check() const noexcept {
        return is_square_attacked(king_square(stm_), ~stm_);
    }

    // ---- mutators -------------------------------------------------------------
    void do_move  (Move m, StateInfo& st);
    void undo_move(Move m, const StateInfo& st);
    void do_null_move  (StateInfo& st);
    void undo_null_move(const StateInfo& st);

    // ---- debug ----------------------------------------------------------------
    std::string ascii_board() const;

private:
    Bitboard by_piece_[PieceNB] = {};
    Bitboard by_color_[ColorNB] = {};
    Piece    board_[SquareNB]   = {};

    Color           stm_              = White;
    CastlingRights  castling_         = NoCastling;
    Square          ep_               = SquareNone;
    int             halfmove_clock_   = 0;
    int             fullmove_number_  = 1;
    std::uint64_t   key_              = 0;

    // Incrementally-maintained NNUE accumulator. Refreshed in set_from_fen,
    // updated in do_move, restored from StateInfo in undo_move. `mutable` so
    // evaluate() can lazy-refresh through the const accessor when the network
    // was loaded after the position was last set.
    mutable nnue::Accumulator acc_{};

    // Hash-updating piece moves used by do_move().
    void put_piece   (Piece p, Square s);
    void remove_piece(Square s);
    void move_piece  (Square from, Square to);

    // Non-hash-updating variants used by undo_move(), which restores the key
    // wholesale from the saved StateInfo.
    void raw_put_piece   (Piece p, Square s) noexcept;
    void raw_remove_piece(Square s) noexcept;
    void raw_move_piece  (Square from, Square to) noexcept;

    void clear();
    std::uint64_t compute_key() const noexcept;
};

}  // namespace eclipse
