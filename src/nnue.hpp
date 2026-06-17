// SPDX-License-Identifier: GPL-3.0-or-later
//
// HalfKAv2-1024x2-512-128-1 NNUE evaluation.
//
// Architecture (Stockfish-modern shape):
//   Input features: HalfKAv2 - king-relative, perspective-doubled.
//     Per-perspective: 64 king-sq * 64 piece-sq * 11 piece-type slots = 45056
//       Slots 0..4 : own (P,N,B,R,Q). Own king is NOT a feature -- it always
//                    sits at the indexing king square so the feature would be
//                    constant. Skipping it saves ~9% of the FT table size and
//                    matches Stockfish's HalfKAv2.
//       Slots 5..10: opp (P,N,B,R,Q,K). Opp king IS a feature, which gives
//                    the network direct information about the opposing king's
//                    distance / mating geometry that plain HalfKP lacked.
//   Feature transformer: 45056 -> 1024, int16 weights + bias.
//   Concatenation: [accumulator_us, accumulator_them] -> 2048
//   Activation: clipped ReLU [0, kFtQuant=127] -> uint8[2048]
//   L1: 2048 -> 512, int8 weights, int32 bias.
//   L2:  512 -> 128, int8 weights, int32 bias.
//   L3:  128 -> 1,   int8 weights, int32 bias -> value logit.
//
// Phase 1: scalar forward pass, non-incremental (recompute accumulator per eval).
// Phase 2: incremental updates wired through StateInfo / do_move / undo_move.
// Phase 3: AVX2 + NEON SIMD inference.
//
// The Accumulator type is exposed in the header so Phase 2 can embed it in
// StateInfo without an API break.

#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "accumulator.hpp"
#include "eval.hpp"
#include "position.hpp"
#include "types.hpp"

namespace eclipse::nnue {

// ---- Architecture constants ------------------------------------------------
// kFtOutSize is defined in accumulator.hpp.

constexpr int kL1InSize         = 2 * kFtOutSize; // = 2048
constexpr int kL1OutSize        = 512;
constexpr int kL2OutSize        = 128;
constexpr int kL3OutSize        = 1;

constexpr int kFtKingSquares    = 64;
constexpr int kFtPieceSquares   = 64;
// HalfKAv2 piece-type slots per (king_sq, piece_sq) cell:
//   0..4 : own  (P, N, B, R, Q)          -- 5 slots; own king is implicit
//   5..10: theirs (P, N, B, R, Q, K)     -- 6 slots; opp king IS a feature
constexpr int kFtPieceTypeSlots = 11;
constexpr int kFtNumFeatures    =
    kFtKingSquares * kFtPieceSquares * kFtPieceTypeSlots;  // = 45056

// ---- Quantization constants ------------------------------------------------
//
// All values in this header are referenced from the loader, the inference path,
// and the Python conversion script - keeping them here means the three layers
// can never silently disagree about scale.

// Feature-transformer accumulator and activation scale. After concatenating the
// two int16 accumulators we clamp into [0, kFtQuant], which is the saturating
// upper bound for the uint8 cast that feeds L1. 127 is the classic Stockfish
// choice (leaves one bit of headroom on int8 dot products).
constexpr std::int32_t kFtQuant         = 127;

// int8 weight scale: stored_weight = round(real_weight * kWeightScale).
// 64 = 2^6, so the post-accumulation right-shift by 6 strips it back out
// cheaply.
constexpr std::int32_t kWeightScale     = 64;
constexpr int          kWeightShift     = 6;     // log2(kWeightScale)

// Final-layer output -> centipawn conversion. 16 is the Stockfish convention:
// it places a typical "+1 pawn" position at ~16*64 = 1024 in raw int32 output,
// which is large enough to keep quantization noise irrelevant while still
// fitting comfortably in int32 after the L3 dot product.
constexpr std::int32_t kOutputScale     = 16;

// ---- Public types ----------------------------------------------------------
// Accumulator is defined in accumulator.hpp (so position.hpp can hold one
// without circular include).

// HalfKAv2 feature index for a single piece-on-square, relative to the king
// square of the given perspective. Returns a value in [0, kFtNumFeatures).
//
// `perspective`     - which side is "us" (the king whose square indexes features)
// `king_sq`         - king square for the perspective side (already mirrored
//                     for Black inside the caller)
// `piece_sq`        - square of the piece being indexed (mirrored for Black)
// `pt`              - piece type. Can be {Pawn..King} EXCEPT the own king,
//                     which the caller must skip (own king is implicit at
//                     the indexing king square).
// `piece_is_ours`   - true if the piece belongs to the perspective side
//
// The mirroring convention matches Stockfish: for Black perspective every
// square is reflected vertically (sq ^ 56) so the network always sees the
// position from its own side of the board.
int feature_index(Square king_sq, Square piece_sq,
                  PieceType pt, bool piece_is_ours) noexcept;

// ---- Public API ------------------------------------------------------------

// Load a HalfKP .nnue file. Returns false on bad magic, dimension mismatch, or
// short read. Prints a diagnostic to stderr in all error paths.
bool load(const std::string& path);

// True after a successful load. evaluate() must not be called otherwise.
bool is_loaded() noexcept;

// Centipawns per raw output unit, read from the .nnue file header.
// MCTS uses this to convert cp scores to WDL via tanh(cp / output_cp_per_unit())
// so the conversion is invariant to which cp_scale the net was trained with.
float output_cp_per_unit() noexcept;

// Recompute `acc` from scratch for the given position (full refresh). Both
// perspectives are filled. Used on initial position setup, when crossing a
// king move (HalfKP requires refresh on king moves), and any time the engine
// loses track of the incremental state.
void refresh(const Position& pos, Accumulator& acc) noexcept;

// Rebuild only one perspective (persp = 0 White, 1 Black) from scratch. Used
// for king moves: the moving side's perspective is reindexed by its new king
// square (full rebuild), while the other perspective — indexed by the
// stationary opposite king — only needs the moved king/rook columns patched in
// incrementally via add_piece_one / remove_piece_one. Does NOT set
// acc.computed (the caller manages that, since the other perspective is patched
// separately). Halves the per-king-move FT cost vs a full both-sides refresh().
void refresh_perspective(const Position& pos, Accumulator& acc, int persp) noexcept;

// Add the FT column for one piece (color, pt, sq) to `acc` on both perspectives.
// Caller must guarantee that the king squares in `pos` are the post-move ones
// when (color/pt/sq) describes a post-move piece; equivalently, the kings did
// NOT move on this update (king moves require a full refresh).
void add_piece(Accumulator& acc, const Position& pos,
               Color c, PieceType pt, Square sq) noexcept;

// Symmetric subtract: remove the FT column for (color, pt, sq) from `acc`.
// Same king-invariance constraint as add_piece.
void remove_piece(Accumulator& acc, const Position& pos,
                  Color c, PieceType pt, Square sq) noexcept;

// Single-perspective variants of add_piece / remove_piece (persp = 0 White,
// 1 Black). Used by the king-move path to patch the non-moving perspective
// while the moving one is rebuilt by refresh_perspective. Same king-invariance
// constraint as add_piece: the indexing king (the perspective side's) must not
// have moved relative to the accumulator state.
void add_piece_one(Accumulator& acc, const Position& pos, int persp,
                   Color c, PieceType pt, Square sq) noexcept;
void remove_piece_one(Accumulator& acc, const Position& pos, int persp,
                      Color c, PieceType pt, Square sq) noexcept;

// Forward pass: returns the score in centipawns from the side-to-move's
// perspective, matching eclipse::evaluate's contract.
Score evaluate(const Position& pos) noexcept;

// Batched forward pass: computes scores for multiple positions at once.
// Takes accumulators and side-to-move for each position in the batch.
void evaluate_batch(const Accumulator* accs, const Color* stms, Score* scores, int batch_size) noexcept;

}  // namespace eclipse::nnue
