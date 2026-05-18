// SPDX-License-Identifier: GPL-3.0-or-later
//
// HalfKP-256x2-32-32 NNUE evaluation.
//
// Architecture (Stockfish-classic shape):
//   Input features: HalfKP - king-relative, perspective-doubled.
//     Per-perspective: 64 king-sq * 64 piece-sq * 5 piece-types (P,N,B,R,Q) * 2 colors = 40960
//     King is implicit (defines the perspective), not a feature itself.
//   Feature transformer: 40960 -> 256, int16 weights + bias.
//   Concatenation: [accumulator_us, accumulator_them] -> 512
//   Activation: clipped ReLU [0, kFtQuant=127] -> uint8[512]
//   L1: 512 -> 32, int8 weights, int32 bias.
//   L2:  32 -> 32, int8 weights, int32 bias.
//   L3:  32 -> 1,  int8 weights, int32 bias -> centipawns.
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

#include "eval.hpp"
#include "position.hpp"
#include "types.hpp"

namespace eclipse::nnue {

// ---- Architecture constants ------------------------------------------------

constexpr int kFtOutSize        = 512;          // per perspective
constexpr int kL1InSize         = 2 * kFtOutSize; // = 512
constexpr int kL1OutSize        = 32;
constexpr int kL2OutSize        = 32;
constexpr int kL3OutSize        = 1;

constexpr int kFtKingSquares    = 64;
constexpr int kFtPieceSquares   = 64;
// 5 = {P, N, B, R, Q}, no king as a feature.
constexpr int kFtPieceTypes     = 5;
constexpr int kFtPieceColors    = 2;
constexpr int kFtNumFeatures    =
    kFtKingSquares * kFtPieceSquares * kFtPieceTypes * kFtPieceColors;  // = 40960

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

// One accumulator per perspective. The two perspectives share weights but are
// indexed differently (king-relative + own/opp color), so we keep them as
// separate int16[256] arrays.
//
// alignas(64) so SIMD loads (32B for AVX2, 16B for NEON) stay aligned even when
// the struct is embedded inside StateInfo in Phase 2.
struct alignas(64) Accumulator {
    std::array<std::int16_t, kFtOutSize> v[2]{};   // [perspective]
    bool computed = false;                          // false = needs full refresh
};

// HalfKP feature index for a single piece-on-square, relative to the king
// square of the given perspective. Returns a value in [0, kFtNumFeatures).
//
// `perspective`     - which side is "us" (the king whose square indexes features)
// `king_sq`         - king square for the perspective side (already mirrored
//                     for Black inside the caller)
// `piece_sq`        - square of the piece being indexed (mirrored for Black)
// `pt`              - piece type (must be in {Pawn..Queen})
// `piece_is_ours`   - true if the piece belongs to the perspective side
//
// The mirroring convention matches Stockfish HalfKP: for Black perspective,
// every square is reflected vertically (sq ^ 56) so the network always sees
// the position from its own side of the board.
int feature_index(Square king_sq, Square piece_sq,
                  PieceType pt, bool piece_is_ours) noexcept;

// ---- Public API ------------------------------------------------------------

// Load a HalfKP .nnue file. Returns false on bad magic, dimension mismatch, or
// short read. Prints a diagnostic to stderr in all error paths.
bool load(const std::string& path);

// True after a successful load. evaluate() must not be called otherwise.
bool is_loaded() noexcept;

// Recompute `acc` from scratch for the given position (full refresh). Both
// perspectives are filled. Used on initial position setup, when crossing a
// king move (HalfKP requires refresh on king moves), and any time the engine
// loses track of the incremental state.
void refresh(const Position& pos, Accumulator& acc) noexcept;

// Forward pass: returns the score in centipawns from the side-to-move's
// perspective, matching eclipse::evaluate's contract.
//
// Phase 1: ignores any external accumulator and recomputes from scratch.
// Phase 2: will take the up-to-date accumulator from the current StateInfo.
Score evaluate(const Position& pos) noexcept;

}  // namespace eclipse::nnue
