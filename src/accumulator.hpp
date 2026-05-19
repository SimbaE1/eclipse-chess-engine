// SPDX-License-Identifier: GPL-3.0-or-later
//
// Standalone Accumulator definition.
//
// Extracted from nnue.hpp so position.hpp can hold one without dragging in
// the full nnue/position/eval include cycle. The full NNUE API still lives
// in nnue.hpp, which re-exports the same type by including this header.

#pragma once

#include <array>
#include <cstdint>

namespace eclipse::nnue {

// Per-perspective hidden width of the feature transformer. Must stay in sync
// with the constant referenced from nnue.cpp + the .nnue file header; the
// loader cross-checks it at load time.
constexpr int kFtOutSize = 512;

// One accumulator per perspective. The two perspectives share weights but are
// indexed differently (king-relative + own/opp color), so we keep them as
// separate int16[kFtOutSize] arrays.
//
// alignas(64) so SIMD loads (32B for AVX2, 16B for NEON) stay aligned even
// when this struct is embedded inside StateInfo or Position.
struct alignas(64) Accumulator {
    std::array<std::int16_t, kFtOutSize> v[2]{};   // [perspective]
    bool computed = false;                         // false = needs full refresh
};

}  // namespace eclipse::nnue
