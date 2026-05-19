// SPDX-License-Identifier: GPL-3.0-or-later
#include "nnue.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "bitboard.hpp"

namespace eclipse::nnue {

namespace {

// File format - bumped from the float32 tiny NNUE.
//
// Layout (little-endian, packed):
//   uint32  magic            (kMagic)
//   uint32  version          (kVersion)
//   uint32  ft_in_features   (must equal kFtNumFeatures)
//   uint32  ft_out           (must equal kFtOutSize, per perspective)
//   uint32  l1_out           (must equal kL1OutSize)
//   uint32  l2_out           (must equal kL2OutSize)
//   uint32  l3_out           (must equal kL3OutSize)
//   float   output_cp_per_unit   - centipawns per real-unit of L3 output;
//                                  see kOutputCpDivisor explanation below
//   int16   ft_b[ft_out]
//   int16   ft_w[ft_in_features * ft_out]    (row-major: [feature, hidden])
//   int32   l1_b[l1_out]
//   int8    l1_w[l1_out * (2 * ft_out)]      (row-major: [out, in])
//   int32   l2_b[l2_out]
//   int8    l2_w[l2_out * l1_out]
//   int32   l3_b[l3_out]
//   int8    l3_w[l3_out * l2_out]
//
// Total size for the spec dims: ~20.97 MB (FT weights dominate at 40960*256*2).
constexpr std::uint32_t kMagic   = 0xECCC0002;
constexpr std::uint32_t kVersion = 1;

// L3 produces `out ~= y_real * kWeightScale * kFtQuant` because we don't shift
// after the final layer (we want full precision into the centipawn cast). That
// product is the divisor when converting the int32 output to centipawns.
constexpr std::int32_t kOutputCpDivisor = kWeightScale * kFtQuant;  // 64 * 127 = 8128

bool                                          g_loaded = false;
float                                         g_output_cp_per_unit = 410.0f;

// Feature-transformer weights stored row-major as [feature, hidden] so the
// per-feature "column" we add to the accumulator is contiguous - same pattern
// the tiny NNUE used, and the layout SIMD wants.
std::vector<std::int16_t>                     g_ft_w;
std::array<std::int16_t, kFtOutSize>          g_ft_b{};

// All int8 layers are stored row-major as [out, in] so each output neuron's
// weights are contiguous.
std::vector<std::int8_t>                      g_l1_w;
std::array<std::int32_t, kL1OutSize>          g_l1_b{};
std::vector<std::int8_t>                      g_l2_w;
std::array<std::int32_t, kL2OutSize>          g_l2_b{};
std::vector<std::int8_t>                      g_l3_w;
std::array<std::int32_t, kL3OutSize>          g_l3_b{};

// --- I/O helpers ----------------------------------------------------------

template <typename T>
bool read_pod(std::ifstream& f, T& out) {
    f.read(reinterpret_cast<char*>(&out), sizeof(T));
    return static_cast<bool>(f);
}

template <typename T>
bool read_array(std::ifstream& f, T* dst, std::size_t n) {
    f.read(reinterpret_cast<char*>(dst),
           static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<bool>(f);
}

// Reset all global weights to "not loaded" state. Called on any load failure so
// a partial read can't leave the engine running on garbage.
void reset_state() noexcept {
    g_loaded = false;
    g_output_cp_per_unit = 410.0f;
    g_ft_w.clear();
    g_ft_w.shrink_to_fit();
    g_ft_b.fill(0);
    g_l1_w.clear();
    g_l1_b.fill(0);
    g_l2_w.clear();
    g_l2_b.fill(0);
    g_l3_w.clear();
    g_l3_b.fill(0);
}

}  // namespace

// ---- Feature indexing -----------------------------------------------------

int feature_index(Square king_sq, Square piece_sq,
                  PieceType pt, bool piece_is_ours) noexcept {
    // HalfKP order: king * (64*5*2) + piece_sq * (5*2) + (pt-1)*2 + own/opp
    // Strides match the "innermost = color, middle = piece type, outer = piece
    // square, outermost = king square" layout, which keeps every per-king block
    // contiguous and ~5 KB - good cache behaviour during refresh.
    const int p_idx = static_cast<int>(pt) - 1;  // Pawn=1 -> 0, ..., Queen=5 -> 4
    return static_cast<int>(king_sq) * (kFtPieceSquares * kFtPieceTypes * kFtPieceColors)
         + static_cast<int>(piece_sq) * (kFtPieceTypes * kFtPieceColors)
         + p_idx * kFtPieceColors
         + (piece_is_ours ? 0 : 1);
}

// ---- Loader ---------------------------------------------------------------

bool is_loaded() noexcept { return g_loaded; }

bool load(const std::string& path) {
    reset_state();

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "info string NNUE load failed: cannot open '%s'\n",
                     path.c_str());
        return false;
    }

    std::uint32_t magic = 0, version = 0;
    std::uint32_t ft_in = 0, ft_out = 0, l1_out = 0, l2_out = 0, l3_out = 0;
    if (!read_pod(f, magic) || magic != kMagic) {
        std::fprintf(stderr,
                     "info string NNUE load failed: bad magic 0x%08X (expected 0x%08X)\n",
                     magic, kMagic);
        return false;
    }
    if (!read_pod(f, version) || version != kVersion) {
        std::fprintf(stderr,
                     "info string NNUE load failed: bad version %u (expected %u)\n",
                     version, kVersion);
        return false;
    }
    if (!read_pod(f, ft_in)  || !read_pod(f, ft_out) ||
        !read_pod(f, l1_out) || !read_pod(f, l2_out) || !read_pod(f, l3_out)) {
        std::fprintf(stderr, "info string NNUE load failed: short read on header\n");
        return false;
    }
    if (ft_in  != static_cast<std::uint32_t>(kFtNumFeatures) ||
        ft_out != static_cast<std::uint32_t>(kFtOutSize)     ||
        l1_out != static_cast<std::uint32_t>(kL1OutSize)     ||
        l2_out != static_cast<std::uint32_t>(kL2OutSize)     ||
        l3_out != static_cast<std::uint32_t>(kL3OutSize)) {
        std::fprintf(stderr,
                     "info string NNUE load failed: arch mismatch "
                     "(file %u-%u-%u-%u-%u, expected %d-%d-%d-%d-%d)\n",
                     ft_in, ft_out, l1_out, l2_out, l3_out,
                     kFtNumFeatures, kFtOutSize, kL1OutSize, kL2OutSize, kL3OutSize);
        return false;
    }

    if (!read_pod(f, g_output_cp_per_unit)) {
        std::fprintf(stderr, "info string NNUE load failed: short read on output scale\n");
        return false;
    }

    // FT weights and biases.
    if (!read_array(f, g_ft_b.data(), kFtOutSize)) {
        std::fprintf(stderr, "info string NNUE load failed: short read on ft_b\n");
        return false;
    }
    g_ft_w.assign(static_cast<std::size_t>(kFtNumFeatures) * kFtOutSize, 0);
    if (!read_array(f, g_ft_w.data(), g_ft_w.size())) {
        std::fprintf(stderr, "info string NNUE load failed: short read on ft_w\n");
        return false;
    }

    // L1
    if (!read_array(f, g_l1_b.data(), kL1OutSize)) {
        std::fprintf(stderr, "info string NNUE load failed: short read on l1_b\n");
        return false;
    }
    g_l1_w.assign(static_cast<std::size_t>(kL1OutSize) * kL1InSize, 0);
    if (!read_array(f, g_l1_w.data(), g_l1_w.size())) {
        std::fprintf(stderr, "info string NNUE load failed: short read on l1_w\n");
        return false;
    }

    // L2
    if (!read_array(f, g_l2_b.data(), kL2OutSize)) {
        std::fprintf(stderr, "info string NNUE load failed: short read on l2_b\n");
        return false;
    }
    g_l2_w.assign(static_cast<std::size_t>(kL2OutSize) * kL1OutSize, 0);
    if (!read_array(f, g_l2_w.data(), g_l2_w.size())) {
        std::fprintf(stderr, "info string NNUE load failed: short read on l2_w\n");
        return false;
    }

    // L3
    if (!read_array(f, g_l3_b.data(), kL3OutSize)) {
        std::fprintf(stderr, "info string NNUE load failed: short read on l3_b\n");
        return false;
    }
    g_l3_w.assign(static_cast<std::size_t>(kL3OutSize) * kL2OutSize, 0);
    if (!read_array(f, g_l3_w.data(), g_l3_w.size())) {
        std::fprintf(stderr, "info string NNUE load failed: short read on l3_w\n");
        return false;
    }

    g_loaded = true;
    std::cout << "info string NNUE loaded: " << path
              << " (HalfKP-" << kFtOutSize << "x2-" << kL1OutSize << '-' << kL2OutSize
              << ", output_cp/unit=" << g_output_cp_per_unit << ")" << std::endl;
    return true;
}

// ---- Incremental piece updates --------------------------------------------

namespace {

// Add or subtract one piece's FT column on both perspective accumulators.
// `Sign` is +1 for add, -1 for remove. Both perspectives must be touched in
// lockstep - they share weights but are indexed differently, so updating only
// one would silently desync the other.
//
// King squares are read from `pos`, so the caller must ensure the kings have
// not moved relative to the accumulator state being updated (king moves
// invalidate every feature index and require a full refresh instead).
template <int Sign>
void apply_piece(Accumulator& acc, const Position& pos,
                 Color c, PieceType pt, Square sq) noexcept {
    static_assert(Sign == 1 || Sign == -1, "Sign must be +1 or -1");
    if (!g_loaded) return;
    if (pt == King) return;  // King is implicit (defines perspective), no FT column.

    for (std::size_t persp = 0; persp < 2; ++persp) {
        const Color persp_color = (persp == 0) ? White : Black;
        Square king_sq  = pos.king_square(persp_color);
        Square piece_sq = sq;
        // HalfKP mirror convention: Black perspective sees everything flipped
        // so the network always reasons in own-side-of-board coordinates.
        if (persp_color == Black) {
            king_sq  = flip_rank(king_sq);
            piece_sq = flip_rank(piece_sq);
        }
        const bool is_ours = (c == persp_color);
        const int  idx     = feature_index(king_sq, piece_sq, pt, is_ours);
        const std::int16_t* col =
            &g_ft_w[static_cast<std::size_t>(idx) * kFtOutSize];
        for (std::size_t i = 0; i < kFtOutSize; ++i) {
            acc.v[persp][i] = static_cast<std::int16_t>(
                acc.v[persp][i] + Sign * col[i]);
        }
    }
}

}  // namespace

void add_piece(Accumulator& acc, const Position& pos,
               Color c, PieceType pt, Square sq) noexcept {
    apply_piece<+1>(acc, pos, c, pt, sq);
}

void remove_piece(Accumulator& acc, const Position& pos,
                  Color c, PieceType pt, Square sq) noexcept {
    apply_piece<-1>(acc, pos, c, pt, sq);
}

// ---- Refresh --------------------------------------------------------------

void refresh(const Position& pos, Accumulator& acc) noexcept {
    if (!g_loaded) {
        acc.computed = false;
        return;
    }

    // Fill both perspectives. Each one runs over the entire occupancy (minus
    // kings) and accumulates the king-relative feature column for every piece.
    for (std::size_t persp = 0; persp < 2; ++persp) {
        const Color persp_color = (persp == 0) ? White : Black;
        Square king_sq = pos.king_square(persp_color);
        // HalfKP convention: mirror everything to the Black-side frame when the
        // perspective is Black, so the network always sees pawns advancing up.
        if (persp_color == Black) king_sq = flip_rank(king_sq);

        // Start from the FT bias.
        std::memcpy(acc.v[persp].data(), g_ft_b.data(),
                    kFtOutSize * sizeof(std::int16_t));

        Bitboard occ = pos.occupied();
        while (occ) {
            const Square    s        = pop_lsb(occ);
            const Piece     p        = pos.piece_on(s);
            const PieceType pt       = type_of(p);
            if (pt == King) continue;  // King is implicit (defines perspective)

            const Color  piece_color = color_of(p);
            Square       piece_sq    = s;
            if (persp_color == Black) piece_sq = flip_rank(piece_sq);
            const bool   is_ours     = (piece_color == persp_color);

            const int idx = feature_index(king_sq, piece_sq, pt, is_ours);
            const std::int16_t* col =
                &g_ft_w[static_cast<std::size_t>(idx) * kFtOutSize];
            for (std::size_t i = 0; i < kFtOutSize; ++i) {
                acc.v[persp][i] = static_cast<std::int16_t>(acc.v[persp][i] + col[i]);
            }
        }
    }

    acc.computed = true;
}

// ---- Inference ------------------------------------------------------------

namespace {

// L1/L2 dense int8 dot product with int32 bias, post-shift by kWeightShift,
// clamp into [0, kFtQuant] for the next layer's uint8 input.
template <int kIn, int kOut>
void affine_clipped_relu(const std::uint8_t* in,
                         const std::int8_t*  w,        // [kOut, kIn]
                         const std::int32_t* b,        // [kOut]
                         std::uint8_t*       out) noexcept {
    for (int o = 0; o < kOut; ++o) {
        std::int32_t sum = b[o];
        const std::int8_t* wo = w + o * kIn;
        for (int i = 0; i < kIn; ++i) {
            sum += static_cast<std::int32_t>(wo[i]) * static_cast<std::int32_t>(in[i]);
        }
        sum >>= kWeightShift;
        if (sum < 0)         sum = 0;
        if (sum > kFtQuant)  sum = kFtQuant;
        out[o] = static_cast<std::uint8_t>(sum);
    }
}

}  // namespace

Score evaluate(const Position& pos) noexcept {
    // Phase 2: read the incrementally-maintained accumulator off the Position.
    // do_move keeps it in sync; we only refresh on the cold path (network
    // loaded after set_from_fen, or someone forgot to mark it computed).
    Accumulator& acc = pos.accumulator();
    if (!acc.computed) refresh(pos, acc);
    if (!acc.computed) {
        // refresh() left it false -> the network is not loaded. Fall back.
        return material_evaluate(pos);
    }

    const Color       stm  = pos.side_to_move();
    const std::size_t us   = (stm == White) ? 0u : 1u;
    const std::size_t them = us ^ 1u;

    // FT concat + clipped ReLU: [acc_us, acc_them] -> uint8[512]
    alignas(64) std::uint8_t ft_out[kL1InSize];
    for (std::size_t i = 0; i < kFtOutSize; ++i) {
        const std::int16_t a = acc.v[us][i];
        const std::int16_t b = acc.v[them][i];
        ft_out[i]              = static_cast<std::uint8_t>(std::clamp<std::int32_t>(a, 0, kFtQuant));
        ft_out[i + kFtOutSize] = static_cast<std::uint8_t>(std::clamp<std::int32_t>(b, 0, kFtQuant));
    }

    alignas(64) std::uint8_t l1_out[kL1OutSize];
    alignas(64) std::uint8_t l2_out[kL2OutSize];

    affine_clipped_relu<kL1InSize, kL1OutSize>(
        ft_out, g_l1_w.data(), g_l1_b.data(), l1_out);
    affine_clipped_relu<kL1OutSize, kL2OutSize>(
        l1_out, g_l2_w.data(), g_l2_b.data(), l2_out);

    // L3: int8 dot product, NO shift, NO clamp - we want full int32 precision
    // for the centipawn conversion.
    std::int32_t y = g_l3_b[0];
    const std::int8_t* w3 = g_l3_w.data();
    for (int i = 0; i < kL2OutSize; ++i) {
        y += static_cast<std::int32_t>(w3[i]) * static_cast<std::int32_t>(l2_out[i]);
    }

    // y ~= kWeightScale * kFtQuant * y_real (because no post-shift was applied)
    // cp = y_real * output_cp_per_unit = y * output_cp_per_unit / kOutputCpDivisor
    const float cp = static_cast<float>(y) * g_output_cp_per_unit
                   / static_cast<float>(kOutputCpDivisor);
    return static_cast<Score>(cp);
}

}  // namespace eclipse::nnue
