// SPDX-License-Identifier: GPL-3.0-or-later
#include "nnue.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

// Phase 3 SIMD inference. The scalar path below is preserved as a fallback;
// the SIMD path is enabled when the compiler advertises AVX2 (x86) or NEON
// (ARM). With -march=native this lights up automatically on the build host.
#if defined(__AVX2__)
#  include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif
#include <vector>

#include "bitboard.hpp"

namespace eclipse::nnue {

namespace {

// File format.
//
// Layout (little-endian, packed):
//   uint32  magic            (kMagic; bumped to 0xECCC0003 for HalfKAv2)
//   uint32  version          (kVersion)
//   uint32  ft_in_features   (must equal kFtNumFeatures = 45056)
//   uint32  ft_out           (must equal kFtOutSize = 1024, per perspective)
//   uint32  l1_out           (must equal kL1OutSize = 128)
//   uint32  l2_out           (must equal kL2OutSize = 32)
//   uint32  l3_out           (must equal kL3OutSize = 1)
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
// Total size for the spec dims: ~92.5 MB (FT weights dominate at
// 45056*1024*2 = 88.0 MB).
// Magic bumped when the feature set switched from HalfKP (40960 features,
// 5 piece types * 2 colors) to HalfKAv2 (45056 features, 11 piece-type
// slots including opp king). Old eclipse.nnue files now fail to load with
// a clear "bad magic" message instead of silently producing garbage evals.
constexpr std::uint32_t kMagic   = 0xECCC0003;
constexpr std::uint32_t kVersion = 1;

// L3 produces `out ~= y_real * kWeightScale * kFtQuant` because we don't shift
// after the final layer (we want full precision into the centipawn cast). That
// product is the divisor when converting the int32 output to centipawns.
constexpr std::int32_t kOutputCpDivisor = kWeightScale * kFtQuant;  // 64 * 127 = 8128

bool                                          g_loaded = false;
float                                         g_output_cp_per_unit = 410.0f;

// Feature-transformer weights stored row-major as [feature, hidden] so the
// per-feature "column" we add to the accumulator is contiguous.
// We use a custom alignment-aware allocation here so SIMD loads from weights
// are always 64-byte aligned.
std::int16_t*                                 g_ft_w = nullptr;
std::size_t                                   g_ft_w_size = 0;
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
    if (g_ft_w) {
#if defined(_MSC_VER)
        _aligned_free(g_ft_w);
#else
        free(g_ft_w);
#endif
        g_ft_w = nullptr;
    }
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
    // HalfKAv2 slot layout within each (king_sq, piece_sq) cell:
    //   0..4  : own  (P,N,B,R,Q)        -- own king must NOT be passed here
    //   5..10 : opp  (P,N,B,R,Q,K)
    //
    // Stride order is "innermost = type slot, middle = piece square, outer =
    // king square". That keeps every per-king block (64 squares * 11 slots = 704
    // FT columns = ~1.4 KB at FT_OUT=1024) contiguous for refresh, and the
    // hot per-piece path only touches one cache line worth of weight indices.
    const int p_idx = static_cast<int>(pt) - 1;  // Pawn=1 -> 0, ..., King=6 -> 5
    const int slot  = piece_is_ours ? p_idx : (5 + p_idx);
    return static_cast<int>(king_sq) * (kFtPieceSquares * kFtPieceTypeSlots)
         + static_cast<int>(piece_sq) * kFtPieceTypeSlots
         + slot;
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
    
    g_ft_w_size = static_cast<std::size_t>(kFtNumFeatures) * kFtOutSize;
#if defined(_MSC_VER)
    g_ft_w = static_cast<std::int16_t*>(_aligned_malloc(g_ft_w_size * sizeof(std::int16_t), 64));
#else
    if (posix_memalign(reinterpret_cast<void**>(&g_ft_w), 64, g_ft_w_size * sizeof(std::int16_t)) != 0) {
        g_ft_w = nullptr;
    }
#endif

    if (!g_ft_w || !read_array(f, g_ft_w, g_ft_w_size)) {
        std::fprintf(stderr, "info string NNUE load failed: FT weights allocation or read failed\n");
        reset_state();
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
              << " (HalfKAv2-" << kFtOutSize << "x2-" << kL1OutSize << '-' << kL2OutSize
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

    for (std::size_t persp = 0; persp < 2; ++persp) {
        const Color persp_color = (persp == 0) ? White : Black;
        const bool  is_ours     = (c == persp_color);
        // HalfKAv2 skip: own king is implicit at the indexing king square,
        // so we don't have a column for it. The opp king IS a feature.
        if (is_ours && pt == King) continue;

        Square king_sq  = pos.king_square(persp_color);
        Square piece_sq = sq;
        // Mirror convention: Black perspective sees everything flipped so the
        // network always reasons in own-side-of-board coordinates.
        if (persp_color == Black) {
            king_sq  = flip_rank(king_sq);
            piece_sq = flip_rank(piece_sq);
        }
        const int  idx = feature_index(king_sq, piece_sq, pt, is_ours);
        const std::int16_t* col = &g_ft_w[static_cast<std::size_t>(idx) * kFtOutSize];

#if defined(__AVX2__)
        for (std::size_t i = 0; i < kFtOutSize; i += 16) {
            __m256i acc_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&acc.v[persp][i]));
            __m256i col_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&col[i]));
            if constexpr (Sign == 1) {
                acc_vec = _mm256_add_epi16(acc_vec, col_vec);
            } else {
                acc_vec = _mm256_sub_epi16(acc_vec, col_vec);
            }
            _mm256_store_si256(reinterpret_cast<__m256i*>(&acc.v[persp][i]), acc_vec);
        }
#elif defined(__ARM_NEON)
        for (std::size_t i = 0; i < kFtOutSize; i += 8) {
            int16x8_t acc_vec = vld1q_s16(&acc.v[persp][i]);
            int16x8_t col_vec = vld1q_s16(&col[i]);
            if constexpr (Sign == 1) {
                acc_vec = vaddq_s16(acc_vec, col_vec);
            } else {
                acc_vec = vsubq_s16(acc_vec, col_vec);
            }
            vst1q_s16(&acc.v[persp][i], acc_vec);
        }
#else
        for (std::size_t i = 0; i < kFtOutSize; ++i) {
            acc.v[persp][i] = static_cast<std::int16_t>(
                acc.v[persp][i] + Sign * col[i]);
        }
#endif
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

    // Fill both perspectives. Each one runs over the entire occupancy and
    // accumulates the king-relative feature column for every piece EXCEPT
    // the perspective side's own king (which is implicit). Opp king is a
    // normal feature.
    for (std::size_t persp = 0; persp < 2; ++persp) {
        const Color persp_color = (persp == 0) ? White : Black;
        Square king_sq = pos.king_square(persp_color);
        // Mirror everything to the Black-side frame when the perspective is
        // Black, so the network always sees pawns advancing up.
        if (persp_color == Black) king_sq = flip_rank(king_sq);

        // Start from the FT bias.
        std::memcpy(acc.v[persp].data(), g_ft_b.data(),
                    kFtOutSize * sizeof(std::int16_t));

        Bitboard occ = pos.occupied();
        while (occ) {
            const Square    s        = pop_lsb(occ);
            const Piece     p        = pos.piece_on(s);
            const PieceType pt       = type_of(p);
            const Color     pc       = color_of(p);
            const bool      is_ours  = (pc == persp_color);
            if (is_ours && pt == King) continue;  // own king is implicit

            Square piece_sq = s;
            if (persp_color == Black) piece_sq = flip_rank(piece_sq);

            const int idx = feature_index(king_sq, piece_sq, pt, is_ours);
            const std::int16_t* col = &g_ft_w[static_cast<std::size_t>(idx) * kFtOutSize];

#if defined(__AVX2__)
            for (std::size_t i = 0; i < kFtOutSize; i += 16) {
                __m256i acc_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&acc.v[persp][i]));
                __m256i col_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(&col[i]));
                acc_vec = _mm256_add_epi16(acc_vec, col_vec);
                _mm256_store_si256(reinterpret_cast<__m256i*>(&acc.v[persp][i]), acc_vec);
            }
#elif defined(__ARM_NEON)
            for (std::size_t i = 0; i < kFtOutSize; i += 8) {
                int16x8_t acc_vec = vld1q_s16(&acc.v[persp][i]);
                int16x8_t col_vec = vld1q_s16(&col[i]);
                acc_vec = vaddq_s16(acc_vec, col_vec);
                vst1q_s16(&acc.v[persp][i], acc_vec);
            }
#else
            for (std::size_t i = 0; i < kFtOutSize; ++i) {
                acc.v[persp][i] = static_cast<std::int16_t>(acc.v[persp][i] + col[i]);
            }
#endif
        }
    }

    acc.computed = true;
}

// ---- Inference ------------------------------------------------------------

namespace {

// L1/L2 dense int8 dot product with int32 bias, post-shift by kWeightShift,
// clamp into [0, kFtQuant] for the next layer's uint8 input.
//
// Three implementations of the same arithmetic contract:
//   sum = b[o] + dot(uint8 in[kIn], int8 w[o, kIn])
//   out[o] = clamp(sum >> kWeightShift, 0, kFtQuant)
//
// Inputs are alignas(64) stack buffers from evaluate(). Weights live in a
// std::vector and have only 1-byte guaranteed alignment, so the weight load
// is unaligned; the throughput cost of loadu vs load is zero on Haswell+ when
// the address actually is aligned, which it usually is once kIn ≥ 32.
template <int kIn, int kOut>
void affine_clipped_relu(const std::uint8_t* in,
                         const std::int8_t*  w,        // [kOut, kIn]
                         const std::int32_t* b,        // [kOut]
                         std::uint8_t*       out) noexcept {
#if defined(__AVX2__)
    static_assert(kIn % 32 == 0, "AVX2 affine_clipped_relu: kIn must be a multiple of 32");
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int o = 0; o < kOut; ++o) {
        const std::int8_t* wo = w + o * kIn;
        __m256i acc = _mm256_setzero_si256();
        for (int i = 0; i < kIn; i += 32) {
            // 32 uint8 inputs (aligned), 32 int8 weights (unaligned-safe).
            const __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(in + i));
            const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wo + i));
            // maddubs_epi16: u8 × i8 → i16, summing adjacent pairs (saturating).
            // For uint8≤127 × int8 in [-128,127], the pair sum fits in int16
            // without saturation (worst case 127*127 + 127*(-128) ≈ -32).
            const __m256i p = _mm256_maddubs_epi16(x, y);
            // madd_epi16: i16 × i16 (here ×1) → i32, summing adjacent pairs.
            // Result: 8 lanes of int32 partial sums, no saturation possible.
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones16));
        }
        // Horizontal-sum the 8 int32 lanes into one.
        __m128i lo  = _mm256_castsi256_si128(acc);
        __m128i hi  = _mm256_extracti128_si256(acc, 1);
        __m128i s4  = _mm_add_epi32(lo, hi);
        __m128i s2  = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));  // swap halves
        __m128i s1  = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0xB1));  // swap pairs
        std::int32_t sum = b[o] + _mm_cvtsi128_si32(s1);
        sum >>= kWeightShift;
        if (sum < 0)         sum = 0;
        if (sum > kFtQuant)  sum = kFtQuant;
        out[o] = static_cast<std::uint8_t>(sum);
    }
#elif defined(__ARM_NEON)
    static_assert(kIn % 16 == 0, "NEON affine_clipped_relu: kIn must be a multiple of 16");
    for (int o = 0; o < kOut; ++o) {
        const std::int8_t* wo = w + o * kIn;
        int32x4_t acc = vdupq_n_s32(0);
        for (int i = 0; i < kIn; i += 16) {
            // 16 uint8 → 16 int16 (in fits in int8 since values ≤ 127), then
            // signed multiply-accumulate against int8 weights via widening.
            const int8x16_t x = vreinterpretq_s8_u8(vld1q_u8(in + i));
            const int8x16_t y = vld1q_s8(wo + i);
            const int16x8_t lo = vmull_s8(vget_low_s8(x),  vget_low_s8(y));   // 8× i16
            const int16x8_t hi = vmull_s8(vget_high_s8(x), vget_high_s8(y));  // 8× i16
            acc = vpadalq_s16(acc, lo);
            acc = vpadalq_s16(acc, hi);
        }
        std::int32_t sum = b[o] + vaddvq_s32(acc);
        sum >>= kWeightShift;
        if (sum < 0)         sum = 0;
        if (sum > kFtQuant)  sum = kFtQuant;
        out[o] = static_cast<std::uint8_t>(sum);
    }
#else
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
#endif
}

#if defined(__AVX2__)
template <int kIn, int kOut>
void affine_clipped_relu_batch(const std::uint8_t* in,
                               const std::int8_t*  w,
                               const std::int32_t* b,
                               std::uint8_t*       out,
                               int                 batch_size) noexcept {
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int o = 0; o < kOut; ++o) {
        const std::int8_t* wo = w + o * kIn;
        
        // Process 4 inputs in the batch at a time to maximize weight vector reuse.
        int b_idx = 0;
        for (; b_idx <= batch_size - 4; b_idx += 4) {
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();

            for (int i = 0; i < kIn; i += 32) {
                const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wo + i));
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(in + (b_idx + 0) * kIn + i)), wv), ones16));
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(in + (b_idx + 1) * kIn + i)), wv), ones16));
                acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(in + (b_idx + 2) * kIn + i)), wv), ones16));
                acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(in + (b_idx + 3) * kIn + i)), wv), ones16));
            }

            auto hsum = [](const __m256i& acc) -> std::int32_t {
                __m128i lo = _mm256_castsi256_si128(acc);
                __m128i hi = _mm256_extracti128_si256(acc, 1);
                __m128i s4 = _mm_add_epi32(lo, hi);
                __m128i s2 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
                __m128i s1 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0xB1));
                return _mm_cvtsi128_si32(s1);
            };

            auto finalize = [&](int batch_offset, const __m256i& acc) {
                std::int32_t sum = b[o] + hsum(acc);
                sum >>= kWeightShift;
                if (sum < 0)         sum = 0;
                if (sum > kFtQuant)  sum = kFtQuant;
                out[batch_offset * kOut + o] = static_cast<std::uint8_t>(sum);
            };

            finalize(b_idx + 0, acc0);
            finalize(b_idx + 1, acc1);
            finalize(b_idx + 2, acc2);
            finalize(b_idx + 3, acc3);
        }

        // Remainder
        for (; b_idx < batch_size; ++b_idx) {
            affine_clipped_relu<kIn, 1>(in + b_idx * kIn, wo, b + o, out + b_idx * kOut + o);
        }
    }
}
#endif

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

    // FT concat + clipped ReLU: [acc_us, acc_them] -> uint8[2048].
    //
    // Per int16 lane: clamp into [0, kFtQuant=127] and pack into uint8. On
    // AVX2 packus_epi16 handles both the upper saturation (>127 -> 127) and
    // the lower (signed negatives -> 0) for free in one instruction per 16
    // lanes, vs the scalar form which needs branchy clamp+cast per lane.
    // Called once per evaluate() so it's not the dominant cost, but at 36
    // NNUE evals per MCTS expansion it adds up.
    alignas(64) std::uint8_t ft_out[kL1InSize];
#if defined(__AVX2__)
    {
        const __m256i clamp_hi = _mm256_set1_epi16(static_cast<std::int16_t>(kFtQuant));
        // Process 32 int16 lanes (= 32 uint8 outputs) per iteration.
        // packus_epi16 takes two __m256i of int16, returns __m256i of
        // uint8 with cross-lane interleaving, so unpack_perm with the
        // permute below restores [0..15, 16..31] order.
        // packus_epi16(a, b) produces lanes [a_lo, b_lo, a_hi, b_hi] in
        // 64-bit chunks. We want [a_lo, a_hi, b_lo, b_hi] so the 32
        // contiguous int16 input values map to 32 contiguous uint8 outputs
        // in their original order.
        const __m256i perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        for (int half = 0; half < 2; ++half) {
            const std::size_t persp = (half == 0) ? us : them;
            const std::int16_t* acc_ptr = acc.v[persp].data();
            std::uint8_t*       out_ptr = ft_out + half * kFtOutSize;
            for (int i = 0; i < kFtOutSize; i += 32) {
                __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc_ptr + i));
                __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc_ptr + i + 16));
                a = _mm256_min_epi16(a, clamp_hi);
                b = _mm256_min_epi16(b, clamp_hi);
                // packus also saturates negatives to 0, so no explicit max needed.
                __m256i packed = _mm256_packus_epi16(a, b);
                packed = _mm256_permutevar8x32_epi32(packed, perm);
                _mm256_store_si256(reinterpret_cast<__m256i*>(out_ptr + i), packed);
            }
        }
    }
#else
    for (std::size_t i = 0; i < kFtOutSize; ++i) {
        const std::int16_t a = acc.v[us][i];
        const std::int16_t b = acc.v[them][i];
        ft_out[i]              = static_cast<std::uint8_t>(std::clamp<std::int32_t>(a, 0, kFtQuant));
        ft_out[i + kFtOutSize] = static_cast<std::uint8_t>(std::clamp<std::int32_t>(b, 0, kFtQuant));
    }
#endif

    alignas(64) std::uint8_t l1_out[kL1OutSize];
    affine_clipped_relu<kL1InSize, kL1OutSize>(ft_out, g_l1_w.data(), g_l1_b.data(), l1_out);

    alignas(64) std::uint8_t l2_out[kL2OutSize];
    affine_clipped_relu<kL1OutSize, kL2OutSize>(l1_out, g_l2_w.data(), g_l2_b.data(), l2_out);

    // Final layer: kL2OutSize -> kL3OutSize (3), no ReLU.
    std::array<float, 3> logits;
    for (int j = 0; j < kL3OutSize; ++j) {
        std::int32_t y = g_l3_b[static_cast<std::size_t>(j)];
        const std::int8_t* w3 = &g_l3_w[static_cast<std::size_t>(j * kL2OutSize)];
        for (int i = 0; i < kL2OutSize; ++i) {
            y += static_cast<std::int32_t>(w3[static_cast<std::size_t>(i)]) * static_cast<std::int32_t>(l2_out[static_cast<std::size_t>(i)]);
        }
        logits[static_cast<std::size_t>(j)] = static_cast<float>(y);
    }

    // Softmax over logits to get [P(Win), P(Draw), P(Loss)]
    float max_l = logits[0];
    if (logits[1] > max_l) max_l = logits[1];
    if (logits[2] > max_l) max_l = logits[2];

    float sum_p = 0.0f;
    std::array<float, 3> probs;
    constexpr float kLogitScale = static_cast<float>(kWeightScale * kFtQuant);
    for (int j = 0; j < 3; ++j) {
        probs[static_cast<std::size_t>(j)] = std::exp((logits[static_cast<std::size_t>(j)] - max_l) / kLogitScale);
        sum_p += probs[static_cast<std::size_t>(j)];
    }

    // Inverse-sigmoid (logit) of P(W) + 0.5·P(D). Half of P(D) is treated as
    // expected score, matching the standard Elo-style win-expectancy convention.
    // The transform is unbounded — a clearly winning position with P_W=0.99
    // produces cp ≈ 4.6·scale; the previous linear formula `(P_W − P_L)·scale`
    // was hard-capped at ±scale, which compressed every eval near the ceiling
    // and starved MCTS priors of signal in winning positions (e.g. K+Q showed
    // the same +355 cp as K+R because both saturated).
    const float pw = probs[0] / sum_p;
    const float pd = probs[1] / sum_p;
    const float win_expectancy = std::clamp(pw + 0.5f * pd, 1e-6f, 1.0f - 1e-6f);
    const float cp = std::log(win_expectancy / (1.0f - win_expectancy)) * g_output_cp_per_unit;

    return static_cast<Score>(cp);
}

void evaluate_batch(const Accumulator* accs, const Color* stms, Score* scores, int batch_size) noexcept {
    if (batch_size <= 0) return;
    if (!is_loaded()) {
        for (int i = 0; i < batch_size; ++i) scores[i] = 0;
        return;
    }

    // Thread-local scratch buffers to avoid allocations per call.
    // Max batch size is usually moves.size (max ~218 in chess, but usually < 50).
    // We'll support up to 256.
    constexpr int kMaxBatch = 256;
    alignas(64) static thread_local std::uint8_t ft_scratch[kMaxBatch * kL1InSize];
    alignas(64) static thread_local std::uint8_t l1_scratch[kMaxBatch * kL1OutSize];
    alignas(64) static thread_local std::uint8_t l2_scratch[kMaxBatch * kL2OutSize];

    const int n = std::min(batch_size, kMaxBatch);

    // 1. Feature Transformer (Accumulator -> FT_out). Same packus_epi16
    //    SIMD as evaluate(), applied per batch element.
#if defined(__AVX2__)
    {
        const __m256i clamp_hi = _mm256_set1_epi16(static_cast<std::int16_t>(kFtQuant));
        const __m256i perm     = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for (int b = 0; b < n; ++b) {
            const Accumulator& acc = accs[b];
            const Color       stm  = stms[b];
            const std::size_t us   = (stm == White) ? 0u : 1u;
            const std::size_t them = us ^ 1u;
            std::uint8_t* ft_out = &ft_scratch[b * kL1InSize];
            for (int half = 0; half < 2; ++half) {
                const std::size_t persp = (half == 0) ? us : them;
                const std::int16_t* acc_ptr = acc.v[persp].data();
                std::uint8_t*       out_ptr = ft_out + half * kFtOutSize;
                for (int i = 0; i < kFtOutSize; i += 32) {
                    __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc_ptr + i));
                    __m256i e = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc_ptr + i + 16));
                    a = _mm256_min_epi16(a, clamp_hi);
                    e = _mm256_min_epi16(e, clamp_hi);
                    __m256i packed = _mm256_packus_epi16(a, e);
                    packed = _mm256_permutevar8x32_epi32(packed, perm);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(out_ptr + i), packed);
                }
            }
        }
    }
#else
    for (int b = 0; b < n; ++b) {
        const Accumulator& acc = accs[b];
        const Color       stm  = stms[b];
        const std::size_t us   = (stm == White) ? 0u : 1u;
        const std::size_t them = us ^ 1u;
        std::uint8_t* ft_out = &ft_scratch[b * kL1InSize];
        for (std::size_t i = 0; i < kFtOutSize; ++i) {
            ft_out[i]              = static_cast<std::uint8_t>(std::clamp<std::int32_t>(acc.v[us][i], 0, kFtQuant));
            ft_out[i + kFtOutSize] = static_cast<std::uint8_t>(std::clamp<std::int32_t>(acc.v[them][i], 0, kFtQuant));
        }
    }
#endif

    // 2. Batched L1 Layer
#if defined(__AVX2__)
    affine_clipped_relu_batch<kL1InSize, kL1OutSize>(ft_scratch, g_l1_w.data(), g_l1_b.data(), l1_scratch, n);
#else
    for (int b = 0; b < n; ++b) {
        affine_clipped_relu<kL1InSize, kL1OutSize>(&ft_scratch[b * kL1InSize], g_l1_w.data(), g_l1_b.data(), &l1_scratch[b * kL1OutSize]);
    }
#endif

    // 3. Batched L2 Layer
#if defined(__AVX2__)
    affine_clipped_relu_batch<kL1OutSize, kL2OutSize>(l1_scratch, g_l2_w.data(), g_l2_b.data(), l2_scratch, n);
#else
    for (int b = 0; b < n; ++b) {
        affine_clipped_relu<kL1OutSize, kL2OutSize>(&l1_scratch[b * kL1OutSize], g_l2_w.data(), g_l2_b.data(), &l2_scratch[b * kL2OutSize]);
    }
#endif

    // 4. Final L3 and Score Calculation
    for (int b = 0; b < n; ++b) {
        const std::uint8_t* l2_out = &l2_scratch[b * kL2OutSize];
        std::array<float, 3> logits;
        for (int j = 0; j < kL3OutSize; ++j) {
            std::int32_t y = g_l3_b[static_cast<std::size_t>(j)];
            const std::int8_t* w3 = &g_l3_w[static_cast<std::size_t>(j * kL2OutSize)];
            for (int i = 0; i < kL2OutSize; ++i) {
                y += static_cast<std::int32_t>(w3[static_cast<std::size_t>(i)]) * static_cast<std::int32_t>(l2_out[static_cast<std::size_t>(i)]);
            }
            logits[static_cast<std::size_t>(j)] = static_cast<float>(y);
        }

        float max_l = logits[0];
        if (logits[1] > max_l) max_l = logits[1];
        if (logits[2] > max_l) max_l = logits[2];

        float sum_p = 0.0f;
        std::array<float, 3> probs;
        constexpr float kLogitScale = static_cast<float>(kWeightScale * kFtQuant);
        for (int j = 0; j < 3; ++j) {
            probs[static_cast<std::size_t>(j)] = std::exp((logits[static_cast<std::size_t>(j)] - max_l) / kLogitScale);
            sum_p += probs[static_cast<std::size_t>(j)];
        }

        // Inverse-sigmoid (logit) — see evaluate() for the rationale.
        const float pw = probs[0] / sum_p;
        const float pd = probs[1] / sum_p;
        const float we = std::clamp(pw + 0.5f * pd, 1e-6f, 1.0f - 1e-6f);
        scores[b] = static_cast<Score>(std::log(we / (1.0f - we)) * g_output_cp_per_unit);
    }
}

}  // namespace eclipse::nnue
