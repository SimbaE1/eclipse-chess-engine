// SPDX-License-Identifier: GPL-3.0-or-later
//
// Verifies nnue::evaluate_batch() is bit-for-bit identical to repeated
// nnue::evaluate() calls. The MCTS hot path relies entirely on the batched
// kernel, but the standing test suite only exercises the single-eval path, so
// a divergence in the batched affine (e.g. the AVX-512 group-of-8 kernel)
// would otherwise go unnoticed.

#include <cstdlib>
#include <string>
#include <vector>

#include "attacks.hpp"
#include "check.hpp"
#include "nnue.hpp"
#include "position.hpp"
#include "zobrist.hpp"

using namespace eclipse;

int main() {
    zobrist::init();
    init_attacks();

    const char* path = std::getenv("ECLIPSE_NNUE_PATH");
    if (path == nullptr) {
        std::printf("SKIP  nnue_batch (ECLIPSE_NNUE_PATH not set)\n");
        return 0;
    }
    if (!nnue::load(path)) {
        std::printf("SKIP  test_nnue_batch (failed to load %s)\n", path);
        return 0;
    }

    const std::vector<std::string> fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",
        "rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 6",
        "2kr3r/pppb1ppp/2n1pn2/8/2BP4/2N1BN2/PPP2PPP/2KR3R b - - 0 1",
    };

    // Build accumulators for every position, then evaluate one-by-one and as a
    // single batch and require identical scores.
    std::vector<nnue::Accumulator> accs;
    std::vector<Color>             stms;
    std::vector<Score>             single;
    for (const auto& fen : fens) {
        Position p;
        ECLIPSE_CHECK(p.set_from_fen(fen));
        // Touch the accumulator through evaluate() so it is computed.
        const Score s = nnue::evaluate(p);
        single.push_back(s);
        accs.push_back(p.accumulator());
        stms.push_back(p.side_to_move());
    }

    // Batch over a range of sizes so the 8-wide, 4-wide, and 1-wide paths in
    // affine_clipped_relu_batch are all exercised.
    for (int n = 1; n <= static_cast<int>(fens.size()); ++n) {
        std::vector<Score> batch(static_cast<std::size_t>(n));
        nnue::evaluate_batch(accs.data(), stms.data(), batch.data(), n);
        for (int i = 0; i < n; ++i) {
            ECLIPSE_CHECK(batch[static_cast<std::size_t>(i)] ==
                          single[static_cast<std::size_t>(i)]);
        }
    }

    return eclipse::test::summarize("test_nnue_batch");
}
