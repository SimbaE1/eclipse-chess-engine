// SPDX-License-Identifier: GPL-3.0-or-later
#include "policy.hpp"
#include "movegen.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

namespace eclipse::policy {

namespace {

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EclipsePolicy");
Ort::SessionOptions session_options;
std::unique_ptr<Ort::Session> session;

const int kInputPlanes = 112;
const int kBoardSize = 8;
const int kPolicySize = 1858;

// Map from UCI move string to Lc0 policy index.
std::map<std::string, int> uci_to_idx;
std::vector<std::string> idx_to_uci;

#include <unistd.h>
#include <limits.h>

bool init_policy_map(const std::string& path) {
    char cwd[PATH_MAX];
    getcwd(cwd, sizeof(cwd));
    std::cout << "info string Trying to open policy map: " << path << " in " << cwd << std::endl;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "info string Failed to open: " << path << std::endl;
        return false;
    }

    uci_to_idx.clear();
    idx_to_uci.clear();
    idx_to_uci.resize(kPolicySize);

    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        
        try {
            int idx = std::stoi(line.substr(0, colon));
            std::string uci = line.substr(colon + 1);
            if (idx >= 0 && idx < kPolicySize) {
                idx_to_uci[static_cast<size_t>(idx)] = uci;
                uci_to_idx[uci] = idx;
                count++;
            }
        } catch (...) {
            continue;
        }
    }
    
    if (count > 0) {
        std::cout << "info string Loaded " << count << " policy moves from " << path << std::endl;
        return true;
    }
    return false;
}

// Lc0 112-plane encoding.
//
// Plane layout (matches lczero classical position encoder):
//   0..103   : 8 history frames * 13 planes each
//                per frame: 6 our piece types, 6 their piece types, 1 repetition
//   104..107 : castling rights (us KS, us QS, them KS, them QS)
//   108      : side-to-move (1 if Black to move)
//   109      : 50-move counter / 100 (normalized [0, 1])
//   110      : all 1s (positional bias / "always-on" plane)
//   111      : always 0
//
// We don't thread move history through MCTS yet, so the 8 history frames are
// all filled with the *current* position. Lc0 nets saw identical-frames inputs
// during early-game training (before move 8), so degraded gracefully - we
// recover most of the policy quality this way without a search-side refactor.
// True history threading can come as Phase 5.
void tensorize(const Position& pos, float* input) {
    std::fill(input, input + kInputPlanes * kBoardSize * kBoardSize, 0.0f);

    const Color us   = pos.side_to_move();
    const Color them = ~us;

    // Fill the 13 planes of one history frame at plane_base.
    auto fill_frame = [&](int plane_base) {
        // Our pieces - 6 planes (P, N, B, R, Q, K -> 0..5)
        for (int pt = Pawn; pt <= King; ++pt) {
            Bitboard b = pos.pieces(us, PieceType(pt));
            while (b) {
                Square s = pop_lsb(b);
                const int sq_idx = (us == White) ? static_cast<int>(s)
                                                  : static_cast<int>(s ^ 56);
                input[(plane_base + (pt - Pawn)) * 64 + sq_idx] = 1.0f;
            }
        }
        // Their pieces - 6 planes (offsets 6..11)
        for (int pt = Pawn; pt <= King; ++pt) {
            Bitboard b = pos.pieces(them, PieceType(pt));
            while (b) {
                Square s = pop_lsb(b);
                const int sq_idx = (us == White) ? static_cast<int>(s)
                                                  : static_cast<int>(s ^ 56);
                input[(plane_base + 6 + (pt - Pawn)) * 64 + sq_idx] = 1.0f;
            }
        }
        // Plane base+12: repetition indicator - left zero (we don't track
        // 3-fold rep at the policy layer; MCTS already terminates on it).
    };

    // 8 history frames * 13 planes = 104 planes (0..103).
    for (int frame = 0; frame < 8; ++frame) {
        fill_frame(frame * 13);
    }

    // Castling rights, perspective-relative.
    const CastlingRights cr = pos.castling_rights();
    const CastlingRights us_ks   = (us == White) ? WhiteKingside  : BlackKingside;
    const CastlingRights us_qs   = (us == White) ? WhiteQueenside : BlackQueenside;
    const CastlingRights them_ks = (us == White) ? BlackKingside  : WhiteKingside;
    const CastlingRights them_qs = (us == White) ? BlackQueenside : WhiteQueenside;
    if (cr & us_ks)   std::fill_n(input + 104 * 64, 64, 1.0f);
    if (cr & us_qs)   std::fill_n(input + 105 * 64, 64, 1.0f);
    if (cr & them_ks) std::fill_n(input + 106 * 64, 64, 1.0f);
    if (cr & them_qs) std::fill_n(input + 107 * 64, 64, 1.0f);

    // Side-to-move marker. Lc0 convention is 1 if Black to move. Note that
    // the board has already been mirrored to "us"'s frame above, so this
    // plane gives the network a small bias signal about whose POV it's in.
    if (us == Black) std::fill_n(input + 108 * 64, 64, 1.0f);

    // 50-move counter / 100, clipped. Capped at 100 plies so the plane stays
    // in [0, 1].
    const float rule50 = std::min(100, pos.halfmove_clock()) / 100.0f;
    std::fill_n(input + 109 * 64, 64, rule50);

    // Always-on plane 110.
    std::fill_n(input + 110 * 64, 64, 1.0f);
    // Plane 111 stays zero.
}

} // namespace

bool load(const std::string& path) {
    std::cout << "info string Loading policy from: " << path << std::endl;
    try {
        // Use every physical core for intra-op parallelism (matrix multiplies
        // inside the transformer). 0 = let ORT pick, which on macOS resolves
        // to physical core count. Inter-op stays at 1 because each call to
        // get_policy()/get_root_info() runs one inference at a time anyway -
        // dispatching multiple concurrent sessions would just thrash the cache.
        // Combined with ORT_ENABLE_ALL graph optimization this is usually
        // 3-5x faster than the defaults on Apple Silicon.
        session_options.SetIntraOpNumThreads(0);
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef __APPLE__
        // CoreML EP: dispatches supported ops to the Apple Neural Engine
        // (Apple Silicon) or the discrete/integrated GPU via Metal (Intel
        // Macs with Radeon Vega / Iris). Ops the EP can't handle silently
        // fall back to the CPU EP, so this is safe to always enable. Honor
        // ECLIPSE_DISABLE_COREML=1 as an escape hatch for triage.
        //
        // First-run cost: ORT compiles supported subgraphs to MLProgram
        // format, which can take a few seconds on a large transformer. The
        // result is cached in /tmp by default; survives within a process but
        // not across processes (would need ModelCacheDirectory for that).
        bool coreml_disabled = false;
        if (const char* env_disable = std::getenv("ECLIPSE_DISABLE_COREML")) {
            coreml_disabled = env_disable[0] != '\0' && env_disable[0] != '0';
        }
        if (!coreml_disabled) {
            try {
                uint32_t coreml_flags = COREML_FLAG_USE_NONE;
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
                    session_options, coreml_flags));
                std::cout << "info string CoreML EP enabled "
                             "(falls back to CPU for unsupported ops)" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "info string CoreML EP unavailable, using CPU only: "
                          << e.what() << std::endl;
            }
        }
#endif

        session = std::make_unique<Ort::Session>(env, path.c_str(), session_options);
        std::cout << "info string Policy NN loaded: " << path << std::endl;

        // Resolve lc0_policy_map.txt relative to the .onnx we just opened.
        // The canonical home is `data/lc0_policy_map.txt`; we also look in the
        // same directory as `path` (handy if the user keeps weights in a
        // non-standard location) and finally the CWD.
        std::vector<std::string> candidates;
        auto slash = path.find_last_of('/');
        if (slash != std::string::npos) {
            candidates.push_back(path.substr(0, slash + 1) + "lc0_policy_map.txt");
        }
        candidates.push_back("data/lc0_policy_map.txt");
        candidates.push_back("lc0_policy_map.txt");

        bool loaded_map = false;
        for (const auto& c : candidates) {
            if (init_policy_map(c)) { loaded_map = true; break; }
        }
        if (!loaded_map) {
            std::cerr << "info string Warning: lc0_policy_map.txt not found; "
                         "policy net will fall back to uniform priors "
                         "(expect very weak play)." << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading policy net: " << e.what() << std::endl;
        return false;
    }
}

RootInfo get_root_info(const Position& pos) {
    RootInfo result;  // defaults: 60 plies left, 0 cp
    if (!session) return result;

    std::vector<float> input_tensor_values(1 * kInputPlanes * kBoardSize * kBoardSize);
    tensorize(pos, input_tensor_values.data());

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, kInputPlanes, kBoardSize, kBoardSize};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    // Request WDL + MLH only. Policy is computed by the network anyway but
    // we don't read it - that's get_policy()'s job.
    const char* input_names[]  = {"/input/planes"};
    const char* output_names[] = {"/output/policy", "/output/wdl", "/output/mlh"};
    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                        input_names, &input_tensor, 1,
                                        output_names, 3);

    const float* wdl = output_tensors[1].GetTensorMutableData<float>();
    const float  pw = wdl[0], pd = wdl[1], pl = wdl[2];

    // Convert WDL to a centipawn-ish score. Two reasonable choices:
    //   (a) cheap linear:  cp = 410 * (pw - pl)
    //   (b) log-odds:      cp = 410 * 0.5 * ln((pw + pd/2) / (pl + pd/2))
    // (b) better matches the sigmoid-trained NNUE scale at the extremes but
    // both are dominated by NNUE in current usage. Going with (a) for speed.
    result.value_cp = static_cast<int>(410.0f * (pw - pl));

    const float mlh = output_tensors[2].GetTensorMutableData<float>()[0];
    // The MLH head outputs an unbounded positive scalar; clamp to a sane
    // window so a bad prediction can't make time management nonsensical.
    result.mlh_plies = std::clamp(mlh, 2.0f, 200.0f);

    return result;
}

std::map<Move, float> get_policy(const Position& pos) {
    if (!session || uci_to_idx.empty()) {
        // Fallback to uniform if no session or no map
        MoveList moves;
        Position temp_pos = pos;
        generate_legal_moves(temp_pos, moves);
        std::map<Move, float> distribution;
        if (moves.size == 0) return distribution;
        float prob = 1.0f / static_cast<float>(moves.size);
        for (const Move m : moves) distribution[m] = prob;
        return distribution;
    }

    std::vector<float> input_tensor_values(1 * kInputPlanes * kBoardSize * kBoardSize);
    tensorize(pos, input_tensor_values.data());

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, kInputPlanes, kBoardSize, kBoardSize};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), 
        input_shape.data(), input_shape.size());

    // Tensor names match what `lc0 leela2onnx` emits for modern T-series nets
    // (attention-policy + WDL head). The old "input"/"policy"/"value" naming
    // belonged to the legacy CNN export path.
    //
    // WDL output is requested but ignored here - we only consume policy. If a
    // future caller wants game-result probabilities for value-net usage, read
    // output_tensors[1] which is a 3-element [P(win), P(draw), P(loss)] tensor.
    const char* input_names[]  = {"/input/planes"};
    const char* output_names[] = {"/output/policy", "/output/wdl"};

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                        input_names, &input_tensor, 1,
                                        output_names, 2);
    auto& policy_tensor = output_tensors[0];

    float* policy_logits = policy_tensor.GetTensorMutableData<float>();

    // Per-call debug print is noisy in real search; uncomment when triaging.
    // std::cout << "info string Logits: " << policy_logits[0] << ", " << policy_logits[1] << std::endl;

    // Softmax over legal moves
    MoveList moves;
    Position temp_pos = pos;
    generate_legal_moves(temp_pos, moves);
    
    std::map<Move, float> distribution;
    if (moves.size == 0) return distribution;

    Color us = pos.side_to_move();
    float sum = 0.0f;

    for (const Move m : moves) {
        Move model_m = m;
        if (us == Black) {
            Square f = flip_rank(m.from());
            Square t = flip_rank(m.to());
            if (m.type() == Move::Promotion) {
                model_m = Move::make_promotion(f, t, m.promotion_piece());
            } else if (m.type() == Move::EnPassant) {
                model_m = Move::make_en_passant(f, t);
            } else if (m.type() == Move::Castling) {
                model_m = Move::make_castling(f, t);
            } else {
                model_m = Move(f, t);
            }
        }
        
        std::string uci = model_m.to_uci();
        float logit = -10.0f; 
        auto it = uci_to_idx.find(uci);
        if (it != uci_to_idx.end()) {
            logit = policy_logits[it->second];
        }
        
        float prob = std::exp(logit);
        distribution[m] = prob;
        sum += prob;
    }

    if (sum > 0) {
        for (auto& [move, prob] : distribution) {
            prob /= sum;
        }
    } else {
        float uniform = 1.0f / static_cast<float>(moves.size);
        for (const Move m : moves) distribution[m] = uniform;
    }

    return distribution;
}

}  // namespace eclipse::policy

