// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <vector>
#include <string>
#include <map>

#include "move.hpp"
#include "position.hpp"

namespace eclipse::policy {

// Initialize the ONNX policy head (load weights if needed).
// Only consulted when set_mode(Mode::Onnx) has been called.
bool load(const std::string& path);

// Source of move priors. Onnx routes through the Lc0 transformer
// (~80ms per call, dominates NPS); Nnue runs the NNUE static eval on each
// child (microseconds total) and softmaxes the resulting score differential.
// The latter is the default — the whole point of NNUE+MCTS is to keep the
// per-expansion cost bounded by a fast static eval.
enum class Mode { Nnue, Onnx };
void set_mode(Mode m) noexcept;
Mode get_mode() noexcept;

// Returns a probability distribution over all legal moves in the position.
// The map keys are legal moves, and values are probabilities in [0, 1].
std::map<Move, float> get_policy(const Position& pos);

// One-shot root-position query for time management. Uses the same network
// forward pass as get_policy() but returns the MLH (moves-left-head) and
// WDL value head instead. Call once per `go` command, not per MCTS node.
//
// `mlh_plies` is the network's estimate of remaining half-moves in the game
// (clipped to a sane range). `value_cp` is the WDL converted to centipawns.
// Both default to neutral fallback values when no policy network is loaded.
struct RootInfo {
    float mlh_plies = 60.0f;   // ~30 full moves left
    int   value_cp  = 0;
};
RootInfo get_root_info(const Position& pos);

}  // namespace eclipse::policy
