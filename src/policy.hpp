// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <vector>
#include <string>
#include <map>

#include "move.hpp"
#include "position.hpp"

namespace eclipse::policy {

// Initialize the policy head (load weights if needed).
// For now, this is a placeholder for the Lc0 distilled net.
bool load(const std::string& path);

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
