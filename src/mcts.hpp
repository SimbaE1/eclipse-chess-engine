// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <map>

#include "move.hpp"
#include "position.hpp"
#include "search.hpp"

namespace eclipse::mcts {

struct Node {
    Move move = MoveNone;
    Node* parent = nullptr;
    std::vector<std::unique_ptr<Node>> children;
    
    float Q = 0.0f;          // Average value (from perspective of side to move in parent)
    float W = 0.0f;          // Total accumulated value
    float N = 0.0f;          // Visit count
    float P = 0.0f;          // Policy prior
    
    bool is_expanded = false;
    bool is_terminal = false;

    Node(Move m, Node* p, float prior) : move(m), parent(p), P(prior) {}

    // PUCT formula: Q + c * P * sqrt(N_parent) / (1 + N)
    float puct_score(float total_n, float cpuct) const {
        return Q + cpuct * P * std::sqrt(total_n) / (1.0f + N);
    }
};

class MCTS {
public:
    MCTS(Position& pos, SearchInfo& info) : root_pos(pos), search_info(info) {}
    
    Move search();

private:
    void iterate();
    Node* select(Node* node, Position& pos);
    void expand(Node* node, const Position& pos);
    float evaluate_node(const Position& pos);
    void backpropagate(Node* node, float value);

    Position& root_pos;
    SearchInfo& search_info;
    std::unique_ptr<Node> root;
    
    const float kCpuct = 1.41f; // Tuning constant
};

}  // namespace eclipse::mcts
