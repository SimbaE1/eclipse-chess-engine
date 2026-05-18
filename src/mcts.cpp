// SPDX-License-Identifier: GPL-3.0-or-later
#include "mcts.hpp"
#include "policy.hpp"
#include "eval.hpp"
#include "movegen.hpp"
#include <iostream>
#include <algorithm>

namespace eclipse::mcts {

Move MCTS::search() {
    root = std::make_unique<Node>(MoveNone, nullptr, 1.0f);
    
    // Initial expansion
    expand(root.get(), root_pos);

    while (!search_info.time_up() && !search_info.stop.load(std::memory_order_relaxed)) {
        iterate();
        search_info.nodes_searched++;
        
        // Depth limit: simple visit limit per root child for now.
        if (search_info.limits.depth > 0 && search_info.nodes_searched >= search_info.limits.depth) {
            break;
        }
        
        // Output info occasionally
        if ((search_info.nodes_searched & 0x3FF) == 0) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - search_info.start_time).count();
            if (elapsed > 0) {
                // Find best move so far (max N)
                Node* best_child = nullptr;
                for (const auto& child : root->children) {
                    if (!best_child || child->N > best_child->N) {
                        best_child = child.get();
                    }
                }
                if (best_child) {
                    std::cout << "info nodes " << search_info.nodes_searched 
                              << " nps " << (search_info.nodes_searched * 1000 / elapsed)
                              << " pv " << best_child->move.to_uci() << std::endl;
                }
            }
        }
    }

    // Return move with highest visit count
    Node* best_child = nullptr;
    for (const auto& child : root->children) {
        if (!best_child || child->N > best_child->N) {
            best_child = child.get();
        }
    }

    return best_child ? best_child->move : MoveNone;
}

void MCTS::iterate() {
    Position pos = root_pos;
    Node* selected = select(root.get(), pos);
    
    // Log selected move for debugging
    if (selected->move != MoveNone) {
        // std::cout << "info string Selecting " << selected->move.to_uci() << std::endl;
    }
    
    float value;
    if (selected->is_terminal) {
        value = selected->Q;
    } else {
        expand(selected, pos);
        value = evaluate_node(pos);
    }
    
    backpropagate(selected, value);
}

Node* MCTS::select(Node* node, Position& pos) {
    StateInfo st; // Temporary state for do_move
    while (node->is_expanded && !node->is_terminal) {
        float best_score = -1e9f;
        Node* best_child = nullptr;
        
        for (const auto& child : node->children) {
            float score = child->puct_score(node->N, kCpuct);
            if (score > best_score) {
                best_score = score;
                best_child = child.get();
            }
        }
        
        if (!best_child) break;
        pos.do_move(best_child->move, st);
        node = best_child;
    }
    return node;
}

void MCTS::expand(Node* node, const Position& pos) {
    if (node->is_expanded) return;

    MoveList moves;
    Position temp_pos = pos;
    generate_legal_moves(temp_pos, moves);
    
    if (moves.size == 0) {
        node->is_terminal = true;
        node->Q = pos.in_check() ? -1.0f : 0.0f; // Loss if in check, Draw otherwise
        return;
    }

    auto priors = policy::get_policy(pos);

    // Sanity-check the policy distribution. Only meaningful when there are
    // at least 2 legal moves - a single-move position trivially has prob=1.0
    // which would match "uniform = 1/N for N=1" and fire a spurious warning
    // (e.g. when the only legal move is recapturing a piece out of check).
    if (moves.size > 1) {
        bool uniform = true;
        const float expected = 1.0f / static_cast<float>(moves.size);
        for (const auto& [move, prob] : priors) {
            if (std::abs(prob - expected) > 1e-4f) {
                uniform = false;
                break;
            }
        }
        if (uniform) {
            std::cout << "info string Warning: Uniform policy returned for "
                      << (pos.side_to_move() == White ? "White" : "Black")
                      << std::endl;
        }
    }

    for (const Move m : moves) {
        float p = priors.count(m) ? priors.at(m) : 0.0f;
        node->children.push_back(std::make_unique<Node>(m, node, p));
    }
    
    node->is_expanded = true;
}

float MCTS::evaluate_node(const Position& pos) {
    Score s = evaluate(pos);
    float eval = std::tanh(static_cast<float>(s) / 400.0f);
    
    // Rule 2: Dynamic Policy-Hijack / Q-boost for strong evaluations.
    // Instead of locking to 0.999, we apply a non-linear sigmoid-style boost
    // that increases visit priority for the move without terminating the search.
    if (s > 200) { // e.g., > 200cp advantage
        eval = std::min(0.99f, eval + 0.1f);
    }
    return eval;
}

void MCTS::backpropagate(Node* node, float value) {
    while (node) {
        node->N += 1.0f;
        node->W += value;
        node->Q = node->W / node->N;
        
        // Flip value for parent (alternate turns)
        value = -value;
        node = node->parent;
    }
}

}  // namespace eclipse::mcts
