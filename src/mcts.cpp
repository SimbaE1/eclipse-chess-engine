// SPDX-License-Identifier: GPL-3.0-or-later
#include "mcts.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "eval.hpp"
#include "movegen.hpp"
#include "policy.hpp"

namespace eclipse::mcts {

namespace {

// Logging cadence shared across workers. Cheap atomic CAS dance to make sure
// only one thread emits an info line per 1024-iteration boundary; others skip.
std::atomic<std::int64_t> g_last_info_at{0};

}  // namespace

Move MCTS::search() {
    root = std::make_unique<Node>(MoveNone, nullptr, 1.0f);

    // Root expansion is single-threaded so workers see a populated child list
    // before they start traversing. Done under the (uncontested) root mutex
    // for symmetry with deeper expansion.
    {
        std::lock_guard<std::mutex> lock(root->expand_mutex);
        expand_under_lock(root.get(), root_pos);
    }

    const int threads = std::clamp(search_info.threads, 1, 128);
    g_last_info_at.store(0, std::memory_order_relaxed);

    // For threads=1 we stay on the calling thread to keep the codepath as
    // close to the previous behaviour as possible (useful for the test suite
    // and for triage when something looks off).
    if (threads == 1) {
        worker_loop();
    } else {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(threads));
        for (int i = 0; i < threads; ++i) {
            workers.emplace_back([this] { worker_loop(); });
        }
        for (auto& w : workers) w.join();
    }

    // Best move = most-visited root child. Visit count is the standard MCTS
    // signal because it incorporates both Q (a child gets more visits when
    // its Q is good) and prior (PUCT exploration tilts visits toward priors
    // early). Tie-broken by Q to avoid picking a fluky-high-prior move that
    // happened to share visits with a better one.
    Node* best_child = nullptr;
    for (const auto& child : root->children) {
        if (!best_child) { best_child = child.get(); continue; }
        const auto cn = child->N.load(std::memory_order_relaxed);
        const auto bn = best_child->N.load(std::memory_order_relaxed);
        if (cn > bn || (cn == bn && child->Q() > best_child->Q())) {
            best_child = child.get();
        }
    }
    if (best_child) {
        search_info.best_score = static_cast<Score>(best_child->Q() * 400.0f);
        return best_child->move;
    }
    return MoveNone;
}

void MCTS::worker_loop() {
    while (!search_info.time_up() &&
           !search_info.stop.load(std::memory_order_relaxed)) {
        iterate();
        const auto seen = search_info.nodes_searched.fetch_add(
            1, std::memory_order_relaxed) + 1;

        // Visit-count budget (UCI `go depth N` is reinterpreted as a visit
        // ceiling under MCTS; alpha-beta depth doesn't translate). Hit on
        // any worker stops everyone.
        if (search_info.limits.depth > 0 &&
            seen >= search_info.limits.depth) {
            search_info.stop.store(true, std::memory_order_relaxed);
            break;
        }

        // Emit an info line every ~1024 iterations. CAS so only one thread
        // does the printing per boundary, even with N workers racing.
        if ((seen & 0x3FF) == 0) {
            auto prev = g_last_info_at.load(std::memory_order_relaxed);
            if (seen > prev &&
                g_last_info_at.compare_exchange_strong(
                    prev, seen, std::memory_order_relaxed)) {
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - search_info.start_time).count();
                if (elapsed > 0) {
                    Node* bc = nullptr;
                    std::int32_t bn = -1;
                    for (const auto& child : root->children) {
                        const auto cn = child->N.load(std::memory_order_relaxed);
                        if (cn > bn) { bn = cn; bc = child.get(); }
                    }
                    if (bc) {
                        std::cout << "info nodes " << seen
                                  << " nps " << (seen * 1000 / elapsed)
                                  << " pv "   << bc->move.to_uci()
                                  << std::endl;
                    }
                }
            }
        }
    }
}

void MCTS::iterate() {
    Position pos = root_pos;
    StateInfo st;

    // Selection: walk the tree, apply virtual loss on each non-root node
    // visited so other workers see the path as already in flight. Record the
    // traversal so backprop can find its way home.
    std::vector<Node*> path;
    path.reserve(64);
    path.push_back(root.get());

    Node* node = root.get();
    while (node->is_expanded.load(std::memory_order_acquire) && !node->is_terminal) {
        // Snapshot parent N for the PUCT denominator. Workers update it
        // concurrently, so we read once and use that throughout the loop.
        const auto parent_n = node->N.load(std::memory_order_relaxed);

        Node* best_child = nullptr;
        float best_score = -1e30f;
        for (const auto& child : node->children) {
            const float s = child->puct_score(parent_n, kCpuct);
            if (s > best_score) {
                best_score = s;
                best_child = child.get();
            }
        }
        if (!best_child) break;

        best_child->apply_virtual_loss();
        pos.do_move(best_child->move, st);
        path.push_back(best_child);
        node = best_child;
    }

    // Expansion + evaluation. Only one thread expands a given node; others
    // wait on the mutex, then notice is_expanded is true and skip. The leaf's
    // evaluation runs after the lock is released so we don't serialize ONNX
    // calls on the same node's mutex when many threads land here.
    float value;
    {
        std::unique_lock<std::mutex> lock(node->expand_mutex);
        if (!node->is_expanded.load(std::memory_order_acquire) && !node->is_terminal) {
            expand_under_lock(node, pos);
        }
        // is_terminal is set inside expand_under_lock when there are no legal
        // moves. Snapshot it before unlocking - safe because expand happens
        // exactly once per node.
    }

    if (node->is_terminal) {
        // Terminal Q was written by expand_under_lock (-1 for in-check loss,
        // 0 for stalemate). Read after the lock release - it's frozen now.
        value = node->Q();
    } else {
        value = evaluate_node(pos);
    }

    // Backprop: undo virtual loss on each node we touched (except root, which
    // never had one applied), then bake in the real value. Sign flips at each
    // ply since the network reports score from the side-to-move's POV.
    const std::int64_t value_fx = static_cast<std::int64_t>(value * kWScale);
    std::int64_t v_fx = value_fx;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Node* n = *it;
        if (n != root.get()) {
            n->remove_virtual_loss();
        }
        n->N.fetch_add(1, std::memory_order_relaxed);
        n->W_fx.fetch_add(v_fx, std::memory_order_relaxed);
        v_fx = -v_fx;
    }
}

void MCTS::expand_under_lock(Node* node, const Position& pos) {
    if (node->is_expanded.load(std::memory_order_acquire)) return;  // double-checked

    MoveList moves;
    Position temp_pos = pos;
    generate_legal_moves(temp_pos, moves);

    if (moves.size == 0) {
        node->is_terminal = true;
        // Loss if in check (got mated), draw if stalemated. Stored in W so
        // Q() returns the right value to peers; N=1 means "evaluated".
        const float terminal_v = pos.in_check() ? -1.0f : 0.0f;
        node->N.store(1, std::memory_order_relaxed);
        node->W_fx.store(static_cast<std::int64_t>(terminal_v * kWScale),
                         std::memory_order_relaxed);
        node->is_expanded.store(true, std::memory_order_release);
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
        for (const auto& [m, prob] : priors) {
            if (std::abs(prob - expected) > 1e-4f) { uniform = false; break; }
        }
        if (uniform) {
            std::cout << "info string Warning: Uniform policy returned for "
                      << (pos.side_to_move() == White ? "White" : "Black")
                      << std::endl;
        }
    }

    for (const Move m : moves) {
        const float p = priors.count(m) ? priors.at(m) : 0.0f;
        node->children.push_back(std::make_unique<Node>(m, node, p));
    }

    // Release ordering ensures other threads that acquire-load is_expanded
    // see the fully-populated children vector before iterating it.
    node->is_expanded.store(true, std::memory_order_release);
}

float MCTS::evaluate_node(const Position& pos) {
    const Score s = evaluate(pos);
    float v = std::tanh(static_cast<float>(s) / 400.0f);

    // Dynamic Q-boost for strongly winning positions, so PUCT pushes more
    // visits toward the winning continuation without locking it in.
    if (s > 200) {
        v = std::min(0.99f, v + 0.1f);
    }
    return v;
}

}  // namespace eclipse::mcts
