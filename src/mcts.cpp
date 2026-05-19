// SPDX-License-Identifier: GPL-3.0-or-later
#include "mcts.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include "eval.hpp"
#include "movegen.hpp"
#include "policy.hpp"
#include "see.hpp"

namespace eclipse::mcts {

namespace {

// Logging cadence shared across workers. Time-based (1s) rather than node-based
// because at low NPS (a few hundred per second on a single CPU thread without
// SIMD) a 1024-node boundary takes 3-5s, which is useless for live monitoring.
// CAS dance so only one thread emits per second; others skip.
std::atomic<std::int64_t> g_last_info_ms{0};
constexpr std::int64_t kInfoIntervalMs = 1000;

// Snapshot of the top-K root children for an info line. K=3 keeps the line
// short enough to fit in a UCI client's typical info panel.
template <std::size_t K>
struct RootSnapshot {
    struct Entry { Move m; std::int32_t n; float q; float p; };
    Entry slots[K]{};
    std::size_t filled = 0;
};

template <std::size_t K>
RootSnapshot<K> snapshot_root_topk(const Node& root) {
    RootSnapshot<K> snap;
    for (const auto& child : root.children) {
        const auto n = child->N.load(std::memory_order_relaxed);
        // Insertion sort into the top-K by N.
        std::size_t pos = snap.filled;
        while (pos > 0 && snap.slots[pos - 1].n < n) {
            if (pos < K) snap.slots[pos] = snap.slots[pos - 1];
            --pos;
        }
        if (pos < K) {
            snap.slots[pos] = {child->move, n, child->Q(), child->P};
            if (snap.filled < K) ++snap.filled;
        }
    }
    return snap;
}

}  // namespace

Move MCTS::search() {
    run();
    return get_best_move();
}

void MCTS::run() {
    root = std::make_unique<Node>(MoveNone, nullptr, 1.0f);

    // Root expansion is single-threaded so workers see a populated child list
    // before they start traversing. Done under the (uncontested) root mutex
    // for symmetry with deeper expansion.
    {
        std::lock_guard<std::mutex> lock(root->expand_mutex);
        expand_under_lock(root.get(), root_pos);
    }

    const int threads = std::clamp(search_info.threads, 1, 128);
    g_last_info_ms.store(0, std::memory_order_relaxed);

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

    // Proactive mate-in-1 sweep. We can't trust MCTS visit counts here: under
    // tight visit budgets (test_search runs 4 iterations) or with weak priors,
    // the mating move may have been visited zero times. Instead we just play
    // every legal root move and check for immediate checkmate. Cost is
    // bounded - one movegen per legal move, ~microseconds each, far cheaper
    // than a single MCTS iteration's ONNX call.
    {
        MoveList moves;
        Position scratch = root_pos;
        generate_legal_moves(scratch, moves);
        for (const Move m : moves) {
            StateInfo st;
            Position after = root_pos;
            after.do_move(m, st);
            MoveList opp_replies;
            generate_legal_moves(after, opp_replies);
            if (opp_replies.size == 0 && after.in_check()) {
                // Opp has no legal reply and is in check -> checkmate by `m`.
                search_info.best_score = static_cast<Score>(kMateScore - 1);
                // Find the matching child for logging if it was expanded.
                Node* mating = nullptr;
                for (const auto& c : root->children) {
                    if (c->move == m) { mating = c.get(); break; }
                }
                if (mating) {
                    log_search_summary(*mating,
                        mating->N.load(std::memory_order_relaxed));
                } else {
                    std::cout << "info string mate-in-1 found by sweep: "
                              << m.to_uci() << " (MCTS did not visit)"
                              << std::endl;
                }
                search_info.best_move = m;
                return;
            }
        }
    }
}

void MCTS::adjust_root_q(Move m, Score s) {
    if (!root) return;
    for (auto& child : root->children) {
        if (child->move == m) {
            // Convert centipawns back to Q-value [-1, 1].
            // Score s is from our perspective. Child Q is from opponent's.
            // So we want child Q to be -s / 400.
            const float target_q = -static_cast<float>(s) / 400.0f;
            const auto n = child->N.load(std::memory_order_relaxed);
            if (n > 0) {
                // Adjust W such that W/N = target_q
                const int64_t target_w_fx = static_cast<int64_t>(target_q * n * kWScale);
                child->W_fx.store(target_w_fx, std::memory_order_relaxed);
                
                std::cout << "info string MCTS adjusted " << m.to_uci() 
                          << " Q to " << std::fixed << std::setprecision(3) << target_q
                          << " based on AB score " << s << "cp" << std::endl;
            }
            break;
        }
    }
}

Move MCTS::get_best_move() {
    if (search_info.best_move != MoveNone) return search_info.best_move;

    Node* best_child = nullptr;
    for (const auto& child : root->children) {
        if (!best_child) { best_child = child.get(); continue; }
        const auto cn = child->N.load(std::memory_order_relaxed);
        const auto bn = best_child->N.load(std::memory_order_relaxed);
        // Best move = most-visited root child. Tie-broken by Q.
        if (cn > bn || (cn == bn && child->Q() < best_child->Q())) {
            best_child = child.get();
        }
    }
    if (best_child) {
        search_info.best_score = static_cast<Score>(-best_child->Q() * 400.0f);
        log_search_summary(*best_child, best_child->N.load(std::memory_order_relaxed));
        return best_child->move;
    }
    return MoveNone;
}

void MCTS::log_search_summary(const Node& chosen, std::int32_t chosen_visits) const {
    // End-of-search diagnostic: top-5 root children sorted by visit count.
    // Lets the operator see whether the chosen move dominated by a wide
    // margin (engine confident) or barely beat alternatives (engine guessing).
    const auto snap = snapshot_root_topk<5>(*root);
    std::cout << "info string final root (chose " << chosen.move.to_uci()
              << " N=" << chosen_visits << "):";
    for (std::size_t i = 0; i < snap.filled; ++i) {
        const auto& e = snap.slots[i];
        std::cout << "  " << e.m.to_uci()
                  << " N=" << e.n
                  << " Q=" << std::fixed << std::setprecision(3) << e.q
                  << " P=" << std::fixed << std::setprecision(3) << e.p;
    }
    std::cout << std::endl;
}

void MCTS::worker_loop() {
    while (!search_info.time_up() &&
           !search_info.stop.load(std::memory_order_relaxed)) {
        iterate();
        const auto seen = search_info.nodes_searched.fetch_add(
            1, std::memory_order_relaxed) + 1;

        if (search_info.limits.depth > 0 &&
            seen >= search_info.limits.depth) {
            search_info.stop.store(true, std::memory_order_relaxed);
            break;
        }
        if (search_info.limits.nodes > 0 &&
            seen >= search_info.limits.nodes) {
            search_info.stop.store(true, std::memory_order_relaxed);
            break;
        }

        // Time-based info emission: at most one line per kInfoIntervalMs across
        // all workers. Sample only every 128 iterations to keep clock reads
        // cheap, then CAS-race to be the printer if enough time has passed.
        if ((seen & 0x7F) == 0) {
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - search_info.start_time).count();
            auto prev_ms = g_last_info_ms.load(std::memory_order_relaxed);
            if (elapsed_ms - prev_ms >= kInfoIntervalMs &&
                g_last_info_ms.compare_exchange_strong(
                    prev_ms, elapsed_ms, std::memory_order_relaxed)) {
                const auto snap = snapshot_root_topk<3>(*root);
                if (snap.filled > 0 && elapsed_ms > 0) {
                    // Main info line: nodes/nps/time/pv of best (most-visited).
                    // Negate the child Q: it is stored from the child's STM
                    // (our opponent's) perspective; UCI score is from us.
                    std::cout << "info nodes " << seen
                              << " nps "  << (seen * 1000 / elapsed_ms)
                              << " time " << elapsed_ms
                              << " score cp " << static_cast<int>(-snap.slots[0].q * 400.0f)
                              << " pv "   << snap.slots[0].m.to_uci()
                              << std::endl;
                    // Diagnostic line: top-3 root children with N / Q / prior.
                    // Cheap; only fires once per second.
                    std::cout << "info string root:";
                    for (std::size_t i = 0; i < snap.filled; ++i) {
                        const auto& e = snap.slots[i];
                        std::cout << "  " << e.m.to_uci()
                                  << " N=" << e.n
                                  << " Q=" << std::fixed << std::setprecision(3) << e.q
                                  << " P=" << std::fixed << std::setprecision(3) << e.p;
                    }
                    std::cout << std::endl;
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
        const float parent_q = node->Q();

        Node* best_child = nullptr;
        float best_score = -1e30f;
        for (const auto& child : node->children) {
            const float s = child->puct_score(parent_n, kCpuct, parent_q);
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

    float sum_p = 0.0f;
    std::vector<float> filtered_priors;
    filtered_priors.reserve(moves.size);

    for (const Move m : moves) {
        float p = priors.count(m) ? priors.at(m) : 0.0f;
        
        // Tactical filtering: if move loses material according to SEE, reduce prior.
        // We only do this for captures or if the side to move is not in check.
        if (!pos.in_check() && !see_ge(pos, m, 0)) {
            p *= 0.1f;
        }
        
        filtered_priors.push_back(p);
        sum_p += p;
    }

    // Re-normalize and create children.
    for (int i = 0; i < moves.size; ++i) {
        const float p = (sum_p > 0.0f) ? (filtered_priors[static_cast<std::size_t>(i)] / sum_p) : (1.0f / static_cast<float>(moves.size));
        node->children.push_back(std::make_unique<Node>(moves[static_cast<std::size_t>(i)], node, p));
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
