// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "move.hpp"
#include "position.hpp"
#include "search.hpp"

namespace eclipse::mcts {

// ---------------------------------------------------------------------------
// MCTS transposition table — a flat position-value cache separate from the AB
// TT.  Maps Zobrist key -> {N, Q} so that positions reached via different
// paths within the same search (or revisited in subsequent moves) inherit
// already-accumulated visit statistics rather than starting cold.
//
// Thread-safety: intentionally racy (plain struct reads/writes).  A torn
// read/write occasionally produces a stale Q for one MCTS iteration, which
// the tree corrects immediately through further visits.  Locking every probe
// would cost more than the collision rate justifies.
// ---------------------------------------------------------------------------

struct MCTSEntry {
    std::uint64_t key = 0;
    std::int32_t  N   = 0;
    float         Q   = 0.0f;
};

class MCTSTable {
public:
    MCTSTable() = default;
    explicit MCTSTable(std::size_t mb_size) { resize(mb_size); }

    void resize(std::size_t mb_size);
    void clear();

    // Returns true and fills `out` if `key` matches the stored entry.
    bool probe(std::uint64_t key, MCTSEntry& out) const;

    // Always writes; prefers higher-N entries when key matches.
    void store(std::uint64_t key, std::int32_t N, float Q);

private:
    std::vector<MCTSEntry> table_;
    std::size_t            mask_ = 0;
};

extern MCTSTable g_mcts_tt;

// Minimum visit count in a TT entry before we trust it over a fresh NNUE eval.
inline constexpr std::int32_t kMCTSTTMinN = 4;

// W is stored as a fixed-point int64 (multiply real value by kWScale) so we
// can fetch_add it atomically without relying on the still-uneven compiler
// support for std::atomic<float>::fetch_add. 2^20 keeps us comfortably away
// from int64 overflow even after 2^40 visits of values in [-1, 1].
inline constexpr int     kWShift = 20;
inline constexpr int64_t kWScale = 1LL << kWShift;

// PUCT search constants — exposed at runtime so the UCI client can A/B them
// without rebuilding. Defaults match the values we shipped before; both are
// in the typical AlphaZero range. Read by every puct_score() call, so kept
// as float globals rather than going through a SearchInfo pointer.
extern float g_cpuct;        // exploration coefficient
extern float g_fpu_offset;   // First-Play Urgency discount on parent_q for unvisited children

struct Node {
    Move  move   = MoveNone;
    Node* parent = nullptr;
    std::vector<std::unique_ptr<Node>> children;

    // -- Lockless state ------------------------------------------------------
    // N, W, virtual_loss are mutated by backpropagate() and read by select()
    // from many worker threads. Relaxed ordering is enough: PUCT scores are
    // approximate anyway, and we never read N and W as a transactional pair.
    std::atomic<std::int32_t> N{0};
    std::atomic<std::int64_t> W_fx{0};      // W * kWScale
    std::atomic<std::int32_t> virtual_loss{0};

    float P = 0.0f;                          // policy prior - written once, then read-only

    // is_expanded gates safe reads of `children`. The expanding thread sets it
    // with release ordering at the end of expansion; readers acquire-load it
    // before iterating children. expand_mutex serializes the expansion itself
    // so children is never observed half-built.
    std::atomic<bool> is_expanded{false};
    bool              is_terminal = false;  // set under expand_mutex, then frozen
    std::mutex        expand_mutex;

    Node(Move m, Node* p, float prior) noexcept : move(m), parent(p), P(prior) {}

    // -- Derived getters -----------------------------------------------------
    float W() const noexcept {
        return static_cast<float>(W_fx.load(std::memory_order_relaxed)) /
               static_cast<float>(kWScale);
    }

    float Q() const noexcept {
        const auto n = N.load(std::memory_order_relaxed);
        return n > 0 ? W() / static_cast<float>(n) : 0.0f;
    }

    // PUCT with virtual loss applied. Treats each outstanding virtual_loss
    // as a visit that returned -1.0 from this child's perspective, which is
    // what dissuades other workers from selecting the same path while it's
    // in flight.
    //
    // Sign convention: Q() returns the child's W/N from the CHILD's STM
    // perspective (= the parent's opponent). When the parent picks among its
    // children it wants to MAXIMIZE its own expected outcome, which is the
    // NEGATION of the child's. Hence the `-q` in the formula. Forgetting this
    // negation makes the engine play moves that maximize the opponent's
    // expected outcome — for example, refusing to capture a hanging piece
    // because doing so makes the opponent's Q drop sharply.
    float puct_score(std::int32_t parent_n, float cpuct, float parent_q) const noexcept {
        const auto n  = N.load(std::memory_order_relaxed);
        const auto vl = virtual_loss.load(std::memory_order_relaxed);
        const auto n_eff = n + vl;

        float q;
        if (n_eff == 0) {
            // First Play Urgency (FPU): discount the parent's Q.
            // parent_q is the value of the parent position from the perspective
            // of the player who just moved to reach this node.
            // We subtract a small discount to favor exploration of unvisited nodes
            // without being reckless. Tunable via UCI `FpuOffset`.
            q = parent_q - g_fpu_offset;
        } else {
            // Virtual loss makes an in-flight node look maximally good for its
            // OWN side-to-move (W pushed toward +n_eff), hence maximally bad for
            // the parent picking it (-q pushed toward -1). That repels other
            // selectors — threads in flight, and successive paths within one
            // batch where the real backprop (and parent_n bump) is deferred
            // until the whole batch is evaluated. Adding (not subtracting) vl is
            // what makes that repulsion point the right way.
            const float w_eff = W() + static_cast<float>(vl);
            q = w_eff / static_cast<float>(n_eff);
        }
        return -q + cpuct * P * std::sqrt(static_cast<float>(parent_n)) /
                               (1.0f + static_cast<float>(n_eff));
    }

    void apply_virtual_loss() noexcept {
        virtual_loss.fetch_add(1, std::memory_order_relaxed);
    }
    void remove_virtual_loss() noexcept {
        virtual_loss.fetch_sub(1, std::memory_order_relaxed);
    }
};

class MCTS {
public:
    MCTS(Position& pos, SearchInfo& info) : root_pos(pos), search_info(info) {}

    Move search(); // deprecated/wrapper

    void run();
    void save_to_cache();          // call after get_best_move() to enable tree reuse
    void adjust_root_q(Move m, Score s);
    Move get_best_move();

private:
    void   worker_loop();
    // Collect up to `batch_size` leaves (selection + expansion, with virtual
    // loss separating them), evaluate them in one batched NNUE call, then
    // backprop all of them. Returns the number of leaves processed (== visits
    // added). Batching amortizes the L1 weight-matrix memory traffic — the
    // dominant NNUE cost — across several leaves per pass.
    int    iterate_batch(int batch_size);
    void   expand_under_lock(Node* node, const Position& pos);
    float  evaluate_node(const Position& pos);
    void   log_search_summary(const Node& chosen, std::int32_t chosen_visits) const;

    Position&   root_pos;
    SearchInfo& search_info;
    std::unique_ptr<Node> root;
};

}  // namespace eclipse::mcts
