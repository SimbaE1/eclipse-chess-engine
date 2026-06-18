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
extern float g_select_visit_frac;  // visit-fraction gate for value-aware final move selection
extern float g_select_q_margin;    // Q margin below which selection prefers more visits
extern int   g_policy_depth;       // plies from root that get NNUE-scored policy priors

struct Node;

// Nodes are allocated from a pool (see mcts.cpp) rather than the general heap.
// At 4 threads, per-node malloc/free was contending in the system allocator as
// heavily as the NNUE inference itself (every expansion allocates one Node per
// legal move). NodePtr is a unique_ptr with a deleter that returns the slot to
// the pool, so all existing ownership/move semantics (tree reuse, std::move of
// subtrees) keep working unchanged.
struct NodeDeleter {
    void operator()(Node* n) const noexcept;
};
using NodePtr = std::unique_ptr<Node, NodeDeleter>;

// Construct a pooled Node. Drop-in replacement for std::make_unique<Node>.
NodePtr make_node(Move m, Node* parent, float prior) noexcept;

// One-byte spinlock used to serialize node expansion. Replaces a per-Node
// std::mutex (64 B on macOS) — every Node carries one, so shrinking it to a
// single byte nearly halves Node's footprint and packs the hot fields (N/W/P
// read by select() across millions of nodes) far more densely in cache. The
// expansion critical section is tiny and same-node contention is rare (virtual
// loss disperses workers), so a brief spin beats a kernel futex round-trip.
// Exposes lock()/unlock() so std::lock_guard / std::unique_lock work unchanged.
struct Spinlock {
    std::atomic<bool> locked{false};
    void lock() noexcept {
        for (;;) {
            if (!locked.exchange(true, std::memory_order_acquire)) return;
            while (locked.load(std::memory_order_relaxed)) {
#if defined(__i386__) || defined(__x86_64__)
                __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
                __asm__ __volatile__("yield");
#endif
            }
        }
    }
    void unlock() noexcept { locked.store(false, std::memory_order_release); }
};

struct Node {
    Move  move   = MoveNone;
    Node* parent = nullptr;
    std::vector<NodePtr> children;

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
    Spinlock          expand_mutex;

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
        // Use parent_n+1 so sqrt never collapses to 0 when the parent hasn't
        // been visited yet (e.g. first batch at root). Without this, all FPU
        // children score identically and the engine visits moves in list order
        // rather than by policy prior, producing nonsense on the first pass.
        return puct_score_fast(cpuct * std::sqrt(static_cast<float>(parent_n + 1)),
                               parent_q);
    }

    // Hot-path variant: the selection loop computes the exploration term's
    // `cpuct * sqrt(parent_n + 1)` ONCE per parent (it is identical for every
    // child) and passes it in here, so the sqrt is amortized across all
    // siblings instead of being recomputed per child. Bit-for-bit identical to
    // puct_score() above.
    float puct_score_fast(float explore_num, float parent_q) const noexcept {
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
        return -q + explore_num * P / (1.0f + static_cast<float>(n_eff));
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

    // keep_existing_root=true skips the tree-reuse/rebuild step and continues
    // iterating on whatever tree is already in `root` (used to extend a
    // search with more time on the SAME position, e.g. search.cpp's AB/MCTS
    // reconciliation loop). Only valid on a second+ call after run() has
    // already populated root at least once.
    void run(bool keep_existing_root = false);
    void save_to_cache();          // call after get_best_move() to enable tree reuse
    void adjust_root_q(Move m, Score s);
    Move get_best_move();
    // Returns the most-visited reply to `best_move` in the current tree.
    // Must be called before save_to_cache() (which nulls the root).
    Move get_ponder_move_after(Move best_move) const;

    // ---- Tactic-node seeding (used by search.cpp's AB-validation step) ----
    //
    // AB's iterative deepening can find a refutation several plies into a
    // line before MCTS's own policy priors would ever steer it there -- by
    // the time MCTS's NNUE-based judgment of the position after AB's move
    // matters, the critical node is buried deep enough that visits never
    // reach it. Two soft (not forced) nudges address that:
    //
    //  - set_bias_path(): at each depth along this move sequence (root=0),
    //    expand_under_lock multiplies that depth's matching child's prior so
    //    PUCT is more likely to actually walk down this line instead of
    //    wherever the policy net's untouched judgment would normally go.
    //  - set_value_seed(): the FIRST time a node matching this exact
    //    position (by Zobrist key) gets created, it starts with
    //    `virtual_visits` worth of `q` already backed in, instead of 0
    //    visits. This is a prior, not a terminal marking (see is_terminal
    //    elsewhere) -- real visits accumulated afterward average normally
    //    and can move the value away from the seed if deeper search there
    //    disagrees with AB's read.
    //
    // Both are no-ops unless explicitly set (empty path / key 0), so they
    // add no cost to normal root search.
    void set_bias_path(std::vector<Move> path) { bias_path_ = std::move(path); }
    void set_value_seed(std::uint64_t key, float q, int virtual_visits) {
        value_seed_key_    = key;
        value_seed_q_      = q;
        value_seed_visits_ = virtual_visits;
    }

private:
    void   worker_loop();
    // Collect up to `batch_size` leaves (selection + expansion, with virtual
    // loss separating them), evaluate them in one batched NNUE call, then
    // backprop all of them. Returns the number of leaves processed (== visits
    // added). Batching amortizes the L1 weight-matrix memory traffic — the
    // dominant NNUE cost — across several leaves per pass.
    int    iterate_batch(int batch_size);
    void   expand_under_lock(Node* node, Position& pos, int depth);
    float  evaluate_node(const Position& pos);
    void   log_search_summary(const Node& chosen, std::int32_t chosen_visits) const;

    // See set_bias_path()/set_value_seed() above.
    std::vector<Move> bias_path_;
    std::uint64_t      value_seed_key_    = 0;
    float              value_seed_q_      = 0.0f;
    int                value_seed_visits_ = 0;

    Position&   root_pos;
    SearchInfo& search_info;
    NodePtr root;
    std::atomic<int> max_depth_seen{0};
};

}  // namespace eclipse::mcts
