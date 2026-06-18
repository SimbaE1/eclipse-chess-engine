// SPDX-License-Identifier: GPL-3.0-or-later
#include "mcts.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include <new>

#include "bitboard.hpp"
#include "eval.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "see.hpp"
#include "syzygy.hpp"

namespace eclipse::mcts {

// ---------------------------------------------------------------------------
// Node pool
//
// Every MCTS expansion allocates one Node per legal move (~35). With the system
// allocator this was, at 4 threads, contending in nanov2 as heavily as the
// whole NNUE forward pass — the per-Node malloc/free lock was the dominant
// limiter on multi-thread scaling.
//
// Key invariant that makes a simple pool correct here: during the parallel
// search the tree only GROWS (workers only allocate); Nodes are freed only at
// teardown between moves, which runs single-threaded on the UCI dispatch thread
// (root reassignment, save_to_cache, cached-tree replacement). So:
//   * allocation (hot, multi-threaded) is served from a per-thread magazine and
//     only touches the global lock once per kMagBatch allocations, and
//   * deallocation (cold, single-threaded) pushes straight to the global free
//     list under an uncontended lock.
// Freed slots are recycled across searches, so steady-state memory is bounded
// by the peak live node count, not the cumulative allocation count.
//
// The free list is intrusive: a free slot's storage holds the `next` pointer
// (safe because the Node has been destroyed and sizeof(Node) >> sizeof(void*)).
// ---------------------------------------------------------------------------
namespace {

class NodePool {
public:
    Node* alloc() {
        Mag& mag = magazine();
        if (mag.count == 0) refill(mag);
        return mag.slots[--mag.count];
    }

    void free(Node* n) noexcept {
        std::lock_guard<std::mutex> lk(mutex_);
        push_locked(n);
    }

    // Return a thread's leftover magazine slots to the global free list. Called
    // from the magazine's destructor when a worker thread exits, so the slots a
    // worker pre-fetched but never used are not lost when its thread joins.
    void drain(Node** slots, int count) noexcept {
        if (count <= 0) return;
        std::lock_guard<std::mutex> lk(mutex_);
        for (int i = 0; i < count; ++i) push_locked(slots[i]);
    }

private:
    static constexpr int         kMagBatch = 256;    // slots fetched per refill
    static constexpr std::size_t kSlab     = 8192;   // Nodes per slab grow

    struct Mag {
        Node*    slots[kMagBatch];
        int      count = 0;
        NodePool* pool = nullptr;
        ~Mag() { if (pool) pool->drain(slots, count); }
    };

    static Mag& magazine() {
        static thread_local Mag mag;
        mag.pool = &instance();
        return mag;
    }

    // free_head_ is guarded by mutex_; intrusive next stored in the slot.
    static Node*& next_of(Node* n) noexcept {
        return *reinterpret_cast<Node**>(n);
    }
    void push_locked(Node* n) noexcept {
        next_of(n) = free_head_;
        free_head_ = n;
    }

    // Fill `mag` with up to kMagBatch slots. Recycled nodes (freed at a previous
    // teardown) are drained from the intrusive free list first; the rest are
    // bump-allocated as a contiguous run from the current slab. Both paths are
    // O(slots) with no per-node pointer chasing through cold memory while the
    // lock is held — the slab grow itself is O(1) (one allocation + a pointer
    // reset), which is what keeps the critical section short under 4 threads.
    void refill(Mag& mag) {
        std::lock_guard<std::mutex> lk(mutex_);
        int got = 0;
        while (got < kMagBatch && free_head_) {
            Node* n = free_head_;
            free_head_ = next_of(n);
            mag.slots[got++] = n;
        }
        while (got < kMagBatch) {
            if (slab_used_ == kSlab) new_slab_locked();
            mag.slots[got++] = reinterpret_cast<Node*>(
                slab_base_ + (slab_used_++) * sizeof(Node));
        }
        mag.count = got;
    }

    void new_slab_locked() {
        void* raw = ::operator new(kSlab * sizeof(Node),
                                   std::align_val_t{alignof(Node)});
        slabs_.push_back(raw);
        slab_base_ = static_cast<std::byte*>(raw);
        slab_used_ = 0;
    }

public:
    static NodePool& instance() {
        // Deliberately leaked: thread_local magazine destructors can run during
        // process-exit static teardown and call drain() on the pool. A normal
        // function-local static could be destroyed first (unspecified order),
        // leaving them to lock a dead mutex. Leaking sidesteps the fiasco; the
        // OS reclaims the slabs at exit anyway.
        static NodePool* pool = new NodePool();
        return *pool;
    }

private:
    std::mutex          mutex_;
    Node*               free_head_ = nullptr;     // recycled nodes (intrusive list)
    std::byte*          slab_base_ = nullptr;     // current bump slab
    std::size_t         slab_used_ = kSlab;       // forces a slab alloc on first use
    std::vector<void*>  slabs_;  // owned raw storage; freed at process exit
};

}  // namespace

NodePtr make_node(Move m, Node* parent, float prior) noexcept {
    Node* raw = NodePool::instance().alloc();
    return NodePtr(::new (raw) Node(m, parent, prior));
}

void NodeDeleter::operator()(Node* n) const noexcept {
    if (!n) return;
    n->~Node();
    NodePool::instance().free(n);
}

// ---------------------------------------------------------------------------
// MCTSTable implementation
// ---------------------------------------------------------------------------

MCTSTable g_mcts_tt(8);  // 8 MB default

void MCTSTable::resize(std::size_t mb_size) {
    std::size_t n = (mb_size * 1024 * 1024) / sizeof(MCTSEntry);
    std::size_t p2 = 1;
    while (p2 * 2 <= n) p2 *= 2;
    table_.assign(p2, MCTSEntry{});
    mask_ = p2 - 1;
}

void MCTSTable::clear() {
    std::fill(table_.begin(), table_.end(), MCTSEntry{});
}

bool MCTSTable::probe(std::uint64_t key, MCTSEntry& out) const {
    if (table_.empty()) return false;
    const MCTSEntry& e = table_[key & mask_];
    if (e.key != key || e.N <= 0) return false;
    out = e;
    return true;
}

void MCTSTable::store(std::uint64_t key, std::int32_t N, float Q) {
    if (table_.empty()) return;
    MCTSEntry& e = table_[key & mask_];
    if (e.key != key || N > e.N) {
        e.key = key;
        e.N   = N;
        e.Q   = Q;
    }
}

// PUCT runtime knobs. Tunable via UCI Cpuct / FpuOffset. Cpuct is bumped above
// the old 1.41 to widen exploration: the engine had been pouring ~98% of root
// visits into one early-favored move and missing the refutation of it. FpuOffset
// is the First-Play-Urgency discount on a parent's Q for its unvisited children;
// a smaller value makes untried moves look less pessimistic, so more of them get
// at least one look.
float g_cpuct      = 1.70f;
float g_fpu_offset = 0.20f;

// Final root-move selection (UCI SelectVisitFrac / SelectQMargin). Among root
// children with at least g_select_visit_frac * max_visits visits, pick the one
// with the best value for us (lowest child Q) instead of blindly the most
// visited. The visit filter keeps Q trustworthy; this only overrides raw visit
// count when a well-explored sibling is genuinely better, so it is a safety net
// against a move whose value collapsed late while it still led on visits.
// frac >= 1.0 reduces to pure most-visited selection.
float g_select_visit_frac = 0.60f;
float g_select_q_margin   = 0.02f;  // Q units (~6 cp); near-ties go to more visits

// NNUE-informed policy priors (UCI PolicyDepth). At expansions within this many
// plies of the root, score every child with the value net (one batched forward
// pass) and softmax the results into priors, instead of the cheap-but-tactically
// blind MVV-LVA/SEE heuristic. This is where the engine decides which moves are
// worth investigating, so accuracy near the top matters most; it is depth-gated
// because doing it at every node would cut NPS hard. -1 disables (heuristic
// everywhere); 0 = root only. Default 2 covers the root, its replies, and their
// replies — enough to surface tactical refutations of root moves — at only ~2%
// NPS cost; the cost explodes past depth 2 as the node count grows ~35x/ply.
int   g_policy_depth = 2;

// MCTS leaf-batch size. Each worker collects this many leaves (kept apart by
// virtual loss), evaluates them in a single nnue::evaluate_batch call, and
// backprops them together. The batched L1 kernel reuses each weight row across
// 4 inputs, so a multiple of 4 fully amortizes the (memory-bound) weight
// traffic; 8 also amortizes the per-call FT/L2/L3 fixed costs while keeping
// in-flight staleness negligible against a multi-hundred-thousand-visit search.
inline constexpr int kLeafBatch    = 8;
inline constexpr int kMaxPathLen   = 256;

// Tree reuse cache: survives across MCTS instances so the next search can pick
// up where the last one left off. Saved by save_to_cache(), consumed by
// try_find_subtree(). Guarded by single-threaded UCI dispatch.
static NodePtr                   s_cached_root;
static std::unique_ptr<Position> s_cached_pos;

// Walk the cached tree up to 2 plies to find the subtree rooted at `target`.
// Returns the matching node (with parent nulled) or nullptr.
static NodePtr try_find_subtree(const Position& target_pos) {
    if (!s_cached_root || !s_cached_pos) return nullptr;
    const std::uint64_t target = target_pos.key();

    Position work = *s_cached_pos;  // mutable copy to replay moves

    for (auto& c1 : s_cached_root->children) {
        if (!c1) continue;
        StateInfo st1;
        work.do_move(c1->move, st1);

        if (work.key() == target) {
            // 1-ply match: opponent played our predicted best move.
            c1->parent = nullptr;
            return std::move(c1);
        }

        // 2-ply: we played c1, opponent played c2.
        if (c1->is_expanded.load(std::memory_order_acquire)) {
            for (auto& c2 : c1->children) {
                if (!c2) continue;
                StateInfo st2;
                work.do_move(c2->move, st2);
                if (work.key() == target) {
                    c2->parent = nullptr;
                    return std::move(c2);
                }
                work.undo_move(c2->move, st2);
            }
        }

        work.undo_move(c1->move, st1);
    }
    return nullptr;
}

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

void MCTS::run(bool keep_existing_root) {
    if (!keep_existing_root || !root) {
        // Try to reuse the subtree from the previous search. If the opponent
        // played the move we predicted (or we're pondering the same position),
        // we inherit all accumulated visits instead of rebuilding from scratch.
        NodePtr reused = try_find_subtree(root_pos);
        if (reused) {
            root = std::move(reused);
            const auto cached_n = root->N.load(std::memory_order_relaxed);
            std::cout << "info string tree-reuse: " << cached_n
                      << " cached visits inherited" << std::endl;
        } else {
            root = make_node(MoveNone, nullptr, 1.0f);
            // Root expansion is single-threaded so workers see a populated child
            // list before they start traversing. Done under the (uncontested) root
            // mutex for symmetry with deeper expansion.
            {
                std::lock_guard<std::mutex> lock(root->expand_mutex);
                expand_under_lock(root.get(), root_pos, 0);
            }
        }
    }

    // Root policy refinement: evaluate all root children via NNUE and replace
    // the heuristic priors with softmax(score / temperature). Runs even on
    // reused trees to keep priors fresh (root position hasn't changed, so this
    // is idempotent). ~2ms one-time cost at root depth only.
    if (!root->is_terminal && !root->children.empty()) {
        static constexpr float kPolicyTemp = 150.0f;  // cp temperature for softmax
        const std::size_t n = std::min(root->children.size(), std::size_t{256});
        float scores[256];
        float max_s = -1e30f;
        Position policy_pos = root_pos;  // work on a copy so root_pos is never mutated
        for (std::size_t i = 0; i < n; ++i) {
            StateInfo st;
            policy_pos.do_move(root->children[i]->move, st);
            scores[i] = -static_cast<float>(evaluate(policy_pos));
            policy_pos.undo_move(root->children[i]->move, st);
            if (scores[i] > max_s) max_s = scores[i];
        }
        // Stable softmax: subtract max before exp to prevent overflow.
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            scores[i] = std::exp((scores[i] - max_s) / kPolicyTemp);
            sum += scores[i];
        }
        if (sum > 1e-6f) {
            for (std::size_t i = 0; i < n; ++i) {
                root->children[i]->P = scores[i] / sum;
            }
        }
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

void MCTS::save_to_cache() {
    if (root) {
        s_cached_pos  = std::make_unique<Position>(root_pos);
        s_cached_root = std::move(root);
        // root is now null — MCTS must not be used after this call.
    }
}

void MCTS::adjust_root_q(Move m, Score s) {
    if (!root) return;
    for (auto& child : root->children) {
        if (child->move == m) {
            // Convert centipawns back to Q-value [-1, 1].
            // Score s is from our perspective. Child Q is from opponent's.
            // So we want child Q to be -s / 400.
            const float target_q = -static_cast<float>(s) / nnue::output_cp_per_unit();
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

    std::int32_t max_n = 0;
    for (const auto& child : root->children)
        max_n = std::max(max_n, child->N.load(std::memory_order_relaxed));
    const std::int32_t thresh = static_cast<std::int32_t>(
        g_select_visit_frac * static_cast<float>(max_n));

    Node* best_child = nullptr;
    for (const auto& child : root->children) {
        const auto cn = child->N.load(std::memory_order_relaxed);
        // Only "seriously explored" children compete on value; this keeps Q
        // trustworthy and reduces to most-visited when frac is high.
        if (cn == 0 || cn < thresh) continue;
        if (!best_child) { best_child = child.get(); continue; }
        // Lower child Q == better for us (child Q is from the opponent's POV).
        // Prefer clearly-better value; on a near-tie prefer the more-visited.
        const float cq = child->Q();
        const float bq = best_child->Q();
        const auto  bn = best_child->N.load(std::memory_order_relaxed);
        if (cq < bq - g_select_q_margin ||
            (cq <= bq + g_select_q_margin && cn > bn)) {
            best_child = child.get();
        }
    }
    if (best_child) {
        search_info.best_score = static_cast<Score>(-best_child->Q() * nnue::output_cp_per_unit());
        log_search_summary(*best_child, best_child->N.load(std::memory_order_relaxed));
        return best_child->move;
    }
    return MoveNone;
}

Move MCTS::get_ponder_move_after(Move best_move) const {
    if (!root) return MoveNone;
    for (const auto& child : root->children) {
        if (child->move != best_move) continue;
        if (!child->is_expanded.load(std::memory_order_acquire)) return MoveNone;
        Node* best_gc = nullptr;
        for (const auto& gc : child->children) {
            if (!best_gc ||
                gc->N.load(std::memory_order_relaxed) > best_gc->N.load(std::memory_order_relaxed))
                best_gc = gc.get();
        }
        if (!best_gc || best_gc->N.load(std::memory_order_relaxed) == 0) return MoveNone;
        return best_gc->move;
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
    // Always do at least one batch so go/quit piped back-to-back (or any case
    // where stop is set before the thread starts) still produces real MCTS
    // output instead of N=0 for every child.
    bool first_iter = true;
    while (first_iter ||
           (!search_info.time_up() &&
            !search_info.stop.load(std::memory_order_relaxed))) {
        first_iter = false;
        const int processed = iterate_batch(kLeafBatch);
        if (processed <= 0) break;  // root terminal / nothing to do
        const auto seen = search_info.nodes_searched.fetch_add(
            processed, std::memory_order_relaxed) + processed;

        if (search_info.limits.depth > 0 &&
            seen >= search_info.limits.depth) {
            search_info.stop.store(true, std::memory_order_relaxed);
            break;
        }
        if (search_info.limits.nodes > 0 &&
            static_cast<std::uint64_t>(seen) >= search_info.limits.nodes) {
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
                if (snap.filled > 0 && elapsed_ms > 0 && !search_info.silent) {
                    // Main info line: nodes/nps/time/pv of best (most-visited).
                    // Negate the child Q: it is stored from the child's STM
                    // (our opponent's) perspective; UCI score is from us.
                    const int seldepth = max_depth_seen.load(std::memory_order_relaxed);
                    std::cout << "info depth " << seldepth
                              << " seldepth " << seldepth
                              << " nodes " << seen
                              << " nps "  << (seen * 1000 / elapsed_ms)
                              << " time " << elapsed_ms
                              << " score cp " << static_cast<int>(-snap.slots[0].q * nnue::output_cp_per_unit())
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

int MCTS::iterate_batch(int batch_size) {
    if (batch_size < 1) batch_size = 1;
    if (batch_size > kLeafBatch) batch_size = kLeafBatch;

    // Per-collected-leaf bookkeeping. One entry per selected path.
    struct Leaf {
        Node*         path[kMaxPathLen];
        int           path_len   = 0;
        bool          needs_eval = false;  // true => value comes from the batch eval
        bool          is_rep     = false;  // true => draw by repetition; don't cache
        float         value      = 0.0f;   // terminal/repetition value when !needs_eval
        int           eval_idx   = -1;     // slot into the batch-eval input arrays
        std::uint64_t leaf_key   = 0;      // Zobrist key of the leaf position
    };

    // Thread-local scratch so nothing large lands on the (512 KB on macOS)
    // worker stack — the previous batch attempt SIGABRT'd from a 1 MB
    // Accumulator[256] stack array. Sized once; reused every call.
    static thread_local std::vector<Leaf>              leaves;
    static thread_local std::vector<nnue::Accumulator> batch_accs;  // eval inputs
    static thread_local std::vector<Color>             batch_stms;
    static thread_local std::vector<Score>             batch_scores;
    static thread_local std::vector<std::uint64_t>     keys;        // per-path rep keys

    if (static_cast<int>(leaves.size()) < batch_size) {
        leaves.resize(static_cast<std::size_t>(batch_size));
        batch_accs.resize(static_cast<std::size_t>(batch_size));
        batch_stms.resize(static_cast<std::size_t>(batch_size));
        batch_scores.resize(static_cast<std::size_t>(batch_size));
    }
    if (keys.size() < static_cast<std::size_t>(kMaxPathLen)) {
        keys.resize(static_cast<std::size_t>(kMaxPathLen));
    }

    Position  pos;        // reused per path (reset from root_pos each time)
    StateInfo st;
    int       n_leaves = 0;  // paths actually collected this batch
    int       n_eval   = 0;  // subset that need a network eval

    // ---- Selection + expansion: collect `batch_size` leaves --------------
    // Virtual loss is applied along each path as we descend and is NOT removed
    // until backprop, so successive paths in this batch (and other workers) are
    // steered away from leaves already in flight.
    for (int b = 0; b < batch_size; ++b) {
        Leaf& leaf = leaves[static_cast<std::size_t>(b)];
        leaf.needs_eval = false;
        leaf.eval_idx   = -1;

        pos = root_pos;
        int  path_len = 0;
        leaf.path[path_len] = root.get();
        keys[static_cast<std::size_t>(path_len)] = root_pos.key();
        path_len++;

        Node* node = root.get();
        bool  is_repetition = false;
        while (node->is_expanded.load(std::memory_order_acquire) && !node->is_terminal) {
            const auto  parent_n = node->N.load(std::memory_order_relaxed);
            const float parent_q = node->Q();
            // sqrt(parent_n+1) is identical for every child; hoist it (and the
            // cpuct multiply) out of the sibling loop so the sqrt runs once per
            // parent instead of once per child.
            const float explore_num = g_cpuct * std::sqrt(static_cast<float>(parent_n + 1));

            Node* best_child = nullptr;
            float best_score = -1e30f;
            for (const auto& child : node->children) {
                const float s = child->puct_score_fast(explore_num, parent_q);
                if (s > best_score) {
                    best_score = s;
                    best_child = child.get();
                }
            }
            if (!best_child) break;

            best_child->apply_virtual_loss();
            // Descent never undoes (each path replays from a fresh root copy),
            // so skip the 4 KB accumulator snapshot do_move would otherwise take.
            pos.do_move(best_child->move, st, /*snapshot_acc=*/false);
            const std::uint64_t new_key = pos.key();

            // 2-fold repetition: check every 2 steps back (same side to move).
            for (int j = path_len - 2; j >= 0; j -= 2) {
                if (keys[static_cast<std::size_t>(j)] == new_key) {
                    is_repetition = true;
                    break;
                }
            }

            if (path_len < kMaxPathLen) {
                leaf.path[path_len] = best_child;
                keys[static_cast<std::size_t>(path_len)] = new_key;
                path_len++;
            }
            node = best_child;
            if (is_repetition) break;
        }
        leaf.path_len = path_len;
        leaf.leaf_key = keys[static_cast<std::size_t>(path_len - 1)];

        // Track deepest leaf reached (path_len counts root as 1, so depth = path_len - 1).
        const int leaf_depth = path_len - 1;
        int prev = max_depth_seen.load(std::memory_order_relaxed);
        while (leaf_depth > prev &&
               !max_depth_seen.compare_exchange_weak(prev, leaf_depth,
                   std::memory_order_relaxed, std::memory_order_relaxed)) {}


        if (is_repetition) {
            leaf.value  = 0.0f;  // draw
            leaf.is_rep = true;  // don't cache — value is history-dependent
        } else {
            {
                std::unique_lock<std::mutex> lock(node->expand_mutex);
                if (!node->is_expanded.load(std::memory_order_acquire) && !node->is_terminal) {
                    // depth of the leaf being expanded: root is path index 0.
                    expand_under_lock(node, pos, path_len - 1);
                }
            }
            if (node->is_terminal) {
                leaf.value = node->Q();
            } else {
                // Probe the MCTS TT first: if we've visited this position
                // enough times before (same or different path), reuse the
                // cached Q rather than spending an NNUE call.
                MCTSEntry tt_hit;
                if (g_mcts_tt.probe(leaf.leaf_key, tt_hit) &&
                        tt_hit.N >= kMCTSTTMinN) {
                    leaf.value = tt_hit.Q;
                } else {
                    // Queue for the batched NNUE eval. The accumulator is kept
                    // in sync incrementally by do_move, so the leaf position
                    // already carries a valid one — copy it out (pos is reused
                    // next path). Cold path (not yet computed, e.g. net loaded
                    // mid-game): fall back to the single-eval path which
                    // refreshes on demand.
                    nnue::Accumulator& acc = pos.accumulator();
                    if (acc.computed) {
                        leaf.needs_eval = true;
                        leaf.eval_idx   = n_eval;
                        batch_accs[static_cast<std::size_t>(n_eval)] = acc;
                        batch_stms[static_cast<std::size_t>(n_eval)] = pos.side_to_move();
                        ++n_eval;
                    } else {
                        leaf.value = evaluate_node(pos);
                    }
                }
            }
        }
        ++n_leaves;
    }

    // ---- One batched NNUE forward pass for all queued leaves --------------
    if (n_eval > 0) {
        nnue::evaluate_batch(batch_accs.data(), batch_stms.data(),
                             batch_scores.data(), n_eval);
    }

    // ---- Backprop every collected path -----------------------------------
    for (int b = 0; b < n_leaves; ++b) {
        Leaf& leaf = leaves[static_cast<std::size_t>(b)];
        float value;
        if (leaf.needs_eval) {
            const Score s = batch_scores[static_cast<std::size_t>(leaf.eval_idx)];
            value = std::tanh(static_cast<float>(s) / nnue::output_cp_per_unit());
        } else {
            value = leaf.value;
        }

        // Undo virtual loss on each touched node (root never had one), then
        // bake in the real value. Sign flips each ply: the network reports
        // from the side-to-move's POV.
        std::int64_t v_fx = static_cast<std::int64_t>(value * kWScale);
        for (int i = leaf.path_len - 1; i >= 0; --i) {
            Node* n = leaf.path[i];
            if (n != root.get()) {
                n->remove_virtual_loss();
            }
            n->N.fetch_add(1, std::memory_order_relaxed);
            n->W_fx.fetch_add(v_fx, std::memory_order_relaxed);
            v_fx = -v_fx;

            // Write the leaf node into the MCTS TT so future paths to the
            // same position inherit its accumulated stats.  Skip repetitions
            // (value is history-dependent, not positional) and skip the root
            // (key==0 bucket would be poisoned).  A higher-N existing entry
            // is kept by MCTSTable::store.
            if (i == leaf.path_len - 1 && !leaf.is_rep && leaf.leaf_key != 0) {
                g_mcts_tt.store(leaf.leaf_key,
                                n->N.load(std::memory_order_relaxed),
                                n->Q());
            }
        }
    }

    return n_leaves;
}

void MCTS::expand_under_lock(Node* node, Position& pos, int depth) {
    if (node->is_expanded.load(std::memory_order_acquire)) return;  // double-checked

    // generate_legal_moves do/undo-moves in place and restores `pos` exactly
    // (board, key, AND accumulator — legality testing runs with update_acc off),
    // so we can run it directly on the caller's position instead of copying it.
    // `pos` is always the worker's thread-local descent position, so this
    // in-place mutate-then-restore is race-free. Saves a ~4.5 KB Position copy
    // (dominated by the NNUE accumulator) on every expansion.
    MoveList moves;
    generate_legal_moves(pos, moves);

    // 50-move rule: if halfmove clock reached 100, it's a draw regardless
    // of whether there are legal moves (unless the position is checkmate,
    // which takes precedence — but checkmate resets the clock anyway).
    if (pos.halfmove_clock() >= 100) {
        node->is_terminal = true;
        node->N.store(1, std::memory_order_relaxed);
        node->W_fx.store(0, std::memory_order_relaxed);  // draw
        node->is_expanded.store(true, std::memory_order_release);
        return;
    }

    // Syzygy tablebase: mark TB positions as terminal leaves so MCTS
    // propagates accurate WDL values instead of falling back to NNUE.
    if (syzygy::is_enabled() && pos.castling_rights() == NoCastling &&
        static_cast<unsigned>(popcount(pos.occupied())) <= syzygy::max_pieces()) {
        const unsigned wdl = syzygy::probe_wdl(pos);
        if (wdl != syzygy::kTbFailed) {
            const float tb_val = (wdl == syzygy::kTbWin)  ?  1.0f
                               : (wdl == syzygy::kTbLoss) ? -1.0f : 0.0f;
            node->is_terminal = true;
            node->N.store(1, std::memory_order_relaxed);
            node->W_fx.store(static_cast<std::int64_t>(tb_val * kWScale),
                             std::memory_order_relaxed);
            node->is_expanded.store(true, std::memory_order_release);
            return;
        }
    }

    if (moves.size == 0) {
        node->is_terminal = true;
        // Loss if in check (got mated), draw if stalemated.
        const float terminal_v = pos.in_check() ? -1.0f : 0.0f;
        node->N.store(1, std::memory_order_relaxed);
        node->W_fx.store(static_cast<std::int64_t>(terminal_v * kWScale),
                         std::memory_order_relaxed);
        node->is_expanded.store(true, std::memory_order_release);
        return;
    }

    // Fast heuristic priors — no NNUE batch call on every expansion.
    // Captures and promotions get higher priors via MVV-LVA; losing moves
    // (negative SEE) are suppressed. This lets MCTS search ~50x more nodes
    // per second vs the old get_policy_nnue batch path (~4ms/expansion).
    // Q-values from evaluate_node (single NNUE call per leaf) still guide
    // the tree correctly via PUCT even without precise policy priors.
    static constexpr int kPV[8] = {0, 100, 325, 335, 500, 900, 20000, 0};

    // Stack-allocated priors array — avoids heap allocation on the hot path.
    // Max legal moves in a chess position is 218 (theoretical), 256 is safe.
    float priors_buf[256];
    float sum_p = 0.0f;
    const int n_moves = moves.size;

    for (int i = 0; i < n_moves; ++i) {
        const Move m = moves[static_cast<std::size_t>(i)];
        float p;

        if (m.type() == Move::Promotion) {
            p = (m.promotion_piece() == Queen) ? 18.0f : 4.0f;
        } else {
            const Piece victim = pos.piece_on(m.to());
            if (victim != NoPiece || m.type() == Move::EnPassant) {
                const int vval = (m.type() == Move::EnPassant)
                    ? kPV[Pawn] : kPV[type_of(victim)];
                const int aval = kPV[type_of(pos.piece_on(m.from()))];
                p = 2.0f + std::max(0, vval - aval / 2) * 0.005f;
            } else {
                p = 1.0f;
            }
            if (!pos.in_check() && !see_ge(pos, m, 0)) {
                p *= 0.1f;
            }
        }

        priors_buf[i] = p;
        sum_p += p;
    }

    // NNUE-informed priors near the top of the tree. The heuristic priors above
    // are cheap but tactically blind: they can't tell that a sortie gets trapped
    // (which lost us a piece vs stash17 — 9.Nb5 ... 12.Na3 allowing ...Nxc3).
    // For the high-stakes decisions close to the root, replace them by scoring
    // every child with the value net in one batched forward pass and softmaxing
    // the results, so PUCT spends its visits on the moves the net actually likes.
    // Depth-gated via g_policy_depth because doing this at every node tanks NPS.
    // Requires a live, computed accumulator on `pos` (true on the descent path);
    // do_move maintains it incrementally and undo_move restores it.
    if (g_policy_depth >= 0 && depth <= g_policy_depth && n_moves <= 256 &&
        nnue::is_loaded() && pos.accumulator().computed) {
        static thread_local std::vector<nnue::Accumulator> p_accs;
        static thread_local std::vector<Color>             p_stms;
        static thread_local std::vector<Score>             p_scores;
        if (static_cast<int>(p_accs.size()) < n_moves) {
            p_accs.resize(static_cast<std::size_t>(n_moves));
            p_stms.resize(static_cast<std::size_t>(n_moves));
            p_scores.resize(static_cast<std::size_t>(n_moves));
        }
        StateInfo pst;
        for (int i = 0; i < n_moves; ++i) {
            pos.do_move(moves[static_cast<std::size_t>(i)], pst);  // updates+snapshots acc
            p_accs[static_cast<std::size_t>(i)] = pos.accumulator();
            p_stms[static_cast<std::size_t>(i)] = pos.side_to_move();
            pos.undo_move(moves[static_cast<std::size_t>(i)], pst);
        }
        nnue::evaluate_batch(p_accs.data(), p_stms.data(), p_scores.data(), n_moves);
        // p_scores[i] is cp from the child's side to move (our opponent); our
        // value is its negation. Stable softmax with a cp temperature.
        static constexpr float kPolicyTemp = 150.0f;
        float max_v = -1e30f;
        for (int i = 0; i < n_moves; ++i) {
            const float v = -static_cast<float>(p_scores[static_cast<std::size_t>(i)]);
            priors_buf[i] = v;
            if (v > max_v) max_v = v;
        }
        sum_p = 0.0f;
        for (int i = 0; i < n_moves; ++i) {
            priors_buf[i] = std::exp((priors_buf[i] - max_v) / kPolicyTemp);
            sum_p += priors_buf[i];
        }
    }

    node->children.reserve(static_cast<std::size_t>(n_moves));
    for (int i = 0; i < n_moves; ++i) {
        const float p = (sum_p > 1e-6f)
            ? priors_buf[i] / sum_p
            : 1.0f / static_cast<float>(n_moves);
        node->children.push_back(make_node(moves[static_cast<std::size_t>(i)], node, p));
    }

    // Release ordering ensures other threads that acquire-load is_expanded
    // see the fully-populated children vector before iterating it.
    node->is_expanded.store(true, std::memory_order_release);
}

float MCTS::evaluate_node(const Position& pos) {
    const Score s = evaluate(pos);
    return std::tanh(static_cast<float>(s) / nnue::output_cp_per_unit());
}

}  // namespace eclipse::mcts
