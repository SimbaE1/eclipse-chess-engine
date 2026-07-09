// SPDX-License-Identifier: GPL-3.0-or-later
#include "ab.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include "bitboard.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "syzygy.hpp"
#include "tt.hpp"
#include "types.hpp"

#include "see.hpp"
#include "thread_util.hpp"

namespace eclipse::ab {

// LMR Table: [depth][move_index]
float g_lmr_table[64][256];
bool  g_lmr_init = false;

void init_search_tables() {
    if (g_lmr_init) return;
    for (int d = 1; d < 64; ++d) {
        for (int i = 1; i < 256; ++i) {
            // Standard Stockfish-style LMR formula.
            g_lmr_table[d][i] = 0.75f + std::log(static_cast<float>(d)) * std::log(static_cast<float>(i)) / 2.25f;
        }
    }
    g_lmr_init = true;
}

namespace {

using Clock = std::chrono::steady_clock;

// Hard recursion ceiling for negamax/qsearch. Check extensions keep depth
// constant while in check, and qsearch follows capture/check sequences, so a
// forcing line can recurse far past the nominal search depth. Without a ceiling
// this overflows the thread stack on sharp positions (SIGBUS, exit code -10 in
// real games). 240 is ~4x the deepest seldepth ever observed in real play, so
// it never truncates a legitimate line, while staying well under the
// kMateInMaxPly (=mate-1000) window so mate scores remain valid. Beyond this
// ply we return the static eval — a position this deep is effectively a horizon.
constexpr int kMaxSearchPly = 240;

struct SearchCtx {
    Clock::time_point start;
    std::int64_t       budget_ms;
    std::int64_t       nodes;
    bool               aborted;

    // Killer moves: two per ply. Sized 64; every access is guarded by
    // `ply < 64` (plies beyond that simply get no killer bonus), so deep
    // recursion up to kMaxSearchPly never indexes out of bounds.
    std::array<std::array<Move, 2>, 64> killers{};

    // History heuristic: [color][from][to].
    std::int32_t history[2][64][64]{};

    // Countermove heuristic: the quiet move that caused a beta cutoff in
    // response to the opponent's move [color][from][to].
    Move countermove[2][64][64]{};

    // Capture history: bonus for captures that caused beta cutoffs.
    // Indexed [color][attacker_type][victim_type][to_square].
    std::int32_t capture_history[2][7][7][64]{};

    // Continuation history: bonus for a quiet move that caused a beta cutoff
    // right after a specific previous move. Indexed at piece-to granularity on
    // both sides — [color][prev_piece][prev_to][piece][to] — which separates
    // e.g. "Nf3 after ...e5" from "Bf3 after ...Qe5" (the old square-only
    // table conflated them). 6.4 MB, so it lives on the heap: SearchCtx sits
    // on thread stacks that also carry the deep negamax recursion.
    std::vector<std::int32_t> cont_history =
        std::vector<std::int32_t>(2 * 7 * 64 * 7 * 64, 0);

    static std::size_t cont_hist_idx(Color c, PieceType prev_pt, Square prev_to,
                                     PieceType pt, Square to) noexcept {
        return ((((static_cast<std::size_t>(c) * 7
                 + static_cast<std::size_t>(prev_pt)) * 64
                 + static_cast<std::size_t>(prev_to)) * 7
                 + static_cast<std::size_t>(pt)) * 64)
                 + static_cast<std::size_t>(to);
    }
    std::int32_t& cont_hist(Color c, PieceType prev_pt, Square prev_to,
                            PieceType pt, Square to) noexcept {
        return cont_history[cont_hist_idx(c, prev_pt, prev_to, pt, to)];
    }
    std::int32_t cont_hist(Color c, PieceType prev_pt, Square prev_to,
                           PieceType pt, Square to) const noexcept {
        return cont_history[cont_hist_idx(c, prev_pt, prev_to, pt, to)];
    }

    // Static eval per ply along the current search path, for the "improving"
    // heuristic (is our eval better than two plies ago?). kInfinite = no eval
    // recorded at that ply (node was in check).
    std::array<Score, kMaxSearchPly + 1> eval_stack{};

    // Zobrist key path from the search root.  Populated by negamax push/pop
    // around each do_move so we can detect in-search repetitions.
    std::vector<std::uint64_t> key_history;

    // When set, TT depth-cutoffs are only honoured for entries written in
    // `cutoff_gen` (the generation this search stamped). Used by
    // find_tactic_node so a deeper entry left by a prior search can't
    // short-circuit a shallow depth and hide the depth-by-depth swing.
    // Stale entries are still used for move ordering and still overwritten.
    bool         gen_filter = false;
    std::uint8_t cutoff_gen = 0;

    // Shared across Lazy-SMP workers: any worker that hits the time limit (or
    // the main worker finishing) sets it, and every worker bails at its next
    // stride check. nullptr in single-threaded search.
    std::atomic<bool>* stop = nullptr;
    // Caller-owned abort flag (e.g. &SearchInfo::stop). Distinct from `stop`,
    // which is the internal Lazy-SMP "main worker done, halt helpers" signal.
    // Either being set aborts the search.
    const std::atomic<bool>* ext_stop = nullptr;

    // Ponder-aware budget gate (e.g. &SearchInfo::ponderhit_at_ms), or null for
    // a normal search. When set, budget_ms is measured from the ponderhit
    // instant rather than from `start`: while still pondering (value < 0) the
    // search runs on the opponent's free time, then spends budget_ms after the
    // hit. This keeps a parallel AB worker — launched at `go ponder`, i.e. when
    // pondering begins — from burning its whole budget during free time and
    // leaving nothing for the post-ponderhit verify (the MCTS phase is already
    // ponder-aware; this makes AB match it). Holds steady_clock-epoch ms.
    const std::atomic<std::int64_t>* ponder_hit_ms = nullptr;

    SearchCtx(Clock::time_point s, std::int64_t b)
        : start(s), budget_ms(b), nodes(0), aborted(false) {}

    bool time_up() noexcept {
        // Cheap node-stride check before the clock read so we don't read the
        // clock on every leaf. The stride must be sized for the REAL nps: with
        // NNUE eval per node this search runs ~40k nps, so the old 4096-node
        // stride meant one clock read per ~100ms — every AB phase could blow
        // through its budget (and the move's hard deadline) by that much,
        // which is exactly what flagged sub-300ms movetime games. 256 nodes
        // ≈ 6ms resolution at 40k nps; a steady_clock read is ~20ns, so even
        // at 1M nps the overhead is invisible.
        if ((nodes & 0xFFu) != 0) return false;
        if (stop && stop->load(std::memory_order_relaxed)) return true;
        if (ext_stop && ext_stop->load(std::memory_order_relaxed)) return true;
        if (budget_ms <= 0) return false;
        if (ponder_hit_ms) {
            // Ponder search: budget runs from the ponderhit instant. Still
            // pondering (no hit yet) => unbounded on free time (only stop/
            // ext_stop above can abort, exactly as the MCTS phase behaves).
            const auto ph = ponder_hit_ms->load(std::memory_order_acquire);
            if (ph < 0) return false;
            const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now().time_since_epoch()).count();
            return (now_ms - ph) >= budget_ms;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - start).count();
        return elapsed >= budget_ms;
    }
};

// Score a move for ordering. Captures sorted by MVV-LVA, then killers,
// then countermove, then history.
int order_score(const Position& pos, Move m, int ply, const SearchCtx& ctx,
                Move prev_move = MoveNone) {
    // Castling is encoded king-to-rook, so piece_on(m.to()) sees our own rook;
    // without this guard it would be scored as a rook capture. It's a quiet move.
    const Piece     vict_p = m.type() == Move::Castling ? NoPiece : pos.piece_on(m.to());
    const Piece     aggr_p = pos.piece_on(m.from());
    const PieceType vict   = type_of(vict_p);
    const PieceType aggr   = type_of(aggr_p);

    int s = 0;
    if (vict != NoPieceType) {
        // MVV-LVA + capture history bonus.
        s = 2'000'000 + kPieceValue[vict] * 10 - kPieceValue[aggr]
            + ctx.capture_history[pos.side_to_move()][aggr][vict][m.to()];
    } else if (m.type() == Move::EnPassant) {
        s = 2'000'000 + kPieceValue[Pawn] * 10 - kPieceValue[Pawn]
            + ctx.capture_history[pos.side_to_move()][Pawn][Pawn][m.to()];
    } else if (m.type() == Move::Promotion) {
        s = 2'000'000 + kPieceValue[m.promotion_piece()] * 10;
    } else {
        // Quiet moves.
        if (ply < 64) {
            if (m == ctx.killers[ply][0]) return 1'000'000;
            if (m == ctx.killers[ply][1]) return 900'000;
        }
        // Countermove: move that historically refuted the previous move.
        if (prev_move != MoveNone) {
            const Color pc = pos.side_to_move() == White ? Black : White;
            if (ctx.countermove[pc][prev_move.from()][prev_move.to()] == m)
                return 800'000;
        }
        s = ctx.history[pos.side_to_move()][m.from()][m.to()];
        if (prev_move != MoveNone) {
            s += ctx.cont_hist(pos.side_to_move(),
                               type_of(pos.piece_on(prev_move.to())), prev_move.to(),
                               type_of(pos.piece_on(m.from())), m.to());
        }
    }
    return s;
}

// Apply a history bonus/malus clamped to [-kHistMax, kHistMax].
// Using a gravity formula (bonus = delta - existing * |delta| / kHistMax)
// keeps the table self-normalizing without a periodic full rescale.
constexpr std::int32_t kHistMax = 16384;
inline void update_history(std::int32_t& entry, int bonus) {
    entry += static_cast<std::int32_t>(bonus) - entry * std::abs(bonus) / kHistMax;
}

// Captures/promotions/en-passants only, generated directly (movegen's
// tactical generator) instead of generating every legal move and filtering —
// the per-quiet-move do/undo legality check was most of qsearch's movegen
// cost. For check evasions in qsearch we still use full legal generation;
// otherwise we'd risk hanging into a mate.
void generate_captures(Position& pos, MoveList& out) {
    generate_legal_captures(pos, out);
}

Score negamax(Position& pos, int depth, Score alpha, Score beta, int ply,
              Move& out_best, SearchCtx& ctx, Move excluded = MoveNone,
              Move prev_move = MoveNone);

Score qsearch(Position& pos, Score alpha, Score beta, int ply, SearchCtx& ctx) {
    if (ctx.aborted) return 0;
    ++ctx.nodes;
    if (ctx.time_up()) { ctx.aborted = true; return 0; }

    // Recursion ceiling: bail to static eval rather than overflow the stack.
    if (ply >= kMaxSearchPly) return evaluate(pos);

    // Fifty-move rule: no pawn move or capture in 100 half-moves → draw.
    if (pos.halfmove_clock() >= 100) return kDraw;

    // TT probe. qsearch entries are stored at depth 0 and every stored entry
    // has depth >= 0, so any hit is deep enough to cut here. Same generation
    // filter as negamax so find_tactic_node's depth-by-depth sweep stays
    // honest; the stored move is used for ordering either way.
    const Score alpha_orig = alpha;
    TTEntry tt_entry;
    Move    tt_move = MoveNone;
    if (g_tt.probe(pos.key(), tt_entry) &&
        (!ctx.gen_filter || tt_entry.generation == ctx.cutoff_gen)) {
        tt_move = tt_entry.move;
        const Score s = tt_entry.score_from_tt(tt_entry.score, ply);
        if (tt_entry.flag == TT_EXACT) return s;
        if (tt_entry.flag == TT_LOWERBOUND && s >= beta)  return s;
        if (tt_entry.flag == TT_UPPERBOUND && s <= alpha) return s;
    }

    const bool in_check = pos.in_check();

    // Stand-pat: assume we can do nothing and accept the static eval. The
    // search then only considers moves that BEAT the stand-pat — i.e.
    // captures that improve our position. This is what makes qsearch
    // tactically focused without exploding. Not available in check: every
    // evasion must be searched, and "doing nothing" isn't an option.
    Score stand_pat = -kInfinite;
    if (!in_check) {
        stand_pat = evaluate(pos);
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
    }

    MoveList caps;
    if (in_check) {
        // In check we MUST move; consider all legal evasions so we don't
        // miss a quiet block / king move that escapes mate.
        generate_legal_moves(pos, caps);
        // No evasions at all: checkmate, scored shallowest-first like negamax.
        if (caps.size == 0) return -static_cast<Score>(kMateScore - ply);
    } else {
        generate_captures(pos, caps);
    }

    // Sort by MVV-LVA (descending order_score); TT move first.
    std::array<int, 256> scores{};
    for (int i = 0; i < caps.size; ++i) {
        scores[static_cast<std::size_t>(i)] = (caps[i] == tt_move)
            ? 3'000'000 : order_score(pos, caps[i], ply, ctx);
    }
    for (int i = 1; i < caps.size; ++i) {
        for (int j = i; j > 0 && scores[static_cast<std::size_t>(j)] > scores[static_cast<std::size_t>(j - 1)]; --j) {
            std::swap(scores[static_cast<std::size_t>(j)], scores[static_cast<std::size_t>(j - 1)]);
            std::swap(caps[static_cast<std::size_t>(j)],   caps[static_cast<std::size_t>(j - 1)]);
        }
    }

    Score best      = stand_pat;  // -kInfinite when in check: must find a move
    Move  best_move = MoveNone;
    for (int i = 0; i < caps.size; ++i) {
        const Move m = caps[static_cast<std::size_t>(i)];
        if (!in_check) {
            // SEE pruning: skip losing captures.
            if (!see_ge(pos, m, 0)) continue;
            // Delta pruning: if even capturing the victim can't raise alpha,
            // skip (300cp margin covers typical eval inaccuracies).
            if (m.type() != Move::Promotion) {
                const PieceType vt = (m.type() == Move::EnPassant)
                    ? Pawn : type_of(pos.piece_on(m.to()));
                if (stand_pat + kPieceValue[vt] + 300 <= alpha) continue;
            }
        }

        StateInfo st;
        pos.do_move(m, st);
        const Score s = -qsearch(pos, -beta, -alpha, ply + 1, ctx);
        pos.undo_move(m, st);
        if (ctx.aborted) return 0;
        if (s > best) { best = s; best_move = m; }
        if (s > alpha) alpha = s;
        if (alpha >= beta) break;
    }

    // Store at depth 0: never evicts a real (depth >= 1) negamax entry for
    // the same position, but saves re-running qsearch on transpositions.
    const TTFlag flag = (best >= beta)       ? TT_LOWERBOUND
                      : (best > alpha_orig)  ? TT_EXACT
                                             : TT_UPPERBOUND;
    g_tt.store(pos.key(), best_move, best, 0, flag, ply);

    return best >= beta ? beta : best;
}

Score negamax(Position& pos, int depth, Score alpha, Score beta, int ply,
              Move& out_best, SearchCtx& ctx, Move excluded, Move prev_move) {
    out_best = MoveNone;
    if (ctx.aborted) return 0;
    ++ctx.nodes;

    if (ctx.time_up()) { ctx.aborted = true; return 0; }
    if (depth <= 0) return qsearch(pos, alpha, beta, ply, ctx);

    // Recursion ceiling: check extensions keep depth constant while in check, so
    // a long forcing line can recurse here past any reasonable depth. Bail to
    // static eval rather than overflow the stack.
    if (ply >= kMaxSearchPly) return evaluate(pos);

    // Fifty-move rule.
    if (pos.halfmove_clock() >= 100) return kDraw;

    // Repetition detection: walk backwards through the key path by 2 (same
    // side to move) up to halfmove_clock steps.  A single prior occurrence
    // means 2-fold; we return draw to avoid handing the opponent a 3-fold.
    {
        const int hmc     = pos.halfmove_clock();
        const int hist_sz = static_cast<int>(ctx.key_history.size());
        for (int i = hist_sz - 2; i >= 0 && (hist_sz - i) <= hmc; i -= 2) {
            if (ctx.key_history[static_cast<std::size_t>(i)] == pos.key())
                return kDraw;
        }
    }

    // Pre-root game-history repetition: the loop above only sees positions
    // reached since the search root. If this node repeats a position already
    // played earlier in the game, it's a draw the opponent can claim — score it
    // as one so a winning side won't shuffle into it. Guarded by ply>0: at the
    // root the current position legitimately IS the (already on-board) game
    // state and must be searched for a win, not declared drawn.
    if (ply > 0 && pos.repeats_game_history()) return kDraw;

    // TT probe first so we can use TT score to correct static_eval.
    const Score alpha_orig = alpha;
    TTEntry tt_entry;
    Move    tt_move = MoveNone;
    bool    tt_hit  = false;
    if (excluded == MoveNone && g_tt.probe(pos.key(), tt_entry)) {
        tt_hit  = true;
        tt_move = tt_entry.move;
        if (tt_entry.depth >= depth &&
            (!ctx.gen_filter || tt_entry.generation == ctx.cutoff_gen)) {
            const Score s = tt_entry.score_from_tt(tt_entry.score, ply);
            if (tt_entry.flag == TT_EXACT) return s;
            if (tt_entry.flag == TT_LOWERBOUND) alpha = std::max(alpha, s);
            if (tt_entry.flag == TT_UPPERBOUND) beta  = std::min(beta, s);
            if (alpha >= beta) return s;
        }
    }

    // Syzygy tablebase probe: short-circuit interior nodes once all pieces fit.
    if (syzygy::is_enabled() && excluded == MoveNone &&
        pos.castling_rights() == NoCastling &&
        static_cast<unsigned>(popcount(pos.occupied())) <= syzygy::max_pieces()) {
        const unsigned wdl = syzygy::probe_wdl(pos);
        if (wdl != syzygy::kTbFailed) {
            Score tb_score;
            if      (wdl == syzygy::kTbWin)          tb_score =  kMateScore - 1 - ply;
            else if (wdl == syzygy::kTbCursedWin)    tb_score =  2;
            else if (wdl == syzygy::kTbBlessedLoss)  tb_score = -2;
            else if (wdl == syzygy::kTbLoss)         tb_score = -(kMateScore - 1 - ply);
            else                                       tb_score =  kDraw;
            const TTFlag tf = (tb_score >= beta) ? TT_LOWERBOUND
                            : (tb_score <= alpha) ? TT_UPPERBOUND : TT_EXACT;
            g_tt.store(pos.key(), MoveNone, tb_score, depth, tf, ply);
            return std::clamp(tb_score, alpha, beta);
        }
    }

    // Cache in_check once: used by every pruning guard and move-loop check below.
    const bool in_check = pos.in_check();

    // PV nodes are searched with an open window; every zero-window probe has
    // beta == alpha+1. The speculative prunes below (RFP, null move, probcut,
    // futility, LMP, SEE quiet pruning) are only sound as fail-high/fail-low
    // guesses, so they run on non-PV nodes only; the PV gets the honest search.
    const bool is_pv = (beta - alpha) > 1;

    // Lazy static eval: skipped for in-check positions where no pruning fires.
    Score static_eval = 0;
    if (!in_check) {
        static_eval = evaluate(pos);
        // TT score correction: if the TT gives us a tighter bound, use it.
        if (tt_hit) {
            const Score tt_s = tt_entry.score_from_tt(tt_entry.score, ply);
            if ((tt_entry.flag == TT_LOWERBOUND && tt_s > static_eval) ||
                (tt_entry.flag == TT_UPPERBOUND && tt_s < static_eval) ||
                 tt_entry.flag == TT_EXACT) {
                static_eval = tt_s;
            }
        }
    }

    // Improving: is this node's eval better than the same side's eval two
    // plies up the path? A falling eval means fail-lows are more likely, so
    // pruning thresholds tighten (more pruning); a rising one loosens them.
    // In-check plies record no eval (kInfinite sentinel) and count as not
    // improving; with no usable history, default to improving (prune less).
    ctx.eval_stack[static_cast<std::size_t>(ply)] = in_check ? kInfinite : static_eval;
    bool improving = false;
    if (!in_check) {
        if (ply >= 2 && ctx.eval_stack[static_cast<std::size_t>(ply - 2)] != kInfinite)
            improving = static_eval > ctx.eval_stack[static_cast<std::size_t>(ply - 2)];
        else if (ply >= 4 && ctx.eval_stack[static_cast<std::size_t>(ply - 4)] != kInfinite)
            improving = static_eval > ctx.eval_stack[static_cast<std::size_t>(ply - 4)];
        else
            improving = true;
    }

    // Reverse Futility Pruning (RFP): also known as Static Null Move Pruning.
    // An improving eval discounts one depth from the margin — the fail-high
    // guess is more trustworthy when the trend is with us.
    if (!is_pv && !in_check && depth <= 6 &&
        static_eval >= beta + 120 * (depth - (improving ? 1 : 0))) {
        return static_eval;
    }

    // Null-move pruning.
    if (!is_pv && !in_check && depth >= 3 && static_eval >= beta) {
        const Color us         = pos.side_to_move();
        const Bitboard minors  = pos.pieces(us, Knight) | pos.pieces(us, Bishop);
        const Bitboard majors  = pos.pieces(us, Rook)   | pos.pieces(us, Queen);
        const Bitboard non_p   = minors | majors;
        const bool safe_for_nm = majors || (non_p && (non_p & (non_p - 1)));
        if (safe_for_nm) {
            const int R = 3 + depth / 4;
            StateInfo st;
            pos.do_null_move(st);
            Move dummy;
            const Score s = -negamax(pos, depth - 1 - R, -beta, -beta + 1, ply + 1, dummy, ctx, MoveNone, MoveNone);
            pos.undo_null_move(st);
            if (s >= beta) return beta;
        }
    }

    // ProbCut: at depth >= 5, check if any capture can be proven to exceed
    // a generous beta threshold via a shallow search. If so, cut immediately.
    // The SEE filter skips captures unlikely to reach probcut_beta, keeping
    // the overhead small relative to the pruning benefit.
    if (!is_pv && !in_check && depth >= 5 && excluded == MoveNone
        && std::abs(beta) < kMateInMaxPly) {
        const Score probcut_beta  = beta + 180;
        const int   probcut_depth = depth - 4;
        MoveList    pc_caps;
        generate_captures(pos, pc_caps);
        // Order by MVV-LVA for early cutoff.
        std::array<int, 256> pc_scores{};
        for (int i = 0; i < pc_caps.size; ++i)
            pc_scores[static_cast<std::size_t>(i)] = order_score(pos, pc_caps[i], ply, ctx);
        for (int i = 1; i < pc_caps.size; ++i)
            for (int j = i; j > 0 && pc_scores[static_cast<std::size_t>(j)] >
                                      pc_scores[static_cast<std::size_t>(j - 1)]; --j) {
                std::swap(pc_scores[static_cast<std::size_t>(j)],   pc_scores[static_cast<std::size_t>(j - 1)]);
                std::swap(pc_caps  [static_cast<std::size_t>(j)],   pc_caps  [static_cast<std::size_t>(j - 1)]);
            }
        for (int i = 0; i < pc_caps.size && !ctx.aborted; ++i) {
            const Move m = pc_caps[static_cast<std::size_t>(i)];
            if (!see_ge(pos, m, probcut_beta - static_eval)) continue;
            StateInfo st;
            const std::uint64_t parent_key = pos.key();
            pos.do_move(m, st);
            ctx.key_history.push_back(parent_key);
            Move dummy;
            Score s = -qsearch(pos, -probcut_beta, -probcut_beta + 1, ply + 1, ctx);
            if (s >= probcut_beta && probcut_depth > 0)
                s = -negamax(pos, probcut_depth, -probcut_beta, -probcut_beta + 1, ply + 1, dummy, ctx, MoveNone, m);
            ctx.key_history.pop_back();
            pos.undo_move(m, st);
            if (ctx.aborted) return 0;
            if (s >= probcut_beta) {
                g_tt.store(pos.key(), m, s, depth - 3, TT_LOWERBOUND, ply);
                return probcut_beta;
            }
        }
    }

    // Singular extension: if we have a TT move that is a lower bound (cut-move)
    // with enough depth, check if it's singularly better than all others.
    int extension = 0;
    if (excluded == MoveNone && depth >= 8 && tt_move != MoveNone && tt_entry.depth >= depth - 3 && tt_entry.flag == TT_LOWERBOUND) {
        const Score tt_score = tt_entry.score_from_tt(tt_entry.score, ply);
        const Score margin   = depth; // margin = 1cp per depth

        Move dummy;
        const Score s = negamax(pos, depth / 2, tt_score - margin - 1, tt_score - margin, ply + 1, dummy, ctx, tt_move, prev_move);
        if (ctx.aborted) return 0;

        if (s < tt_score - margin) {
            extension = 1;
        }
    }

    // Check extension: a node where we're in check is a forced reply, not a
    // free choice, so give it an extra ply to avoid horizon effects in
    // forcing sequences. Capped at the same 1-ply budget as singular
    // extension so the two can't stack.
    if (in_check) extension = std::max(extension, 1);

    // Internal Iterative Deepening: nodes without a TT move (e.g. first visit,
    // or the stored entry's move was pruned away) have nothing to sort first.
    // A reduced-depth search fills in tt_move for ordering below. depth - 2
    // keeps this geometrically bounded -- it can recurse, but depth drops by
    // 2 each time until < 4, where IID no longer triggers.
    if (tt_move == MoveNone && excluded == MoveNone && depth >= 4) {
        Move iid_move;
        negamax(pos, depth - 2, alpha, beta, ply, iid_move, ctx, MoveNone, prev_move);
        if (ctx.aborted) return 0;
        if (iid_move != MoveNone) tt_move = iid_move;
    }

    MoveList moves;
    generate_legal_moves(pos, moves);

    if (moves.size == 0) {
        // Checkmate or stalemate. Mate score is depth-adjusted so the
        // shallowest mate is preferred. Stalemate = draw.
        return in_check ? -static_cast<Score>(kMateScore - ply) : Score{0};
    }

    // Futility pruning: if static eval is far below alpha at low depth, quiet
    // moves cannot realistically raise it so we skip them (captures + checks
    // are still searched).
    static constexpr Score kFutilityMargin[4] = {0, 150, 300, 500};
    const bool can_futility_prune =
        !is_pv
        && !in_check
        && depth <= 3
        && excluded == MoveNone
        && alpha > -kMateInMaxPly && alpha < kMateInMaxPly
        && static_eval + kFutilityMargin[depth] <= alpha;

    // LMP: at depth 1-2, after trying N quiet moves, skip the rest. A falling
    // eval (not improving) halves the budget — late quiets are even less
    // likely to rescue a deteriorating position.
    static constexpr int kLmpCount[3] = {0, 8, 15};
    const bool can_lmp =
        !is_pv
        && !in_check
        && depth <= 2
        && excluded == MoveNone
        && alpha > -kMateInMaxPly && alpha < kMateInMaxPly;
    const int lmp_limit = (depth <= 2) ? (improving ? kLmpCount[depth]
                                                    : kLmpCount[depth] / 2)
                                       : 0;
    int quiet_tried = 0;

    // SEE pruning for quiet moves at low depth.
    const bool can_see_prune_quiet =
        !is_pv
        && !in_check
        && depth <= 4
        && excluded == MoveNone
        && alpha > -kMateInMaxPly;

    // Order moves. TT move comes first.
    std::array<int, 256> scores{};
    for (int i = 0; i < moves.size; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        if (moves[idx] == tt_move) scores[idx] = 3'000'000;
        else                       scores[idx] = order_score(pos, moves[idx], ply, ctx, prev_move);
    }
    for (int i = 1; i < moves.size; ++i) {
        for (int j = i; j > 0 && scores[static_cast<std::size_t>(j)] > scores[static_cast<std::size_t>(j - 1)]; --j) {
            std::swap(scores[static_cast<std::size_t>(j)], scores[static_cast<std::size_t>(j - 1)]);
            std::swap(moves[static_cast<std::size_t>(j)],  moves[static_cast<std::size_t>(j - 1)]);
        }
    }

    // Track quiet moves searched so we can apply history malus on beta cutoffs.
    std::array<Move, 64> quiets_tried{};
    int                  quiets_tried_count = 0;

    Score best = -kInfinite;
    Move  best_move = MoveNone;
    for (int i = 0; i < moves.size; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const Move m = moves[idx];
        if (m == excluded) continue;

        // Castling reads as "occupied destination" (king-to-rook encoding)
        // but is a quiet move, not a rook capture.
        const bool is_quiet = m.type() == Move::Castling
                           || (pos.piece_on(m.to()) == NoPiece
                               && m.type() != Move::EnPassant
                               && m.type() != Move::Promotion);

        // Futility pruning: skip quiet moves that can't raise alpha.
        if (can_futility_prune && i > 0 && is_quiet) continue;

        // LMP: once we've tried enough quiet moves at low depth, stop.
        if (can_lmp && is_quiet) {
            if (quiet_tried >= lmp_limit) continue;
        }
        if (is_quiet) {
            // SEE quiet pruning: skip quiet moves with very bad static exchange.
            // Castling excluded: SEE reads its king-to-rook encoding as a
            // capture of our own rook and returns nonsense.
            if (can_see_prune_quiet && i > 0 && m.type() != Move::Castling &&
                !see_ge(pos, m, -50 * depth)) continue;

            ++quiet_tried;
            if (quiets_tried_count < 64)
                quiets_tried[static_cast<std::size_t>(quiets_tried_count++)] = m;
        }

        const Color mover = pos.side_to_move();
        // Piece types for the piece-to continuation history, read BEFORE
        // do_move mutates the board (the LMR block below runs after it).
        const PieceType moved_pt = type_of(pos.piece_on(m.from()));
        const PieceType prev_pt  = prev_move != MoveNone
            ? type_of(pos.piece_on(prev_move.to())) : NoPieceType;
        StateInfo st;
        const std::uint64_t parent_key = pos.key();
        pos.do_move(m, st);
        ctx.key_history.push_back(parent_key);

        Move  dummy;
        Score s;
        if (i == 0) {
            // PV node: search with full window.
            s = -negamax(pos, depth - 1 + extension, -beta, -alpha, ply + 1, dummy, ctx, MoveNone, m);
        } else {
            // LMR: reduce depth for late-ordered quiet moves.
            int reduction = 0;
            if (depth >= 3 && i >= 3 && !in_check && is_quiet) {
                const int d_idx = std::min(depth, 63);
                const int i_idx = std::min(i, 255);
                reduction = static_cast<int>(g_lmr_table[d_idx][i_idx]);

                // Reduce less for killers and high-history moves; more for
                // low-history moves. Keeps reduction bounded to [0, depth-1].
                if (ply < 64 &&
                    (m == ctx.killers[static_cast<std::size_t>(ply)][0] ||
                     m == ctx.killers[static_cast<std::size_t>(ply)][1])) {
                    reduction -= 1;
                }
                const int hist_val = ctx.history[mover][m.from()][m.to()];
                reduction -= hist_val / (kHistMax / 2);  // ±1 at ±½·kHistMax

                if (prev_move != MoveNone) {
                    const int cont_val = ctx.cont_hist(mover, prev_pt, prev_move.to(),
                                                       moved_pt, m.to());
                    reduction -= cont_val / (kHistMax / 2);
                }

                // The PV deserves a fuller look: reduce one ply less there.
                if (is_pv) reduction -= 1;

                reduction = std::clamp(reduction, 0, depth - 1);
            }

            // Non-PV node: search with null window first.
            s = -negamax(pos, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, dummy, ctx, MoveNone, m);
            if (s > alpha && reduction > 0) {
                // Failed high on reduced search: re-search with full depth but null window.
                s = -negamax(pos, depth - 1, -alpha - 1, -alpha, ply + 1, dummy, ctx, MoveNone, m);
            }
            if (s > alpha && s < beta) {
                // Failed high on null window: re-search with full window.
                s = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, dummy, ctx, MoveNone, m);
            }
        }

        ctx.key_history.pop_back();
        pos.undo_move(m, st);
        if (ctx.aborted) return 0;
        if (s > best) { best = s; best_move = m; }
        if (s > alpha) alpha = s;
        if (alpha >= beta) {
            const Color us = pos.side_to_move();
            const int bonus = depth * depth;
            if (is_quiet) {
                if (ply < 64) {
                    const std::size_t pidx = static_cast<std::size_t>(ply);
                    if (m != ctx.killers[pidx][0]) {
                        ctx.killers[pidx][1] = ctx.killers[pidx][0];
                        ctx.killers[pidx][0] = m;
                    }
                }
                update_history(ctx.history[us][m.from()][m.to()], bonus);
                if (prev_move != MoveNone) {
                    update_history(ctx.cont_hist(us, prev_pt, prev_move.to(),
                                                 moved_pt, m.to()), bonus);
                }

                // History malus: penalize all quiets searched before the cutoff move.
                const int malus = -(depth * depth / 2);
                for (int j = 0; j < quiets_tried_count - 1; ++j) {
                    const Move qm = quiets_tried[static_cast<std::size_t>(j)];
                    update_history(ctx.history[us][qm.from()][qm.to()], malus);
                    if (prev_move != MoveNone) {
                        // pos is restored after undo_move, so qm's mover is
                        // back on its from-square.
                        update_history(ctx.cont_hist(us, prev_pt, prev_move.to(),
                                                     type_of(pos.piece_on(qm.from())),
                                                     qm.to()), malus);
                    }
                }

                // Countermove: record that m refutes prev_move.
                if (prev_move != MoveNone) {
                    const Color opp = us == White ? Black : White;
                    ctx.countermove[opp][prev_move.from()][prev_move.to()] = m;
                }
            } else {
                // Capture/promotion beta cutoff: update capture history.
                const PieceType agg_t = type_of(pos.piece_on(m.from()));
                const PieceType vic_t = (m.type() == Move::EnPassant)
                    ? Pawn : type_of(pos.piece_on(m.to()));
                update_history(ctx.capture_history[us][agg_t][vic_t][m.to()], bonus);
            }
            break;  // beta cutoff
        }
    }

    out_best = best_move;
    TTFlag flag = TT_EXACT;
    if      (best <= alpha_orig) flag = TT_UPPERBOUND;
    else if (best >= beta)       flag = TT_LOWERBOUND;
    g_tt.store(pos.key(), best_move, best, depth, flag, ply);

    return best;
}

// Walk the TT from `pos` (a disposable copy -- this never undoes) following
// each position's stored best move, up to `max_len` plies. Every node
// visited during search already calls g_tt.store() with its best move, so
// this reconstructs the PV without needing a dedicated triangular array.
// Stops early on a TT miss, a stored move that's no longer legal (stale
// entry got overwritten by a different position with the same key slot),
// or a missing move (upper-bound-only entries store one).
std::vector<Move> extract_pv_from_tt(Position pos, int max_len) {
    std::vector<Move> pv;
    pv.reserve(static_cast<std::size_t>(max_len));
    for (int i = 0; i < max_len; ++i) {
        TTEntry entry;
        if (!g_tt.probe(pos.key(), entry) || entry.move == MoveNone) break;

        MoveList legal;
        generate_legal_moves(pos, legal);
        bool is_legal = false;
        for (int j = 0; j < legal.size; ++j) {
            if (legal[static_cast<std::size_t>(j)] == entry.move) { is_legal = true; break; }
        }
        if (!is_legal) break;

        StateInfo st;
        pos.do_move(entry.move, st);
        pv.push_back(entry.move);
    }
    return pv;
}

}  // namespace

// One iterative-deepening worker loop. `start_depth` lets Lazy-SMP helper
// threads begin a ply ahead of the main worker so they fill the shared TT with
// different subtrees instead of duplicating the main line. Best move from each
// completed depth is preserved even if the next depth aborts, so the caller
// always sees a sane move.
static Result id_search(Position& pos, int max_depth, SearchCtx& ctx, int start_depth) {
    Result r;
    Score last_score = 0;
    for (int d = start_depth; d <= max_depth; ++d) {
        Move best_at_d;
        Score s;

        // Aspiration window or full window?
        if (d <= 4) {
            s = negamax(pos, d, -kInfinite, kInfinite, 0, best_at_d, ctx, MoveNone);
        } else {
            // Start with ±50 cp (half a pawn). Iteration-to-iteration eval
            // swings of 30 cp are normal — the previous ±30 window caused
            // frequent re-searches that capped reached depth. On fail-low/
            // high, widen exponentially (50→200→800→full) so we recover from
            // a real eval shift in 2-3 re-searches rather than 8+ linear
            // ones.
            Score delta = 50;
            Score alpha = std::max(-kInfinite, last_score - delta);
            Score beta  = std::min( kInfinite, last_score + delta);
            while (true) {
                s = negamax(pos, d, alpha, beta, 0, best_at_d, ctx, MoveNone);
                if (ctx.aborted) break;
                if      (s <= alpha) { delta *= 4; alpha = std::max(-kInfinite, last_score - delta); }
                else if (s >= beta)  { delta *= 4; beta  = std::min( kInfinite, last_score + delta); }
                else break;
                if (delta >= 800) { alpha = -kInfinite; beta = kInfinite; }
            }
        }

        if (ctx.aborted) break;
        r.move      = best_at_d;
        r.score     = s;
        r.reached_d = d;
        r.depth_scores.push_back(s);
        last_score  = s;
    }
    r.nodes = ctx.nodes;
    return r;
}

Result find_best_move(Position& pos, int max_depth, std::int64_t time_budget_ms,
                      int num_threads, const std::atomic<bool>* ext_stop,
                      const std::atomic<std::int64_t>* ponder_hit_ms) {
    init_search_tables();
    num_threads = std::max(1, num_threads);
    const auto start = Clock::now();

    if (num_threads == 1) {
        SearchCtx ctx{start, time_budget_ms};
        ctx.ext_stop = ext_stop;
        ctx.ponder_hit_ms = ponder_hit_ms;
        Result r = id_search(pos, max_depth, ctx, 1);
        r.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - start).count();
        return r;
    }

    // Lazy SMP: workers share the global TT (lockless -- probe's key check
    // catches most torn reads; an occasional stale score is the standard,
    // accepted Lazy-SMP tradeoff). Each worker gets its own Position (the NNUE
    // accumulator is per-position) and SearchCtx (killers/history are
    // per-thread). Only the main worker's result is returned; the helpers
    // exist to warm the shared TT so the main worker reaches depth faster.
    std::atomic<bool>        stop{false};
    std::vector<Result>      results(static_cast<std::size_t>(num_threads));
    // Helpers (and the main search thread, see uci.cpp) run on a large stack:
    // the recursive negamax/qsearch overflows the 512 KB default secondary-thread
    // stack on macOS for deep forcing lines. See thread_util.hpp.
    std::vector<BigThread>   helpers;
    helpers.reserve(static_cast<std::size_t>(num_threads - 1));

    auto worker = [&](int id) {
        Position  p = pos;                 // own accumulator/board state
        SearchCtx ctx{start, time_budget_ms};
        ctx.stop = &stop;
        ctx.ext_stop = ext_stop;
        ctx.ponder_hit_ms = ponder_hit_ms;
        // Main (id 0) runs every depth so its depth_scores trajectory is the
        // clean per-depth sequence the caller's instability check relies on;
        // helpers start a ply ahead to desync.
        const int start_depth = (id == 0) ? 1 : (1 + (id % 2));
        results[static_cast<std::size_t>(id)] = id_search(p, max_depth, ctx, start_depth);
        if (id == 0) stop.store(true, std::memory_order_relaxed);  // main done -> halt helpers
    };

    for (int i = 1; i < num_threads; ++i) helpers.emplace_back(worker, i);
    worker(0);
    for (auto& t : helpers) t.join();

    // Keep the main worker's move/score/trajectory; sum nodes for honest nps.
    Result r = results[0];
    for (int i = 1; i < num_threads; ++i)
        r.nodes += results[static_cast<std::size_t>(i)].nodes;
    r.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - start).count();
    return r;
}

Score score_move(Position& pos, Move m, int max_depth, std::int64_t time_budget_ms) {
    init_search_tables();
    SearchCtx ctx{Clock::now(), time_budget_ms};

    StateInfo st;
    const std::uint64_t parent_key = pos.key();
    pos.do_move(m, st);
    ctx.key_history.push_back(parent_key);

    Score result = 0;
    for (int d = 1; d <= max_depth; ++d) {
        Move dummy;
        const Score s = -negamax(pos, d - 1, -kInfinite, kInfinite, 1, dummy, ctx, MoveNone, m);
        if (ctx.aborted) break;
        result = s;
    }

    ctx.key_history.pop_back();
    pos.undo_move(m, st);
    return result;
}

TacticNode find_tactic_node(Position& pos, int max_depth, std::int64_t time_budget_ms) {
    init_search_tables();
    TacticNode result;

    // This needs a genuine depth-by-depth progression to detect a swing --
    // negamax's `if (tt_entry.depth >= depth) return cached` shortcut means
    // a TT already warmed by a deeper prior search (the main AB search and
    // the cross-check both just ran on related positions) would hand every
    // shallow depth here the SAME pre-existing deep answer instead of a real
    // depth-limited result, making the whole swing-detection loop see no
    // swing at all -- not because there isn't one, but because every depth
    // looks identical.
    //
    // Rather than clear the whole 256MB table (slow, and it throws away the
    // warm cache the rest of the search wants), we bump the TT generation and
    // only honour depth-cutoffs from entries we ourselves wrote this call.
    // Prior searches' deeper entries are ignored for cutoffs (so each depth
    // is genuinely depth-limited) but still used for move ordering and still
    // overwritten normally, so the surrounding searches keep their warm cache.
    SearchCtx ctx{Clock::now(), time_budget_ms};
    ctx.cutoff_gen = g_tt.new_generation();
    ctx.gen_filter = true;

    std::vector<Score>             scores;
    std::vector<std::vector<Move>> pvs;
    scores.reserve(static_cast<std::size_t>(max_depth));
    pvs.reserve(static_cast<std::size_t>(max_depth));

    Score last_score = 0;
    for (int d = 1; d <= max_depth; ++d) {
        Move best_at_d;
        Score s;
        if (d <= 4) {
            s = negamax(pos, d, -kInfinite, kInfinite, 0, best_at_d, ctx, MoveNone);
        } else {
            Score delta = 50;
            Score alpha = std::max(-kInfinite, last_score - delta);
            Score beta  = std::min( kInfinite, last_score + delta);
            while (true) {
                s = negamax(pos, d, alpha, beta, 0, best_at_d, ctx, MoveNone);
                if (ctx.aborted) break;
                if      (s <= alpha) { delta *= 4; alpha = std::max(-kInfinite, last_score - delta); }
                else if (s >= beta)  { delta *= 4; beta  = std::min( kInfinite, last_score + delta); }
                else break;
                if (delta >= 800) { alpha = -kInfinite; beta = kInfinite; }
            }
        }
        if (ctx.aborted) break;
        last_score = s;
        scores.push_back(s);
        pvs.push_back(extract_pv_from_tt(pos, std::min(d, 12)));
    }

    // Scan for the shallowest stable swing: a real, newly-discovered line,
    // not aspiration-window noise from one unlucky re-search. Skips the
    // first few (noisy) depths and requires the new score to hold for the
    // next couple of completed depths before trusting it.
    constexpr int   kMinDepthIdx        = 3;   // 0-indexed; skips depths 1-3
    constexpr Score kSwingThreshold     = 60;  // cp -- below this, not a real "aha"
    constexpr Score kStabilityTolerance = 25;  // cp drift allowed across confirming depths

    const int n = static_cast<int>(scores.size());
    for (int k = kMinDepthIdx; k < n; ++k) {
        const Score k_score = scores[static_cast<std::size_t>(k)];
        const Score swing = static_cast<Score>(k_score - scores[static_cast<std::size_t>(k - 1)]);
        if (std::abs(swing) < kSwingThreshold) continue;
        if (k + 1 >= n) continue;  // no confirming depth completed -- can't trust it yet

        // A swing into mate/tablebase territory (|score| >= kMateInMaxPly) is
        // the common endgame case: the search/TB reveals the position as
        // decisively won or lost a few plies deep. There the exact magnitude
        // (mate distance, or kMateScore-1-ply from a TB probe) drifts by far
        // more than kStabilityTolerance between depths, so the cp-drift test
        // would wrongly reject it. For a decisive score we instead require the
        // confirming depths to stay decisive for the SAME side -- the verdict,
        // not the precise distance, is what has to hold.
        const bool decisive = std::abs(static_cast<int>(k_score)) >= kMateInMaxPly;

        bool stable = true;
        for (int j = k + 1; j <= k + 2 && j < n; ++j) {
            const Score sj = scores[static_cast<std::size_t>(j)];
            if (decisive) {
                if (std::abs(static_cast<int>(sj)) < kMateInMaxPly ||
                    (sj > 0) != (k_score > 0)) {
                    stable = false;
                    break;
                }
            } else if (std::abs(static_cast<int>(sj) - static_cast<int>(k_score)) > kStabilityTolerance) {
                stable = false;
                break;
            }
        }
        if (!stable) continue;

        const auto& prev_pv = pvs[static_cast<std::size_t>(k - 1)];
        const auto& cur_pv  = pvs[static_cast<std::size_t>(k)];
        std::size_t div = 0;
        while (div < prev_pv.size() && div < cur_pv.size() && prev_pv[div] == cur_pv[div]) ++div;
        if (div >= cur_pv.size()) continue;  // depths agree on every move we have -- no new line found

        result.found        = true;
        result.aha_depth    = k + 1;
        result.root_score_cp = k_score;
        result.path.assign(cur_pv.begin(), cur_pv.begin() + static_cast<long>(div) + 1);

        // scores[k] is from `pos`'s STM perspective; negate once per ply to
        // get the tactic node's value from ITS OWN side-to-move perspective
        // (same alternation as Result::score / adjust_root_q in mcts.cpp).
        const bool  odd_path_len = (result.path.size() % 2) == 1;
        const Score node_score_cp = odd_path_len ? static_cast<Score>(-k_score) : k_score;
        // MCTS Q values live in tanh space (see mcts.cpp leaf backup), so the
        // seed has to too -- a raw cp/unit would blow past (-1, 1) entirely
        // for a decisive score and grossly overweight the seeded node. tanh
        // saturates a mate/TB verdict toward +/-1 as intended.
        result.seed_q = std::tanh(static_cast<float>(node_score_cp) / nnue::output_cp_per_unit());
        break;
    }

    return result;
}

}  // namespace eclipse::ab
