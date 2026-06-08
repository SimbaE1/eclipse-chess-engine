// SPDX-License-Identifier: GPL-3.0-or-later
#include "ab.hpp"

#include <algorithm>
#include <array>
#include <chrono>

#include "movegen.hpp"
#include "tt.hpp"
#include "types.hpp"

#include "see.hpp"

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

struct SearchCtx {
    Clock::time_point start;
    std::int64_t       budget_ms;
    std::int64_t       nodes;
    bool               aborted;

    // Killer moves: two per ply.
    std::array<std::array<Move, 2>, 64> killers{};

    // History heuristic: [color][from][to].
    std::int32_t history[2][64][64]{};

    // Countermove heuristic: the quiet move that caused a beta cutoff in
    // response to the opponent's move [color][from][to].
    Move countermove[2][64][64]{};

    // Capture history: bonus for captures that caused beta cutoffs.
    // Indexed [color][attacker_type][victim_type][to_square].
    std::int32_t capture_history[2][7][7][64]{};

    // Zobrist key path from the search root.  Populated by negamax push/pop
    // around each do_move so we can detect in-search repetitions.
    std::vector<std::uint64_t> key_history;

    SearchCtx(Clock::time_point s, std::int64_t b)
        : start(s), budget_ms(b), nodes(0), aborted(false) {}

    bool time_up() noexcept {
        // Cheap node-stride check before the clock read so we don't read the
        // clock on every leaf. 4095 is one read per ~4096 leaves; at 1M nps
        // that's still ~250 reads/sec — plenty of resolution for a budget
        // measured in milliseconds.
        if ((nodes & 0xFFFu) != 0) return false;
        if (budget_ms <= 0) return false;
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - start).count();
        return elapsed >= budget_ms;
    }
};

// Score a move for ordering. Captures sorted by MVV-LVA, then killers,
// then countermove, then history.
int order_score(const Position& pos, Move m, int ply, const SearchCtx& ctx,
                Move prev_move = MoveNone) {
    const Piece     vict_p = pos.piece_on(m.to());
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

// Captures + en-passants only. For check evasions in qsearch we fall back
// to full legal generation; otherwise we'd risk hanging into a mate.
void generate_captures(Position& pos, MoveList& out) {
    MoveList all;
    generate_legal_moves(pos, all);
    for (const Move m : all) {
        const bool is_cap = pos.piece_on(m.to()) != NoPiece
                         || m.type() == Move::EnPassant
                         || m.type() == Move::Promotion;
        if (is_cap) out.push(m);
    }
}

Score negamax(Position& pos, int depth, Score alpha, Score beta, int ply,
              Move& out_best, SearchCtx& ctx, Move excluded = MoveNone,
              Move prev_move = MoveNone);

Score qsearch(Position& pos, Score alpha, Score beta, int ply, SearchCtx& ctx) {
    ++ctx.nodes;
    if (ctx.time_up()) { ctx.aborted = true; return 0; }

    // Fifty-move rule: no pawn move or capture in 100 half-moves → draw.
    if (pos.halfmove_clock() >= 100) return kDraw;

    // Stand-pat: assume we can do nothing and accept the static eval. The
    // search then only considers moves that BEAT the stand-pat — i.e.
    // captures that improve our position. This is what makes qsearch
    // tactically focused without exploding.
    const Score stand_pat = evaluate(pos);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    MoveList caps;
    if (pos.in_check()) {
        // In check we MUST move; consider all legal evasions so we don't
        // miss a quiet block / king move that escapes mate.
        generate_legal_moves(pos, caps);
    } else {
        generate_captures(pos, caps);
    }

    // Sort by MVV-LVA (descending order_score).
    std::array<int, 256> scores{};
    for (int i = 0; i < caps.size; ++i) {
        scores[static_cast<std::size_t>(i)] = order_score(pos, caps[i], ply, ctx);
    }
    for (int i = 1; i < caps.size; ++i) {
        for (int j = i; j > 0 && scores[static_cast<std::size_t>(j)] > scores[static_cast<std::size_t>(j - 1)]; --j) {
            std::swap(scores[static_cast<std::size_t>(j)], scores[static_cast<std::size_t>(j - 1)]);
            std::swap(caps[static_cast<std::size_t>(j)],   caps[static_cast<std::size_t>(j - 1)]);
        }
    }

    for (int i = 0; i < caps.size; ++i) {
        const Move m = caps[static_cast<std::size_t>(i)];
        if (!pos.in_check()) {
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
        if (s >= beta) return beta;
        if (s > alpha) alpha = s;
    }
    return alpha;
}

Score negamax(Position& pos, int depth, Score alpha, Score beta, int ply,
              Move& out_best, SearchCtx& ctx, Move excluded, Move prev_move) {
    ++ctx.nodes;
    out_best = MoveNone;

    if (ctx.time_up()) { ctx.aborted = true; return 0; }
    if (depth <= 0) return qsearch(pos, alpha, beta, ply, ctx);

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

    // TT probe first so we can use TT score to correct static_eval.
    const Score alpha_orig = alpha;
    TTEntry tt_entry;
    Move    tt_move = MoveNone;
    bool    tt_hit  = false;
    if (excluded == MoveNone && g_tt.probe(pos.key(), tt_entry)) {
        tt_hit  = true;
        tt_move = tt_entry.move;
        if (tt_entry.depth >= depth) {
            const Score s = tt_entry.score_from_tt(tt_entry.score, ply);
            if (tt_entry.flag == TT_EXACT) return s;
            if (tt_entry.flag == TT_LOWERBOUND) alpha = std::max(alpha, s);
            if (tt_entry.flag == TT_UPPERBOUND) beta  = std::min(beta, s);
            if (alpha >= beta) return s;
        }
    }

    // Cache in_check once: used by every pruning guard and move-loop check below.
    const bool in_check = pos.in_check();

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

    // Reverse Futility Pruning (RFP): also known as Static Null Move Pruning.
    if (!in_check && depth <= 6 && static_eval >= beta + 120 * depth) {
        return static_eval;
    }

    // Null-move pruning.
    if (!in_check && depth >= 3 && static_eval >= beta) {
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
    if (!in_check && depth >= 5 && excluded == MoveNone
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

        if (s < tt_score - margin) {
            extension = 1;
        }
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
        !in_check
        && depth <= 3
        && excluded == MoveNone
        && alpha > -kMateInMaxPly && alpha < kMateInMaxPly
        && static_eval + kFutilityMargin[depth] <= alpha;

    // LMP: at depth 1-2, after trying N quiet moves, skip the rest.
    static constexpr int kLmpCount[3] = {0, 8, 15};
    const bool can_lmp =
        !in_check
        && depth <= 2
        && excluded == MoveNone
        && alpha > -kMateInMaxPly && alpha < kMateInMaxPly;
    int quiet_tried = 0;

    // SEE pruning for quiet moves at low depth.
    const bool can_see_prune_quiet =
        !in_check
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

        const bool is_quiet = pos.piece_on(m.to()) == NoPiece
                           && m.type() != Move::EnPassant
                           && m.type() != Move::Promotion;

        // Futility pruning: skip quiet moves that can't raise alpha.
        if (can_futility_prune && i > 0 && is_quiet) continue;

        // LMP: once we've tried enough quiet moves at low depth, stop.
        if (can_lmp && is_quiet) {
            if (quiet_tried >= kLmpCount[depth]) continue;
        }
        if (is_quiet) {
            // SEE quiet pruning: skip quiet moves with very bad static exchange.
            if (can_see_prune_quiet && i > 0 && !see_ge(pos, m, -50 * depth)) continue;

            ++quiet_tried;
            if (quiets_tried_count < 64)
                quiets_tried[static_cast<std::size_t>(quiets_tried_count++)] = m;
        }

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
                if (m == ctx.killers[static_cast<std::size_t>(ply)][0] ||
                    m == ctx.killers[static_cast<std::size_t>(ply)][1]) {
                    reduction -= 1;
                }
                const Color us_lmr = pos.side_to_move();
                const int hist_val = ctx.history[us_lmr][m.from()][m.to()];
                reduction -= hist_val / (kHistMax / 2);  // ±1 at ±½·kHistMax

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

                // History malus: penalize all quiets searched before the cutoff move.
                const int malus = -(depth * depth / 2);
                for (int j = 0; j < quiets_tried_count - 1; ++j) {
                    const Move qm = quiets_tried[static_cast<std::size_t>(j)];
                    update_history(ctx.history[us][qm.from()][qm.to()], malus);
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

}  // namespace

Result find_best_move(Position& pos, int max_depth, std::int64_t time_budget_ms) {
    init_search_tables();
    Result r;
    SearchCtx ctx{Clock::now(), time_budget_ms};

    // Iterative deepening: best move from each completed depth is preserved
    // even if the next depth gets aborted by the time check, so the caller
    // always sees a sane move.
    Score last_score = 0;
    for (int d = 1; d <= max_depth; ++d) {
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
        last_score  = s;
    }
    r.nodes = ctx.nodes;
    return r;
}

}  // namespace eclipse::ab
