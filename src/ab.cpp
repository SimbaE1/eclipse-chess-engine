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
        // MVV-LVA: high-value victim, low-value aggressor first.
        s = 2'000'000 + kPieceValue[vict] * 10 - kPieceValue[aggr];
    } else if (m.type() == Move::EnPassant) {
        s = 2'000'000 + kPieceValue[Pawn] * 10 - kPieceValue[Pawn];
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
        // SEE pruning: skip losing captures unless we are in check.
        if (!pos.in_check() && !see_ge(pos, m, 0)) continue;

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

    const Score static_eval = evaluate(pos);
    
    // Reverse Futility Pruning (RFP): also known as Static Null Move Pruning.
    // If we are at a low depth and the evaluation is far above beta, we can
    // skip the search and return beta.
    if (!pos.in_check() && depth <= 6 && static_eval >= beta + 120 * depth) {
        return static_eval;
    }

    // Null-move pruning: if we are so far ahead that passing a turn still
    // stays above beta, we can prune this branch. Skip if in check, if depth
    // is low, or if the side-to-move has only pawns + king (zugzwang risk
    // in K+P endings — a tempo can be the difference between win and draw,
    // and giving the opponent a free move actively misrepresents the
    // position there).
    if (!pos.in_check() && depth >= 3 && static_eval >= beta) {
        const Color us         = pos.side_to_move();
        const Bitboard minors  = pos.pieces(us, Knight) | pos.pieces(us, Bishop);
        const Bitboard majors  = pos.pieces(us, Rook)   | pos.pieces(us, Queen);
        const Bitboard non_p   = minors | majors;
        // Skip null move in positions with only minor pieces — K+B vs K and
        // K+N vs K can exhibit zugzwang and the tempo gift actively misleads
        // the search. Rooks/queens make zugzwang essentially impossible, and
        // ≥2 minor pieces are also safe enough.
        const bool safe_for_nm = majors || (non_p && (non_p & (non_p - 1)));
        if (safe_for_nm) {
            StateInfo st;
            pos.do_null_move(st);
            Move dummy;
            const Score s = -negamax(pos, depth - 1 - 3, -beta, -beta + 1, ply + 1, dummy, ctx, MoveNone, MoveNone);
            pos.undo_null_move(st);
            if (s >= beta) return beta;
        }
    }

    const Score alpha_orig = alpha;
    TTEntry tt_entry;
    Move    tt_move = MoveNone;
    if (excluded == MoveNone && g_tt.probe(pos.key(), tt_entry)) {
        tt_move = tt_entry.move;
        if (tt_entry.depth >= depth) {
            const Score s = tt_entry.score_from_tt(tt_entry.score, ply);
            if (tt_entry.flag == TT_EXACT) return s;
            if (tt_entry.flag == TT_LOWERBOUND) alpha = std::max(alpha, s);
            if (tt_entry.flag == TT_UPPERBOUND) beta  = std::min(beta, s);
            if (alpha >= beta) return s;
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
        return pos.in_check() ? -static_cast<Score>(kMateScore - ply) : Score{0};
    }

    // Futility pruning: if static eval is far below alpha at low depth, quiet
    // moves cannot realistically raise it so we skip them (captures + checks
    // are still searched).
    static constexpr Score kFutilityMargin[4] = {0, 150, 300, 500};
    const bool can_futility_prune =
        !pos.in_check()
        && depth <= 3
        && excluded == MoveNone
        && alpha > -kMateInMaxPly && alpha < kMateInMaxPly
        && static_eval + kFutilityMargin[depth] <= alpha;

    // LMP: at depth 1-2, after trying N quiet moves, skip the rest.
    // Not applied in check (we must search all evasions) or near mate scores.
    static constexpr int kLmpCount[3] = {0, 8, 15};
    const bool can_lmp =
        !pos.in_check()
        && depth <= 2
        && excluded == MoveNone
        && alpha > -kMateInMaxPly && alpha < kMateInMaxPly;
    int quiet_tried = 0;

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
        if (is_quiet) ++quiet_tried;

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
            if (depth >= 3 && i >= 3 && !pos.in_check() && is_quiet) {
                const int d_idx = std::min(depth, 63);
                const int i_idx = std::min(i, 255);
                reduction = static_cast<int>(g_lmr_table[d_idx][i_idx]);

                // Reduce less if it's a killer or has good history.
                if (m == ctx.killers[static_cast<std::size_t>(ply)][0] ||
                    m == ctx.killers[static_cast<std::size_t>(ply)][1]) {
                    reduction -= 1;
                }

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
            // Beta cutoff. Update heuristics for quiet moves.
            if (is_quiet) {
                if (ply < 64) {
                    const std::size_t pidx = static_cast<std::size_t>(ply);
                    if (m != ctx.killers[pidx][0]) {
                        ctx.killers[pidx][1] = ctx.killers[pidx][0];
                        ctx.killers[pidx][0] = m;
                    }
                }
                const Color us = pos.side_to_move();
                ctx.history[us][m.from()][m.to()] += static_cast<std::int32_t>(depth * depth);

                // History aging via right-shift over the flat backing array.
                if (ctx.history[us][m.from()][m.to()] > 1'000'000) {
                    std::int32_t* base = &ctx.history[0][0][0];
                    for (int k = 0; k < 2 * 64 * 64; ++k) base[k] >>= 1;
                }

                // Countermove: record that m refutes prev_move.
                if (prev_move != MoveNone) {
                    const Color opp = us == White ? Black : White;
                    ctx.countermove[opp][prev_move.from()][prev_move.to()] = m;
                }
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
