// SPDX-License-Identifier: GPL-3.0-or-later
#include "ab.hpp"

#include <algorithm>
#include <array>
#include <chrono>

#include "movegen.hpp"
#include "tt.hpp"
#include "types.hpp"

namespace eclipse::ab {

namespace {

using Clock = std::chrono::steady_clock;

// MVV-LVA piece values for ordering only — not the eval. Indexed by PieceType.
// King is included for completeness (we never capture it, but checks during
// qsearch can be ordered relative to others).
constexpr std::array<int, 8> kPieceValue = {
    0,    // NoPieceType
    100,  // Pawn
    320,  // Knight
    330,  // Bishop
    500,  // Rook
    900,  // Queen
    20000,// King
    0,
};

struct SearchCtx {
    Clock::time_point start;
    std::int64_t       budget_ms;
    std::int64_t       nodes;
    bool               aborted;

    // Killer moves: two per ply.
    std::array<std::array<Move, 2>, 64> killers{};

    // History heuristic: [color][from][to].
    std::uint32_t history[2][64][64]{};

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
// then history.
int order_score(const Position& pos, Move m, int ply, const SearchCtx& ctx) {
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
        s = static_cast<int>(ctx.history[pos.side_to_move()][m.from()][m.to()]);
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

Score qsearch(Position& pos, Score alpha, Score beta, int ply, SearchCtx& ctx) {
    ++ctx.nodes;
    if (ctx.time_up()) { ctx.aborted = true; return 0; }

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
    for (int i = 0; i < caps.size; ++i) scores[i] = order_score(pos, caps[i], ply, ctx);
    for (int i = 1; i < caps.size; ++i) {
        for (int j = i; j > 0 && scores[j] > scores[j - 1]; --j) {
            std::swap(scores[j], scores[j - 1]);
            std::swap(caps[j],   caps[j - 1]);
        }
    }

    for (int i = 0; i < caps.size; ++i) {
        StateInfo st;
        pos.do_move(caps[i], st);
        const Score s = -qsearch(pos, -beta, -alpha, ply + 1, ctx);
        pos.undo_move(caps[i], st);
        if (ctx.aborted) return 0;
        if (s >= beta) return beta;
        if (s > alpha) alpha = s;
    }
    return alpha;
}

Score negamax(Position& pos, int depth, Score alpha, Score beta, int ply,
              Move& out_best, SearchCtx& ctx) {
    ++ctx.nodes;
    out_best = MoveNone;

    if (ctx.time_up()) { ctx.aborted = true; return 0; }
    if (depth <= 0) return qsearch(pos, alpha, beta, ply, ctx);

    const Score static_eval = evaluate(pos);

    // Null-move pruning: if we are so far ahead that passing a turn still
    // stays above beta, we can prune this branch. Skip if in check or if
    // depth is low to avoid zugzwang risks or shallow misses.
    if (!pos.in_check() && depth >= 3 && static_eval >= beta) {
        StateInfo st;
        pos.do_null_move(st);
        const Score s = -negamax(pos, depth - 1 - 3, -beta, -beta + 1, ply + 1, out_best, ctx);
        pos.undo_null_move(st);
        if (s >= beta) return beta;
    }

    const Score alpha_orig = alpha;
    TTEntry tt_entry;
    Move    tt_move = MoveNone;
    if (g_tt.probe(pos.key(), tt_entry)) {
        tt_move = tt_entry.move;
        if (tt_entry.depth >= depth) {
            const Score s = tt_entry.score_from_tt(tt_entry.score, ply);
            if (tt_entry.flag == TT_EXACT) return s;
            if (tt_entry.flag == TT_LOWERBOUND) alpha = std::max(alpha, s);
            if (tt_entry.flag == TT_UPPERBOUND) beta  = std::min(beta, s);
            if (alpha >= beta) return s;
        }
    }

    MoveList moves;
    generate_legal_moves(pos, moves);

    if (moves.size == 0) {
        // Checkmate or stalemate. Mate score is depth-adjusted so the
        // shallowest mate is preferred. Stalemate = draw.
        return pos.in_check() ? -static_cast<Score>(kMateScore - ply) : Score{0};
    }

    // Order moves. TT move comes first.
    std::array<int, 256> scores{};
    for (int i = 0; i < moves.size; ++i) {
        if (moves[i] == tt_move) scores[i] = 3'000'000;
        else                     scores[i] = order_score(pos, moves[i], ply, ctx);
    }
    for (int i = 1; i < moves.size; ++i) {
        for (int j = i; j > 0 && scores[j] > scores[j - 1]; --j) {
            std::swap(scores[j], scores[j - 1]);
            std::swap(moves[j],  moves[j - 1]);
        }
    }

    Score best = -kInfinite;
    Move  best_move = MoveNone;
    for (int i = 0; i < moves.size; ++i) {
        StateInfo st;
        pos.do_move(moves[i], st);
        
        Move  dummy;
        Score s;
        if (i == 0) {
            // PV node: search with full window.
            s = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, dummy, ctx);
        } else {
            // LMR: reduce depth for late-ordered quiet moves.
            int reduction = 0;
            if (depth >= 3 && i >= 4 && !pos.in_check() &&
                pos.piece_on(moves[i].to()) == NoPiece &&
                moves[i].type() != Move::EnPassant &&
                moves[i] != ctx.killers[ply][0] && moves[i] != ctx.killers[ply][1]) {
                reduction = 1;
            }

            // Non-PV node: search with null window first.
            s = -negamax(pos, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, dummy, ctx);
            if (s > alpha && reduction > 0) {
                // Failed high on reduced search: re-search with full depth but null window.
                s = -negamax(pos, depth - 1, -alpha - 1, -alpha, ply + 1, dummy, ctx);
            }
            if (s > alpha && s < beta) {
                // Failed high on null window: re-search with full window.
                s = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, dummy, ctx);
            }
        }

        pos.undo_move(moves[i], st);
        if (ctx.aborted) return 0;
        if (s > best) { best = s; best_move = moves[i]; }
        if (s > alpha) alpha = s;
        if (alpha >= beta) {
            // Beta cutoff. Update heuristics for quiet moves.
            if (pos.piece_on(moves[i].to()) == NoPiece && moves[i].type() != Move::EnPassant) {
                if (ply < 64) {
                    if (moves[i] != ctx.killers[ply][0]) {
                        ctx.killers[ply][1] = ctx.killers[ply][0];
                        ctx.killers[ply][0] = moves[i];
                    }
                }
                const Color us = pos.side_to_move();
                ctx.history[us][moves[i].from()][moves[i].to()] += depth * depth;
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
    Result r;
    SearchCtx ctx{Clock::now(), time_budget_ms, 0, false};

    // Iterative deepening: best move from each completed depth is preserved
    // even if the next depth gets aborted by the time check, so the caller
    // always sees a sane move.
    Score last_score = 0;
    for (int d = 1; d <= max_depth; ++d) {
        Move best_at_d;
        Score s;
        
        if (d >= 4) {
            Score alpha = last_score - 50;
            Score beta  = last_score + 50;
            while (true) {
                s = negamax(pos, d, alpha, beta, 0, best_at_d, ctx);
                if (ctx.aborted) break;
                if (s <= alpha) {
                    alpha = std::max(-kInfinite, alpha - 100);
                } else if (s >= beta) {
                    beta = std::min(kInfinite, beta + 100);
                } else {
                    break;
                }
            }
        } else {
            s = negamax(pos, d, -kInfinite, kInfinite, 0, best_at_d, ctx);
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
