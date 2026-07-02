// SPDX-License-Identifier: GPL-3.0-or-later
#include "uci.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "attacks.hpp"
#include "mcts.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "policy.hpp"
#include "perft.hpp"
#include "position.hpp"
#include "search.hpp"
#include "syzygy.hpp"
#include "thread_util.hpp"
#include "tt.hpp"
#include "zobrist.hpp"

namespace eclipse::uci {

namespace {

constexpr const char* kEngineName    = "Eclipse";
constexpr const char* kEngineVersion = "1.0.0";
constexpr const char* kEngineAuthor  = "SimbaE11";

Position    g_pos;
SearchInfo  g_search_info;

// True once the user sets AbThreads explicitly, after which the Threads
// handler stops auto-deriving the AB-thread count from the total.
static bool g_ab_threads_explicit = false;
// Whether the current session is a Chess960 game. Affects move notation:
// castling moves are output as king-to-rook ("e1h1") instead of king-to-
// destination ("e1g1"), and parsed accordingly.
static bool g_chess960 = false;
// Large stack: the search runs negamax/qsearch (recursive) directly on this
// thread, which would overflow the default secondary-thread stack. See
// thread_util.hpp.
BigThread   g_search_thread;

void join_search_thread() {
    if (g_search_thread.joinable()) {
        g_search_info.stop.store(true, std::memory_order_relaxed);
        g_search_thread.join();
    }
}

std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string t;
    while (iss >> t) out.push_back(std::move(t));
    return out;
}

// Convert "e2e4" / "e7e8q" to the matching legal move, or MoveNone if no such
// move exists from `pos`.
//
// Castling encoding depends on the mode:
//   Standard (UCI_Chess960=false): input is king-to-destination ("e1g1"/"e1c1").
//     Internally castling is encoded as king-to-rook, so we detect when the
//     input squares match a standard castling destination and find the matching
//     legal castling move instead.
//   Chess960 (UCI_Chess960=true): input is king-to-rook ("e1h1"/"e1a1").
//     This matches the internal encoding directly.
Move parse_move(const std::string& s, Position& pos) {
    if (s.size() < 4) return MoveNone;
    const Square from = square_from_string(s.substr(0, 2).c_str());
    const Square to   = square_from_string(s.substr(2, 2).c_str());
    if (!is_valid(from) || !is_valid(to)) return MoveNone;

    PieceType promo = NoPieceType;
    if (s.size() >= 5) {
        switch (s[4]) {
            case 'q': promo = Queen;  break;
            case 'r': promo = Rook;   break;
            case 'b': promo = Bishop; break;
            case 'n': promo = Knight; break;
            default: return MoveNone;
        }
    }

    MoveList legal;
    generate_legal_moves(pos, legal);
    for (const Move m : legal) {
        if (m.from() != from) continue;
        if (m.type() == Move::Castling) {
            // In Chess960 mode the GUI sends king-to-rook, matching the internal
            // encoding (m.to() == rook square). In standard mode the GUI sends
            // king-to-destination; translate: kingside → G-file, queenside → C-file.
            Square effective_to;
            if (g_chess960) {
                effective_to = m.to();  // rook square, matches input directly
            } else {
                const bool kingside = (file_of(m.to()) > file_of(from));
                effective_to = make_square(kingside ? FileG : FileC, rank_of(from));
            }
            if (effective_to != to) continue;
        } else {
            if (m.to() != to) continue;
        }
        if (m.type() == Move::Promotion) {
            if (m.promotion_piece() != promo) continue;
        } else if (promo != NoPieceType) {
            continue;
        }
        return m;
    }
    return MoveNone;
}

void cmd_uci() {
    std::cout << "id name " << kEngineName << ' ' << kEngineVersion << '\n'
              << "id author " << kEngineAuthor << '\n'
              << "option name EvalFile type string default <empty>\n"
              << "option name PolicyFile type string default <empty>\n"
              << "option name Threads type spin default 1 min 1 max 128\n"
              << "option name Hash type spin default 256 min 1 max 65536\n"
              << "option name MctsHash type spin default 64 min 1 max 65536\n"
              << "option name OverrideMargin type spin default 50 min 0 max 1000\n"
              << "option name AbThreads type spin default 1 min 0 max 128\n"
              << "option name Cpuct type string default 1.70\n"
              << "option name FpuOffset type string default 0.20\n"
              << "option name PolicyDepth type spin default 2 min -1 max 64\n"
              << "option name SelectVisitFrac type string default 0.60\n"
              << "option name SelectQMargin type string default 0.02\n"
              << "option name Ponder type check default true\n"
              << "option name UCI_Ponder type check default true\n"
              << "option name UCI_Chess960 type check default false\n"
              << "option name SyzygyPath type string default <empty>\n"
              << "uciok" << std::endl;
}

// `setoption name <name> value <value...>` - value tokens may run to EOL.
void cmd_setoption(const std::vector<std::string>& tok) {
    std::size_t i = 1;
    if (i >= tok.size() || tok[i] != "name") return;
    ++i;
    std::string name;
    while (i < tok.size() && tok[i] != "value") {
        if (!name.empty()) name += ' ';
        name += tok[i++];
    }
    std::string value;
    if (i < tok.size() && tok[i] == "value") {
        ++i;
        while (i < tok.size()) {
            if (!value.empty()) value += ' ';
            value += tok[i++];
        }
    }

    if (name == "EvalFile") {
        if (value.empty() || value == "<empty>") return;
        nnue::load(value);
    } else if (name == "PolicyFile") {
        if (value.empty() || value == "<empty>") return;
        policy::load(value);
    } else if (name == "PolicyMode") {
        // "nnue" (default): each child is scored with NNUE static eval and
        // priors come from softmax of the score differential. Microseconds
        // per expansion. "onnx": legacy Lc0 transformer path; slower by ~3
        // orders of magnitude but available for A/B testing.
        if (value == "nnue" || value == "NNUE") {
            policy::set_mode(policy::Mode::Nnue);
        } else if (value == "onnx" || value == "ONNX") {
            policy::set_mode(policy::Mode::Onnx);
        }
    } else if (name == "Threads") {
        const int n = std::atoi(value.c_str());
        g_search_info.threads = std::clamp(n, 1, 128);
        // Derived AbThreads default: ~1 AB (Lazy-SMP) thread per 4 total,
        // floor, min 1. So 4 threads -> 1 AB / 3 MCTS (unchanged), 8 -> 2 AB /
        // 6 MCTS, etc. Only applied while the user hasn't set AbThreads
        // explicitly; an explicit AbThreads always wins regardless of order.
        if (!g_ab_threads_explicit)
            g_search_info.ab_threads = std::max(1, g_search_info.threads / 4);
    } else if (name == "Hash") {
        g_tt.resize(std::atoi(value.c_str()));
    } else if (name == "MctsHash") {
        mcts::g_mcts_tt.resize(static_cast<std::size_t>(std::atoi(value.c_str())));
    } else if (name == "OverrideMargin") {
        g_search_info.override_margin = std::atoi(value.c_str());
    } else if (name == "AbThreads") {
        const int n = std::atoi(value.c_str());
        g_search_info.ab_threads = std::clamp(n, 0, 128);
        g_ab_threads_explicit = true;  // pin it; stop deriving from Threads
    } else if (name == "Cpuct") {
        char* end = nullptr;
        const float f = std::strtof(value.c_str(), &end);
        if (end != value.c_str() && f > 0.0f && f < 100.0f) mcts::g_cpuct = f;
    } else if (name == "FpuOffset") {
        char* end = nullptr;
        const float f = std::strtof(value.c_str(), &end);
        if (end != value.c_str() && f > -1.0f && f < 2.0f) mcts::g_fpu_offset = f;
    } else if (name == "PolicyDepth") {
        // Plies from the root that get NNUE-scored policy priors (-1 disables).
        const int n = std::atoi(value.c_str());
        if (n >= -1 && n <= 64) mcts::g_policy_depth = n;
    } else if (name == "SelectVisitFrac") {
        char* end = nullptr;
        const float f = std::strtof(value.c_str(), &end);
        if (end != value.c_str() && f >= 0.0f && f <= 1.0f) mcts::g_select_visit_frac = f;
    } else if (name == "SelectQMargin") {
        char* end = nullptr;
        const float f = std::strtof(value.c_str(), &end);
        if (end != value.c_str() && f >= 0.0f && f < 2.0f) mcts::g_select_q_margin = f;
    } else if (name == "SyzygyPath") {
        syzygy::init(value);
    } else if (name == "UCI_Chess960") {
        g_chess960 = (value == "true");
        set_chess960_mode(g_chess960);
    }
    // UCI_Ponder is a capability advertisement; no runtime state to set.
    // Unknown options are silently ignored per UCI convention.
}

void cmd_isready() {
    zobrist::init();
    init_attacks();
#if defined(__AVX512BW__) && !defined(ECLIPSE_NO_AVX512)
    // Must come before the __AVX2__ check: -march=native defines both macros,
    // and the NNUE hot paths use the AVX-512 ladder under this exact guard.
    std::cout << "info string SIMD: AVX-512\n";
#elif defined(__AVX2__)
    std::cout << "info string SIMD: AVX2\n";
#elif defined(__ARM_NEON)
    std::cout << "info string SIMD: NEON\n";
#else
    std::cout << "info string SIMD: None (Scalar)\n";
#endif
    std::cout << "readyok" << std::endl;
}

void cmd_position(const std::vector<std::string>& tok) {
    if (tok.size() < 2) return;
    std::size_t i = 1;

    if (tok[i] == "startpos") {
        g_pos = Position::startpos();
        ++i;
    } else if (tok[i] == "fen") {
        std::string fen;
        ++i;
        // FEN occupies up to 6 whitespace-separated fields. Stop early if we
        // hit `moves` so it doesn't get swallowed into the FEN.
        for (int field = 0; field < 6 && i < tok.size() && tok[i] != "moves";
             ++field, ++i) {
            if (!fen.empty()) fen += ' ';
            fen += tok[i];
        }
        g_pos.set_from_fen(fen);
    } else {
        return;
    }

    // Record the Zobrist key of every position from the start of the move list
    // so the search can detect repetitions against the GAME, not just within
    // its own tree (see Position::repeats_game_history). game_keys.back() is the
    // current root; earlier entries are the prior game positions.
    std::vector<std::uint64_t> game_keys;
    game_keys.push_back(g_pos.key());

    if (i < tok.size() && tok[i] == "moves") {
        ++i;
        // Single StateInfo we keep overwriting. We don't need to undo the
        // applied moves; we just need the final position.
        StateInfo st;
        while (i < tok.size()) {
            const Move m = parse_move(tok[i], g_pos);
            if (m.is_null()) break;
            g_pos.do_move(m, st);
            game_keys.push_back(g_pos.key());
            ++i;
        }
    }

    // Attach the pre-root history: the positions BEFORE the current root that
    // are still reachable by reversible moves (within the halfmove clock).
    // Positions across the last irreversible move can never repeat the current
    // one, so slicing to that window keeps the per-node scan cheap. The current
    // root (game_keys.back()) is excluded — the search's own tables own it.
    const int n_root = static_cast<int>(game_keys.size()) - 1;
    const int window = std::min(n_root, g_pos.halfmove_clock());
    auto hist = std::make_shared<std::vector<std::uint64_t>>(
        game_keys.begin() + (n_root - window), game_keys.begin() + n_root);
    g_pos.set_rep_history(std::move(hist));
}

void cmd_go(const std::vector<std::string>& tok) {
    SearchLimits limits;
    // Time-control parsing keeps wtime/btime/winc/binc and decides how much
    // we may spend based on side_to_move. The /30 + 0.8*inc formula is a
    // placeholder - real time management lands later.
    int  movestogo = 30;
    int  wtime = 0, btime = 0, winc = 0, binc = 0;
    bool have_tc = false;

    for (std::size_t i = 1; i < tok.size(); ++i) {
        const std::string& t = tok[i];
        auto take = [&]() -> int {
            return (i + 1 < tok.size()) ? std::atoi(tok[++i].c_str()) : 0;
        };
        if      (t == "depth")     limits.depth   = take();
        else if (t == "movetime")  limits.time_ms = take();
        else if (t == "nodes")     limits.nodes   = take();
        else if (t == "infinite")  limits.infinite = true;
        else if (t == "wtime")    { wtime = take();     have_tc = true; }
        else if (t == "btime")    { btime = take();     have_tc = true; }
        else if (t == "winc")     { winc  = take(); }
        else if (t == "binc")     { binc  = take(); }
        else if (t == "movestogo"){ movestogo = std::max(1, take()); }
        else if (t == "ponder")    limits.ponder = true;
        else if (t == "perft") {
            const int depth = take();
            const auto split = perft_divided(g_pos, depth);
            std::uint64_t total = 0;
            for (const auto& e : split) {
                std::cout << e.move.to_uci() << ": " << e.nodes << '\n';
                total += e.nodes;
            }
            std::cout << "\nNodes searched: " << total << std::endl;
            return;
        }
    }

    if (have_tc && limits.time_ms == 0 && !limits.infinite) {
        const bool   white  = (g_pos.side_to_move() == White);
        const int    remain = white ? wtime : btime;
        const int    inc    = white ? winc  : binc;

        // MLH-driven time allocation. The Lc0 net's moves-left head estimates
        // how many half-moves remain in the game; we halve it (we only spend
        // on every other ply) and use it instead of the static movestogo
        // divisor. Clamp [5, 60] so a bad MLH prediction can't blow up our
        // budget - 5 keeps us from burning all time on a sharp tactic, 60
        // keeps us from starving the next move in a long endgame.
        const auto root = policy::get_root_info(g_pos);
        const int  mlh_our_moves = static_cast<int>(root.mlh_plies / 2.0f);
        // Floor of 40 (was 20, was 8, was 5): never plan to spend more than
        // ~1/40 of the clock on a routine move. The floor matters when the MLH
        // head UNDER-predicts how long the game will last: it then reports few
        // moves remaining, the divisor collapses to the floor, and we budget
        // that fraction of the clock EVERY move.
        //
        // The soft budget remain/D + 0.8*inc drives the clock to an equilibrium
        // of D*0.2*inc, where each move spends exactly the increment and the
        // clock holds flat. Higher D => smaller per-move spend (faster play) AND
        // a higher resting clock (more safety). D=8 => 12.5%/move, bled to flag
        // (2026-06-18 vs SF lvl5). D=20 => ~5%/move, rests at only 60s @ +15s.
        // D=40 => ~2.5%/move and rests at ~120s @ +15s: noticeably faster moves
        // and a comfortable cushion, while an accurate (larger) MLH estimate can
        // still spend a bit more early. Upper clamp 60 caps the richest case.
        const int  divisor = std::clamp(mlh_our_moves, 40, 60);

        limits.time_ms = remain / divisor + inc * 4 / 5;

        // Flat early-game cap. The remain/divisor term is ~proportional to the
        // clock (divisor sits pinned near its floor for most of a game), so the
        // soft budget is LARGEST early and decays as the clock drains — the
        // front-loaded "logarithmic" curve where the opening costs the most. By
        // design it drains 600s->~120s spending well over the increment per move
        // early, which both wastes time on non-critical opening moves and is what
        // bled won games to a flag. Clamp the routine per-move spend to a small
        // multiple of the increment so the early game is FLAT instead of
        // front-loaded: with +inc we can sustain ~inc/move indefinitely, so a
        // routine move spends ~1.5x that and the clock drains gently and evenly.
        // Critical moves still extend up to the hard ceiling (kHardBudgetMult x
        // soft) below. Increment-only lever: in sudden-death (inc==0) there is no
        // sustainable rate, so keep the pure remain/divisor policy. The low-clock
        // equilibrium (clock resting at ~0.2*divisor*inc, spending exactly the
        // increment) sits below this cap, so the no-flag behaviour is unchanged.
        constexpr int kEarlySoftIncMultNum = 3;  // 3/2 = 1.5x the increment
        constexpr int kEarlySoftIncMultDen = 2;
        if (inc > 0) {
            const int early_soft_cap = inc * kEarlySoftIncMultNum / kEarlySoftIncMultDen;
            if (limits.time_ms > early_soft_cap) limits.time_ms = early_soft_cap;
        }

        // Safety: never burn more than 1/3 of remaining time on one move.
        const int safety_cap = remain / 3;
        if (limits.time_ms > safety_cap) limits.time_ms = safety_cap;

        // Hard wall-clock ceiling for the WHOLE move (every phase combined),
        // not just the soft target above. search() enforces it as a single
        // absolute deadline (see SearchLimits::hard_limit_ms), so the sum of
        // the main search + validation + AB cross-check/tactic probes +
        // extensions can never overrun it. A fixed latency margin is kept below
        // the real remaining time so the bestmove reaches lichess before the
        // flag even with network + I/O lag.
        //
        // CRITICAL (2026-06-20 bleed fix): the ceiling is a small MULTIPLE OF
        // THE SOFT BUDGET, not a fraction of the whole clock. The old code set
        // it to remain/3, which in a 10+3 game is ~200s on move 1 — so the
        // extension/validation/cross-check machinery (which fires on any
        // "unsettled" eval, common positions for the current net) could legally
        // spend up to a THIRD of the entire clock on a single move. No single
        // move flagged, but a handful of 40-79s moves per game bled the clock
        // from 600s to 0 while the opponent played ~12s/move (real loss
        // G0woH69l: Eclipse averaged ~20.6s/move and flagged). Tying the
        // ceiling to the soft budget means a hard move costs at most
        // ~kHardBudgetMult x a routine move and scales DOWN with the clock, so
        // the per-move spend can never run away. remain/3 is kept only as an
        // absolute backstop that the soft-multiple is normally far below.
        constexpr int kLatencyMarginMs = 300;
        constexpr int kHardBudgetMult  = 2;   // a critical move may cost up to 2x a routine one
        const int hard_from_soft = static_cast<int>(limits.time_ms) * kHardBudgetMult;
        const int hard_cap = std::min({hard_from_soft, safety_cap, remain - kLatencyMarginMs});
        limits.hard_limit_ms = std::max(1, hard_cap);

        // No-flag guarantee for the low-clock regime. The soft-budget equilibrium
        // above only holds in the limit; a sharp position late in the game can
        // still trigger extensions up to the 1/3 safety_cap, and 1/3 of a low
        // clock spends MORE than the increment refunds — so the clock keeps
        // draining and eventually flags (the won-game flag we kept seeing). When
        // we are genuinely low (within ~8 increments), clamp the WHOLE move to
        // 2/3 of the increment. The deadline bounds the search phases, but a
        // small tail (final AB verify/reconciliation + bestmove I/O, measured at
        // ~1.5s) lands on top, so capping at 0.9*inc would only net ~+0.1s. At
        // 2/3*inc the real spend (~2/3*inc + tail) stays clearly under the
        // increment, so every move from here nets positive and the clock
        // strictly recovers — flagging is impossible regardless of search phase.
        // With +15s that's a ~10s cap (~11-12s real). Only engages when low, so
        // normal play is unaffected.
        if (inc > 0 && remain < 8 * inc) {
            limits.hard_limit_ms = std::min<std::int64_t>(limits.hard_limit_ms, inc * 2 / 3);
        }

        // The soft target must sit under the hard ceiling, otherwise the main
        // phase alone would try to consume the entire budget and leave nothing
        // for reconciliation (or worse, overshoot the deadline mid-phase).
        if (limits.time_ms > limits.hard_limit_ms) limits.time_ms = limits.hard_limit_ms;

        // Slack the reconciliation may borrow, measured against the SAME hard
        // ceiling the deadline enforces — so extensions stay inside it instead
        // of stacking on top of the safety_cap as they used to.
        limits.extra_budget_ms = std::max<std::int64_t>(0, limits.hard_limit_ms - limits.time_ms);

        // Move-overhead subtraction. KNOWN LIMITATION: Eclipse has a fixed
        // per-move floor of ~120 ms even with a 1 ms budget — the cost of the
        // root MLH/policy probe, the mandatory first MCTS batch across all
        // threads, the mate-in-1 sweep, and move I/O. None of that is
        // interruptible by the time budget. So at ultra-bullet (e.g. 1+0.1)
        // where the increment (100 ms) is below this floor, the engine
        // net-loses time every move and WILL eventually flag — it physically
        // cannot move faster than the floor. We subtract kMoveOverheadMs to
        // stay safe at any normal TC (blitz/rapid and 5+3 bullet are fine);
        // ultra-bullet flagging is accepted, not a bug we can fix without
        // gutting the search floor.
        constexpr int kMoveOverheadMs = 120;
        limits.time_ms -= kMoveOverheadMs;
        if (limits.time_ms < 1) limits.time_ms = 1;
    }

    g_search_info.limits = limits;
    join_search_thread();
    g_search_info.stop.store(false, std::memory_order_relaxed);
    g_search_thread = BigThread([]() {
        search(g_pos, g_search_info);
    });
}

}  // namespace

void loop() {
    zobrist::init();
    init_attacks();
    g_pos = Position::startpos();

    std::string line;
    while (std::getline(std::cin, line)) {
        const auto tok = tokenize(line);
        if (tok.empty()) continue;
        const std::string& cmd = tok[0];

        if      (cmd == "uci")        cmd_uci();
        else if (cmd == "isready")    cmd_isready();
        else if (cmd == "ucinewgame") { join_search_thread(); g_pos = Position::startpos(); g_tt.clear(); mcts::g_mcts_tt.clear(); }
        else if (cmd == "setoption")  { join_search_thread(); cmd_setoption(tok); }
        else if (cmd == "position")   { join_search_thread(); cmd_position(tok); }
        else if (cmd == "go")         cmd_go(tok);
        else if (cmd == "bench") {
            g_pos = Position::startpos();
            g_search_info.limits = SearchLimits{};
            g_search_info.limits.nodes = 100000;
            g_search_info.limits.depth = 0;
            g_search_info.stop.store(false, std::memory_order_relaxed);
            
            const auto start = std::chrono::steady_clock::now();
            search(g_pos, g_search_info);
            const auto end = std::chrono::steady_clock::now();
            
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            const auto nps = (ms > 0) ? (g_search_info.nodes_searched.load() * 1000 / ms) : 0;
            
            std::cout << "Benchmark: " << g_search_info.nodes_searched.load() << " nodes in " << ms << "ms (" << nps << " nps)" << std::endl;
        }
        else if (cmd == "stop")       g_search_info.stop.store(true, std::memory_order_relaxed);
        else if (cmd == "ponderhit") {
            // Signal the search thread via the atomic only — limits.ponder and
            // start_time are read concurrently by time_up()/worker_loop() on
            // the search thread, so writing them here from the main thread is
            // a data race (see SearchInfo::ponderhit_at_ms in search.hpp).
            const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            g_search_info.ponderhit_at_ms.store(now_ms, std::memory_order_release);
        }
        else if (cmd == "quit")       { join_search_thread(); break; }
        // Unknown commands are silently ignored per the UCI spec.
    }
    // EOF path: stop any in-progress search then join. A ponder search with
    // no ponderhit runs indefinitely — "finish naturally" never happens, so
    // just joining (without stop) would deadlock when the game process exits
    // and closes the pipe mid-ponder.
    join_search_thread();
}

}  // namespace eclipse::uci
