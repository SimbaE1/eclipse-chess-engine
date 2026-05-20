// SPDX-License-Identifier: GPL-3.0-or-later
#include "uci.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "attacks.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "policy.hpp"
#include "perft.hpp"
#include "position.hpp"
#include "search.hpp"
#include "tt.hpp"
#include "zobrist.hpp"

namespace eclipse::uci {

namespace {

constexpr const char* kEngineName    = "Eclipse";
constexpr const char* kEngineVersion = "0.0.1";
constexpr const char* kEngineAuthor  = "SimbaE11";

Position    g_pos;
SearchInfo  g_search_info;
std::thread g_search_thread;

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
        if (m.from() != from || m.to() != to) continue;
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
              << "id author " << kEngineAuthor << '\n';

#if defined(__AVX2__)
    std::cout << "info string SIMD: AVX2\n";
#elif defined(__ARM_NEON)
    std::cout << "info string SIMD: NEON\n";
#else
    std::cout << "info string SIMD: None (Scalar)\n";
#endif

    std::cout << "option name EvalFile type string default <empty>\n"
              << "option name PolicyFile type string default <empty>\n"
              << "option name Threads type spin default 1 min 1 max 128\n"
              << "option name Hash type spin default 16 min 1 max 16384\n"
              << "option name OverrideMargin type spin default 50 min 0 max 1000\n"
              << "option name AbThreads type spin default 1 min 0 max 128\n"
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
    } else if (name == "Hash") {
        g_tt.resize(std::atoi(value.c_str()));
    } else if (name == "OverrideMargin") {
        g_search_info.override_margin = std::atoi(value.c_str());
    } else if (name == "AbThreads") {
        const int n = std::atoi(value.c_str());
        g_search_info.ab_threads = std::clamp(n, 0, 128);
    }
    // Unknown options are silently ignored per UCI convention.
}

void cmd_isready() {
    zobrist::init();
    init_attacks();
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

    if (i < tok.size() && tok[i] == "moves") {
        ++i;
        // Single StateInfo we keep overwriting. We don't need to undo the
        // applied moves; we just need the final position.
        StateInfo st;
        while (i < tok.size()) {
            const Move m = parse_move(tok[i], g_pos);
            if (m.is_null()) break;
            g_pos.do_move(m, st);
            ++i;
        }
    }
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
        const int  divisor = std::clamp(mlh_our_moves, 5, 60);

        limits.time_ms = remain / divisor + inc * 4 / 5;
        if (limits.time_ms < 1) limits.time_ms = 1;
    }

    g_search_info.limits = limits;
    g_search_info.stop.store(false, std::memory_order_relaxed);
    
    join_search_thread();
    g_search_thread = std::thread([]() {
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
        else if (cmd == "ucinewgame") { join_search_thread(); g_pos = Position::startpos(); }
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
            g_search_info.limits.ponder = false;
            // Update start time to now so time management is relative to the ponderhit
            g_search_info.start_time = std::chrono::steady_clock::now();
        }
        else if (cmd == "quit")       { join_search_thread(); break; }
        // Unknown commands are silently ignored per the UCI spec.
    }
}

}  // namespace eclipse::uci
