// SPDX-License-Identifier: GPL-3.0-or-later
//
// data_filter: stream `<fen>;<cp>` training lines from stdin, drop positions
// that are not a useful static-eval target, and tag the rest with a sampling
// weight. Output is `<fen>;<cp>;<weight>` on stdout, which
// dev/scripts/dedup_and_chunk.py expands (after dedup) into `weight` copies.
//
// Why: Eclipse plays AB's root move (AbOverrideQMargin -2.0), and AB only asks
// the net about positions qsearch has already resolved. So, as with
// Stockfish's own training data, positions with a pending winning capture are
// noise: their label depends on a tactic the net is never asked to see
// statically. The 2026-11-07 game analysis (dev/sf_analysis/) also found most
// costly errors in 7-18-man endgames/late middlegames, so those get upweighted.
//
// Dropped:
//   - side to move in check
//   - side to move has a legal capture with SEE > 0, or any legal promotion
//   - unparseable FEN
// Weight (by men = total pieces incl. kings and pawns), overridable:
//   men <= --tb-men            -> 1  (tablebase territory at inference)
//   men <= --late-men          -> --late-weight      (default 12 -> 3)
//   men <= --transition-men    -> --transition-weight (default 18 -> 2)
//   otherwise                  -> 1
//
// Usage:
//   gzcat data/eval_training.txt.gz | build/tests/data_filter [opts] \
//       | gzip -1 > data/eval_training_quiet.txt.gz
// Summary counts go to stderr at the end.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "attacks.hpp"
#include "bitboard.hpp"
#include "movegen.hpp"
#include "position.hpp"
#include "see.hpp"
#include "zobrist.hpp"

namespace {

struct Opts {
    int tb_men = 6;
    int late_men = 12, late_weight = 3;
    int transition_men = 18, transition_weight = 2;
};

int arg_int(int& i, int argc, char** argv) {
    if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); std::exit(2); }
    return std::atoi(argv[++i]);
}

}  // namespace

int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--tb-men"))            o.tb_men = arg_int(i, argc, argv);
        else if (!std::strcmp(argv[i], "--late-men"))          o.late_men = arg_int(i, argc, argv);
        else if (!std::strcmp(argv[i], "--late-weight"))       o.late_weight = arg_int(i, argc, argv);
        else if (!std::strcmp(argv[i], "--transition-men"))    o.transition_men = arg_int(i, argc, argv);
        else if (!std::strcmp(argv[i], "--transition-weight")) o.transition_weight = arg_int(i, argc, argv);
        else { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }

    eclipse::zobrist::init();
    eclipse::init_attacks();
    std::ios::sync_with_stdio(false);

    long long n_in = 0, n_bad = 0, n_check = 0, n_tactical = 0, n_out = 0, n_weighted = 0;
    long long by_weight[16] = {};
    std::string line, out;
    out.reserve(1 << 20);
    eclipse::Position pos;
    eclipse::MoveList caps;

    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;
        ++n_in;
        const auto semi = line.find(';');
        if (semi == std::string::npos || !pos.set_from_fen(std::string_view(line).substr(0, semi))) {
            ++n_bad;
            continue;
        }
        if (pos.in_check()) { ++n_check; continue; }

        caps.clear();
        eclipse::generate_legal_captures(pos, caps);
        bool tactical = false;
        for (const eclipse::Move m : caps) {
            if (m.type() == eclipse::Move::Promotion || eclipse::see(pos, m) > 0) {
                tactical = true;
                break;
            }
        }
        if (tactical) { ++n_tactical; continue; }

        const int men = eclipse::popcount(pos.occupied());
        int w = 1;
        if (men > o.tb_men) {
            if      (men <= o.late_men)       w = o.late_weight;
            else if (men <= o.transition_men) w = o.transition_weight;
        }
        ++n_out;
        n_weighted += w;
        ++by_weight[w < 15 ? w : 15];

        // Keep only `<fen>;<cp>` (drop any trailing fields) and append weight.
        const auto semi2 = line.find(';', semi + 1);
        out.append(line, 0, semi2 == std::string::npos ? line.size() : semi2);
        out += ';';
        out += std::to_string(w);
        out += '\n';
        if (out.size() > (1 << 20) - 256) { std::fwrite(out.data(), 1, out.size(), stdout); out.clear(); }
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fflush(stdout);

    std::fprintf(stderr,
                 "data_filter: in=%lld bad=%lld in_check=%lld tactical=%lld kept=%lld (%.1f%%) "
                 "weighted_total=%lld\n",
                 n_in, n_bad, n_check, n_tactical, n_out, 100.0 * n_out / (n_in ? n_in : 1), n_weighted);
    for (int w = 1; w < 16; ++w)
        if (by_weight[w]) std::fprintf(stderr, "  weight %d: %lld\n", w, by_weight[w]);
    return 0;
}
