// SPDX-License-Identifier: GPL-3.0-or-later
#include "move.hpp"

namespace eclipse {

namespace {
bool g_chess960_mode = false;
}

void set_chess960_mode(bool v) noexcept { g_chess960_mode = v; }
bool get_chess960_mode() noexcept       { return g_chess960_mode; }

std::string square_to_string(Square s) {
    if (!is_valid(s)) return "??";
    char buf[3] = {
        static_cast<char>('a' + file_of(s)),
        static_cast<char>('1' + rank_of(s)),
        '\0'};
    return std::string(buf);
}

Square square_from_string(const char* s) {
    if (s == nullptr) return SquareNone;
    if (s[0] < 'a' || s[0] > 'h') return SquareNone;
    if (s[1] < '1' || s[1] > '8') return SquareNone;
    return make_square(File(s[0] - 'a'), Rank(s[1] - '1'));
}

std::string Move::to_uci() const {
    if (is_null()) return "0000";
    if (type() == Castling && !g_chess960_mode) {
        // Internally castling is encoded as (king_from, rook_from). For standard
        // UCI output, translate to the king's destination square instead.
        const Square kfrom = from();
        const Square rook  = to();
        const File   kdest_file = (file_of(rook) > file_of(kfrom)) ? FileG : FileC;
        const Square kdest = make_square(kdest_file, rank_of(kfrom));
        return square_to_string(kfrom) + square_to_string(kdest);
    }
    std::string out = square_to_string(from()) + square_to_string(to());
    if (type() == Promotion) {
        static const char kPromoChar[] = {'n', 'b', 'r', 'q'};
        out += kPromoChar[(data_ >> 12) & 0x3u];
    }
    return out;
}

}  // namespace eclipse
