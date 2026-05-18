// SPDX-License-Identifier: GPL-3.0-or-later
#include "move.hpp"

namespace eclipse {

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
    std::string out = square_to_string(from()) + square_to_string(to());
    if (type() == Promotion) {
        static const char kPromoChar[] = {'n', 'b', 'r', 'q'};
        out += kPromoChar[(data_ >> 12) & 0x3u];
    }
    return out;
}

}  // namespace eclipse
