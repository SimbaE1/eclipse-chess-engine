#include "bitboard.hpp"

#include <sstream>

namespace eclipse {

std::string bb_to_string(Bitboard b) {
    std::ostringstream oss;
    for (int r = 7; r >= 0; --r) {
        oss << (r + 1) << "  ";
        for (int f = 0; f < 8; ++f) {
            Square s = make_square(File(f), Rank(r));
            oss << ((b & square_bb(s)) ? "X " : ". ");
        }
        oss << '\n';
    }
    oss << "   a b c d e f g h\n";
    return oss.str();
}

}  // namespace eclipse
