#pragma once

#include <cstdio>

namespace eclipse::test {

inline int& failure_count() {
    static int n = 0;
    return n;
}

inline void report_failure(const char* file, int line, const char* expr) {
    std::fprintf(stderr, "FAIL  %s:%d  %s\n", file, line, expr);
    ++failure_count();
}

inline int summarize(const char* suite) {
    if (failure_count() == 0) {
        std::printf("PASS  %s\n", suite);
        return 0;
    }
    std::printf("FAIL  %s (%d failures)\n", suite, failure_count());
    return 1;
}

}  // namespace eclipse::test

#define ECLIPSE_CHECK(expr) \
    do { if (!(expr)) ::eclipse::test::report_failure(__FILE__, __LINE__, #expr); } while (0)
