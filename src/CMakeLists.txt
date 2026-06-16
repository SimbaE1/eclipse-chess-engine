# Fathom Syzygy tablebase library (C source, no C++ warnings).
add_library(fathom STATIC
    ${CMAKE_SOURCE_DIR}/extern/fathom/src/tbprobe.c
)
target_include_directories(fathom PUBLIC ${CMAKE_SOURCE_DIR}/extern/fathom/src)
target_compile_options(fathom PRIVATE -w)  # suppress external-library warnings

add_library(eclipse_core STATIC
    ab.cpp
    attacks.cpp
    bitboard.cpp
    eval.cpp
    mcts.cpp
    move.cpp
    movegen.cpp
    nnue.cpp
    perft.cpp
    policy.cpp
    position.cpp
    search.cpp
    see.cpp
    syzygy.cpp
    tt.cpp
    uci.cpp
    zobrist.cpp
)
target_include_directories(eclipse_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(eclipse_core
    PUBLIC  eclipse_arch
    PRIVATE eclipse_warnings onnxruntime fathom)

add_executable(eclipse main.cpp)
target_link_libraries(eclipse
    PRIVATE eclipse_core eclipse_warnings)
