cmake_minimum_required(VERSION 3.20)

project(eclipse
    VERSION 0.0.1
    DESCRIPTION "Eclipse chess engine"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

option(ECLIPSE_NATIVE_ARCH "Build with -march=native (faster, host-specific)" ON)
option(ECLIPSE_SANITIZERS "Enable ASan + UBSan in Debug builds" OFF)
option(ECLIPSE_WERROR "Treat warnings as errors" OFF)
option(ECLIPSE_BUILD_TESTS "Build the unit-test suite" ON)
option(ECLIPSE_LTO "Enable link-time optimization for optimized builds" ON)

# Link-time (interprocedural) optimization. Lets the compiler inline the hot
# NNUE / movegen / PUCT helpers across translation units. Eval is integer SIMD
# and no -ffast-math is used, so results stay bit-identical — pure speed, no
# behavioural change. Enabled only for optimized configs and only when the
# toolchain actually supports it; Debug stays LTO-free for fast iteration.
if(ECLIPSE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _eclipse_ipo_ok OUTPUT _eclipse_ipo_msg)
    if(_eclipse_ipo_ok)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
        message(STATUS "LTO enabled for optimized builds")
    else()
        message(STATUS "LTO requested but unsupported: ${_eclipse_ipo_msg}")
    endif()
endif()

add_library(eclipse_warnings INTERFACE)
target_compile_options(eclipse_warnings INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2)
if(ECLIPSE_WERROR)
    target_compile_options(eclipse_warnings INTERFACE -Werror)
endif()

add_library(eclipse_arch INTERFACE)
if(ECLIPSE_NATIVE_ARCH)
    target_compile_options(eclipse_arch INTERFACE -march=native)
endif()
if(ECLIPSE_SANITIZERS)
    target_compile_options(eclipse_arch INTERFACE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined -fno-omit-frame-pointer>)
    target_link_options(eclipse_arch INTERFACE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
endif()

# Profile-guided optimization (two stages, driven by scripts/pgo_build.sh):
#   ECLIPSE_PGO=generate -> instrument the binary; running it emits .profraw
#   ECLIPSE_PGO=use      -> optimize using the merged profile at ECLIPSE_PGO_DATA
# Like LTO this only changes speed, not results (no -ffast-math; integer-SIMD
# eval). The `use` mode degrades gracefully to a normal build if the profile
# file is absent, so a stray `cmake --build` can never break on a missing
# profile. OFF (default) builds exactly as before.
set(ECLIPSE_PGO "OFF" CACHE STRING "PGO mode: OFF, generate, or use")
set_property(CACHE ECLIPSE_PGO PROPERTY STRINGS OFF generate use)
set(ECLIPSE_PGO_DATA "${CMAKE_BINARY_DIR}/eclipse.profdata"
    CACHE FILEPATH "Merged PGO profile data consumed by ECLIPSE_PGO=use")
if(ECLIPSE_PGO STREQUAL "generate")
    set(_pgo_raw_dir "${CMAKE_BINARY_DIR}/pgo-raw")
    target_compile_options(eclipse_arch INTERFACE -fprofile-generate=${_pgo_raw_dir})
    target_link_options(eclipse_arch INTERFACE -fprofile-generate=${_pgo_raw_dir})
    message(STATUS "PGO: instrumented build (raw profiles -> ${_pgo_raw_dir})")
elseif(ECLIPSE_PGO STREQUAL "use")
    if(EXISTS "${ECLIPSE_PGO_DATA}")
        target_compile_options(eclipse_arch INTERFACE
            -fprofile-use=${ECLIPSE_PGO_DATA}
            -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date)
        target_link_options(eclipse_arch INTERFACE -fprofile-use=${ECLIPSE_PGO_DATA})
        message(STATUS "PGO: optimized build using ${ECLIPSE_PGO_DATA}")
    else()
        message(WARNING "PGO use requested but ${ECLIPSE_PGO_DATA} is missing -- "
                        "building WITHOUT PGO. Run scripts/pgo_build.sh first.")
    endif()
endif()

# ONNX Runtime - locate via Homebrew so this works on both Intel
# (/usr/local) and Apple Silicon (/opt/homebrew) Macs without hardcoding.
# Honor a caller-provided ONNXRUNTIME_ROOT first for cross-builds.
if(NOT ONNXRUNTIME_ROOT)
    execute_process(
        COMMAND brew --prefix onnxruntime
        OUTPUT_VARIABLE ONNXRUNTIME_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _eclipse_onnx_result)
    if(NOT _eclipse_onnx_result EQUAL 0 OR NOT IS_DIRECTORY "${ONNXRUNTIME_ROOT}")
        message(FATAL_ERROR
            "Could not locate onnxruntime. Install with `brew install onnxruntime` "
            "or pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime.")
    endif()
endif()
include_directories(${ONNXRUNTIME_ROOT}/include/onnxruntime)
link_directories(${ONNXRUNTIME_ROOT}/lib)

add_subdirectory(src)

if(ECLIPSE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
