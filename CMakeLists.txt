cmake_minimum_required(VERSION 3.20)

project(eclipse
    VERSION 0.0.1
    DESCRIPTION "Eclipse chess engine"
    LANGUAGES CXX)

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

# ONNX Runtime
set(ONNXRUNTIME_ROOT "/opt/homebrew/Cellar/onnxruntime/1.26.0")
include_directories(${ONNXRUNTIME_ROOT}/include/onnxruntime)
link_directories(${ONNXRUNTIME_ROOT}/lib)

add_subdirectory(src)

if(ECLIPSE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
