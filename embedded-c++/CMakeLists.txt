cmake_minimum_required(VERSION 2.8.12...3.29)
project(example-c++)

set(CMAKE_CXX_STANDARD 11)

add_compile_options(-O3)
# add_compile_options(-g) # for debug -- SLOW

include_directories(../duckdb/src/include include)
# link_directories(../build/debug/src) # for debug
link_directories(../build/release/src)

set(BENCHMARK_LIB_PATH "${CMAKE_SOURCE_DIR}/include/benchmark")
find_library(BENCHMARK_LIB NAMES benchmark PATHS ${BENCHMARK_LIB_PATH})
if(NOT BENCHMARK_LIB)
    message(FATAL_ERROR "Benchmark library not found.")
else()
    message(STATUS "Benchmark library found: ${BENCHMARK_LIB}")
endif()

add_executable(benchmark_deletions benchmark_deletions.cpp)
add_executable(recall recall.cpp)

# Linking libraries
target_link_libraries(benchmark_deletions duckdb ${BENCHMARK_LIB})
target_link_libraries(recall duckdb ${BENCHMARK_LIB})
