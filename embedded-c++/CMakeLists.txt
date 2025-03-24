cmake_minimum_required(VERSION 2.8.12...3.29)
project(example-c++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3)
# add_compile_options(-g) # for debug -- SLOW

include_directories(
    ../duckdb/src/include 
    include 
    ../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}  
    ${CMAKE_CURRENT_SOURCE_DIR}/usearch/helpers
)

# link_directories(../build/debug/src) # for debug
link_directories(../build/release/src)

# List all implementation files explicitly
set(UTILS_SOURCES
    usearch/helpers/index_operations.cpp
    usearch/helpers/database_setup.cpp
    usearch/helpers/file_operations.cpp
    usearch/helpers/query_runner.cpp
    # Add any missing implementation files here
)

# Shared helper classes - built as STATIC library
add_library(usearch_utils STATIC ${UTILS_SOURCES})

target_link_libraries(usearch_utils duckdb)

add_executable(usearch_random usearch/random.cpp)
add_executable(usearch_full_coverage usearch/full_coverage.cpp)
add_executable(usearch_new_data usearch/new_data.cpp)

target_link_libraries(usearch_random usearch_utils duckdb)
target_link_libraries(usearch_full_coverage usearch_utils duckdb)
target_link_libraries(usearch_new_data usearch_utils duckdb)

# Show verbose output during linking if there are issues
set(CMAKE_VERBOSE_MAKEFILE ON)
