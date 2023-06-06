cmake_minimum_required(VERSION 3.13)
project(CeMl CXX)
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(HEADER_FILES "include/ceml/constants.hpp" "include/ceml/tensor.hpp" "include/ceml/operations.hpp")
add_library(cemllib ${HEADER_FILES})
target_include_directories(cemllib PUBLIC include)
set_target_properties(cemllib PROPERTIES LINKER_LANGUAGE CXX)


add_executable(ceml main.cpp)
target_link_libraries(ceml cemllib)