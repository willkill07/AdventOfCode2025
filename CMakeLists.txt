cmake_minimum_required(VERSION 4.0)
project(AdventOfCode2025 CXX)

add_executable(day04 day04.cpp)
target_compile_options(day04 PRIVATE -std=c++23 -acc -O3 -gpu=cc120)
target_link_options(day04 PRIVATE -std=c++23 -acc -gpu=cc120)
