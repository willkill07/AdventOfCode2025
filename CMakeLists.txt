cmake_minimum_required(VERSION 4.0)

set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)

project(AdventOfCode2025 CXX CUDA)

add_executable(day04 day04.cpp)
target_compile_options(day04 PRIVATE -std=c++23 -acc -O3)
target_link_options(day04 PRIVATE -std=c++23 -acc)

add_executable(day07 day07.cpp)
target_compile_options(day07 PRIVATE -std=c++23 -stdpar=gpu -O3)
target_link_options(day07 PRIVATE -std=c++23 -stdpar=gpu)

add_executable(day08 day08.cpp)
set_source_files_properties(day08.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(day08 PRIVATE /opt/nvidia/hpc_sdk/Linux_x86_64/2025/compilers/include-stdpar)
target_compile_options(day08 PRIVATE -std=c++20 -O3 --extended-lambda -arch=native)
target_link_options(day08 PRIVATE -std=c++20)
