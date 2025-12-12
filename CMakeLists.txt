cmake_minimum_required(VERSION 4.0)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_COMPILER_FORCED On)
set(CMAKE_CUDA_ARCHITECTURES all-major)

project(AdventOfCode2025 CXX CUDA)

add_executable(day04 day04.cpp)
target_compile_options(day04 PRIVATE -acc -gpu:mem:separate)
target_link_options(day04 PRIVATE -acc -gpu:mem:separate)

add_executable(day07 day07.cpp)
target_compile_options(day07 PRIVATE -stdpar=gpu -gpu:mem:separate)
target_link_options(day07 PRIVATE -stdpar=gpu -gpu:mem:separate)

add_executable(day08 day08.cpp)
set_source_files_properties(day08.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(day08 PROPERTIES CUDA_SEPARABLE_COMPILATION On)
target_compile_options(day08 PRIVATE --extended-lambda)
target_link_options(day08 PRIVATE -L/opt/cuda/lib64)

add_executable(day09 day09.cpp)
target_compile_options(day09 PRIVATE -mp=gpu -gpu:mem:separate)
target_link_options(day09 PRIVATE -mp=gpu -gpu:mem:separate)

add_executable(day10 day10.cpp)
set_source_files_properties(day10.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(day10 PROPERTIES CUDA_SEPARABLE_COMPILATION On)
target_compile_options(day10 PRIVATE --extended-lambda -g)
target_link_options(day10 PRIVATE -L/opt/cuda/lib64)

add_executable(day11 day11.cpp)
target_compile_options(day11 PRIVATE --experimental-stdpar -stdpar=gpu -gpu:mem:separate)
target_link_options(day11 PRIVATE --experimental-stdpar -stdpar=gpu -gpu:mem:separate)

add_executable(day12 day12.cpp)
target_compile_options(day12 PRIVATE -stdpar=gpu -acc -gpu:mem:separate)
target_link_options(day12 PRIVATE -stdpar=gpu -acc -gpu:mem:separate)
