# Advent of Code 2025

This year, I may try to do each day in a different programming model targeting GPU execution:

C++:
- [ ] CUDA
- [ ] OpenMP
- [ ] OpenACC
- [ ] C++ Standard Parallelism
- [ ] [C++ Executors](https://github.com/NVIDIA/stdexec/)
- [ ] [CCCL](https://github.com/NVIDIA/cccl/)

Python:
- [ ] [cuda.cccl](https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl)
- [ ] [cudf](https://github.com/rapids-ai/cudf)
- [x] [cupy](https://github.com/cupy/cupy/)
- [ ] [numba-cuda](https://github.com/NVIDIA/numba-cuda)
- [x] [warp](https://github.com/NVIDIA/warp)
- [x] [CuTe](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL)

## Solutions

| Day | Programming Model            | Run Command                      |
|:----|:-----------------------------|:---------------------------------|
| 01  | [CuTe](./day01.py)           | `uv run python day01.py`         |
| 02  | [cupy](./day02.py)           | `uv run python day02.py`         |
| 03  | [warp](./day03.py)           | `uv run python day03.py`         |
| 04  | [OpenACC](./day04.cpp)       | `./build/day04` (after building) |

## Dependencies

- `uv` with Python >= 3.13 installed
- CUDA >= 13.0
- NVIDIA HPC SDK >= 2025.09
- NVIDIA GPU with driver supporting CUDA 13
- CMake >= 4.0

## Building

I assume that CXX is set to `nvc++` already :)

```
cmake -B build && cmake --build build
```
