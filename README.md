# Advent of Code 2025

This year, I may try to do each day in a different programming model targeting GPU execution:

C++:
- [ ] CUDA
- [ ] OpenMP
- [x] OpenACC
- [x] C++ Standard Parallelism
- [ ] [C++ Executors](https://github.com/NVIDIA/stdexec/)
- [x] [CCCL](https://github.com/NVIDIA/cccl/)

Python:
- [x] [cuda.cccl](https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl)
- [x] [cudf](https://github.com/rapids-ai/cudf)
- [x] [cupy](https://github.com/cupy/cupy/)
- [x] [numba-cuda](https://github.com/NVIDIA/numba-cuda)
- [x] [warp](https://github.com/NVIDIA/warp)
- [x] [CuTe](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL)

## Solutions

| Day | Programming Model | Run Command |
|:----|:------------------|:------------|
| 01  | [CuTe](./day01.py) | `uv run python day01.py` |
| 02  | [cupy](./day02.py) | `uv run python day02.py` |
| 03  | [warp](./day03.py) | `uv run python day03.py` |
| 04  | [cudf](./day04.py), [OpenACC](./day04.cpp) | `uv run --with "cudf-cu13=25.10.*" python day04.py`, `./build/day04` (after building) |
| 05  | [cuda.cccl](./day05.py) | `uv run python day05.py` |
| 06  | [numba-cuda](./day06.py) | `uv run python day06.py` |
| 07  | [C++ Standard Parallelism](./day07.cpp) | `./build/day07` (after building) |
| 08  | [CUDA with CCCL](./day08.cpp) | `./build/day08` (after building) |

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
