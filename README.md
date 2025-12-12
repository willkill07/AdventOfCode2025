# Advent of Code 2025

This year, I did each day in a different programming model targeting GPU execution!

## Solutions

| Day | Programming Model | Run Command |
|:----|:------------------|:------------|
| [01](./day01.py)  | [CuTe](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL) | `uv run python day01.py` |
| [02](./day02.py)  | [cupy](https://github.com/cupy/cupy/) | `uv run python day02.py` |
| [03](./day03.py)  | [warp](https://github.com/NVIDIA/warp) | `uv run python day03.py` |
| [04](./day04.py)  | [cudf](https://github.com/rapids-ai/cudf) | `uv run --with "cudf-cu13=25.10.*" python day04.py` |
| [05](./day05.py)  | [cuda.cccl](https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl) | `uv run python day05.py` |
| [06](./day06.py)  | [numba-cuda](https://github.com/NVIDIA/numba-cuda) | `uv run python day06.py` |
| [07](./day07.cpp)  | C++ Standard Parallelism | `./build/day07` (after building) |
| [08](./day08.cpp)  | [CUDA with CCCL](https://github.com/NVIDIA/cccl/) | `./build/day08` (after building) |
| [09](./day09.cpp)  | OpenMP | `./build/day09` (after building) |
| [10](./day10.cpp)  | CUDA | `./build/day10` (after building) |
| [11](./day11.cpp)  | [C++ `std::execution`](https://github.com/NVIDIA/stdexec/) | `./build/day11` (after building) |
| [12](./day12.cpp)  | OpenACC | `./build/day12` (after building) |

I also did Day 04 in OpenACC and Day 12 in cudf.

## Dependencies

- `uv` with Python >= 3.13 installed
- CUDA >= 13.0
- NVIDIA HPC SDK >= 2025.09
- NVIDIA GPU with driver supporting CUDA 13
- CMake >= 4.0

## Building

Run/modify `./build_all.sh` as you see fit:

```bash
./build_all.sh
```

## Running

Assuming you have all prerequisites in place and you already compiled the C++ solutions, simply invoke:

```bash
./run_all.sh
```
