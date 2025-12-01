import cupy as cp
import cutlass
import cutlass.cute as cute
import numpy as np
import nvtx
from cutlass.cute.runtime import from_dlpack


@nvtx.annotate("Input Parsing")
def read_input() -> tuple[cute.Tensor, int]:
    with open("inputs/day01.in") as f:
        lines = np.array([line.strip() for line in f])
        signs = np.where(np.char.startswith(lines, "L"), -1, 1)
        counts = np.array([int(s[1:]) for s in lines], dtype=np.int32)
        joined = cp.array(signs * counts)
        return from_dlpack(joined), len(lines)


@cute.kernel
def compute_kernel(data: cute.Tensor, size: cute.Int32, count_all: cutlass.Constexpr):  # noqa: C901
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        count = cute.Int32(0)
        start = cute.Int32(50)
        row = cute.Int32(0)
        while row < size:
            elem = cute.Int32(data[row])
            d = 1 if elem > 0 else -1
            if count_all:
                while elem != 0:
                    elem -= d
                    start += d
                    if start < 0:
                        start += 100
                    if start >= 100:
                        start -= 100
                    if start == 0:
                        count += 1
            else:
                start += elem
                while start < 0:
                    start += 100
                while start >= 100:
                    start -= 100
                if start == 0:
                    count += 1
            row += 1
        cute.printf("{}", count)


@nvtx.annotate("Part 1")
@cute.jit
def part1(data: cute.Tensor, size: cute.Int32) -> None:
    compute_kernel(data, size, False).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


@nvtx.annotate("Part 2")
@cute.jit
def part2(data: cute.Tensor, size: cute.Int32) -> None:
    compute_kernel(data, size, True).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


@nvtx.annotate("Day 01")
def main() -> None:
    parsed_input = read_input()
    part1(*parsed_input)
    part2(*parsed_input)


if __name__ == "__main__":
    main()
