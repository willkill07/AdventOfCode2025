import cupy as cp
import cutlass
import cutlass.cute as cute
import numpy as np
import nvtx
from cutlass.cute.runtime import from_dlpack


@nvtx.annotate("Input Parsing")
def read_input() -> cute.Tensor:
    with open("inputs/day01.in") as f:
        lines = np.array([line.strip() for line in f])
        signs = np.where(np.char.startswith(lines, "L"), -1, 1)
        counts = np.array([int(s[1:]) for s in lines], dtype=np.int32)
        joined = cp.array(signs * counts)
        return from_dlpack(joined)


@cute.kernel
def compute_kernel(data: cute.Tensor, out: cute.Tensor, count_all: cutlass.Constexpr):  # noqa: C901
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        count = cute.Int32(0)
        start = cute.Int32(50)
        for row in cutlass.range(cute.size(data), unroll=32):
            elem = cute.Int32(data[row])
            prev = start

            # clockwise rotation correction
            while elem > 100:
                elem -= 100
                if cutlass.const_expr(count_all):
                    count += 1

            # counter-clockwise rotation correction
            while elem < -100:
                elem += 100
                if cutlass.const_expr(count_all):
                    count += 1

            # perform the movement and bound to [0, 99]
            start += elem
            if start >= 100:
                start -= 100
            if start < 0:
                start += 100

            if cutlass.const_expr(count_all):
                # add IFF we crossed or at zero
                if prev != 0 and (start == 0 or (elem > 0) == (start < prev)):
                    count += 1
            else:
                # add IFF we are at zero
                if start == 0:
                    count += 1
        out[0] = count


@nvtx.annotate("Part 1")
@cute.jit
def part1(data: cute.Tensor) -> int:
    out = cp.array([0], dtype=cp.int32)
    output = cute.make_tensor(cute.make_ptr(cute.Int32, out.data.ptr), (1,))
    compute_kernel(data, output, False).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )
    return out[0]


@nvtx.annotate("Part 2")
@cute.jit
def part2(data: cute.Tensor) -> int:
    out = cp.array([0], dtype=cp.int32)
    output = cute.make_tensor(cute.make_ptr(cute.Int32, out.data.ptr), (1,))
    compute_kernel(data, output, True).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )
    return out[0]


@nvtx.annotate("Day 01")
def main() -> tuple[int, int]:
    data = read_input()
    res1 = part1(data)
    res2 = part2(data)
    return res1, res2


if __name__ == "__main__":
    print(*main())
