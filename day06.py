from dataclasses import dataclass

import numpy as np
import nvtx
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray


@dataclass
class Data:
    n: int
    numbers: DeviceNDArray
    ops: DeviceNDArray
    offsets: DeviceNDArray
    outputs: DeviceNDArray


@nvtx.annotate("Parse Input")
def parse() -> Data:
    with open("inputs/day06.in") as f:
        data = f.read()
    W = 1 + data.index("\n")
    buf = np.frombuffer(data.encode(), dtype=np.uint8)
    H = buf.size // W
    buf = buf.reshape((H, W))[:]
    numbers = buf[:-1]
    ops = buf[-1][:-1]
    mask = np.all(np.logical_or(numbers == ord(" "), numbers == ord("\n")), axis=0)
    ind = np.where(mask)[0]
    prefix = np.zeros(len(ind) + 1, dtype=np.int32)
    prefix[1:] = ind + 1
    n = prefix.size - 1
    return Data(
        n,
        cuda.to_device(numbers),
        cuda.to_device(ops),
        cuda.to_device(prefix),
        cuda.device_array(n, dtype=np.int64),
    )


@cuda.jit
def kernel(
    n: int,
    nums: DeviceNDArray,
    ops: DeviceNDArray,
    offsets: DeviceNDArray,
    out: DeviceNDArray,
    by_col: bool,
):
    pid = cuda.grid(1)
    if pid >= n:
        return

    start = offsets[pid]
    stop = offsets[pid + 1]

    op = ops[start]

    operands = nums.shape[0]
    total = 0 if op == ord("+") else 1

    r1 = range(operands)
    r2 = range(start, stop) if not by_col else range(stop - 1, start - 1, -1)

    outer = r2 if by_col else r1
    inner = r1 if by_col else r2

    for r in outer:
        num = 0
        for c in inner:
            v = nums[c, r] if by_col else nums[r, c]
            if ord("0") < v and v <= ord("9"):
                num = num * 10 + (v - ord("0"))
        if num != 0:
            if op == ord("+"):
                total += num
            else:
                total *= num

    out[pid] = total


@nvtx.annotate("Part 1")
def part1(data: Data) -> int:
    THREADS_PER_BLOCK = 128
    blocks = (data.n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    kernel[blocks, THREADS_PER_BLOCK](
        data.n, data.numbers, data.ops, data.offsets, data.outputs, False
    )
    return data.outputs.copy_to_host().sum()


@nvtx.annotate("Part 2")
def part2(data: Data) -> int:
    THREADS_PER_BLOCK = 128
    blocks = (data.n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    kernel[blocks, THREADS_PER_BLOCK](
        data.n, data.numbers, data.ops, data.offsets, data.outputs, True
    )
    return data.outputs.copy_to_host().sum()


@nvtx.annotate("Day 06")
def main():
    data = parse()
    r1 = part1(data)
    r2 = part2(data)
    return r1, r2


if __name__ == "__main__":
    print(*main())
