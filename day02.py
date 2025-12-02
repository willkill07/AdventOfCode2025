import cupy as cp
import numpy as np
import nvtx
from numpy.typing import NDArray


@nvtx.annotate("Input Parsing")
def read_input() -> NDArray:
    with open("inputs/day02.in") as f:
        line = f.readline()
        parts = np.array(line.split(","), dtype=str)
        parts = parts[parts != ""]

        split = np.char.partition(parts, "-")
        left = split[:, 0]
        right = split[:, 2]

        lo = left.astype(np.int64)
        hi = right.astype(np.int64)

        lengths = hi - lo + 1
        total_ids = lengths.sum()

        # Build all IDs for all ranges in one vector
        ids = cp.empty(total_ids, dtype=cp.int64)

        # Fill the CuPy array range-by-range
        offset = 0
        for a, b in zip(lo, hi):
            count = b - a + 1
            ids[offset : offset + count] = cp.arange(a, b + 1, dtype=cp.int64)
            offset += count

        return ids


@nvtx.annotate("Part 1")
def part1(data: NDArray) -> int:
    digits = cp.floor(cp.log10(data)).astype(cp.int64) + 1
    even_mask = digits % 2 == 0

    half = digits // 2
    base = cp.power(10, half)

    left = data // base
    right = data % base

    invalid_mask = even_mask & (left == right)

    return int(cp.sum(data[invalid_mask]).get())


@nvtx.annotate("Part 2")
def part2(ids: NDArray) -> int:
    digits = cp.floor(cp.log10(ids)).astype(cp.int32) + 1
    max_digits = int(digits.max().get())

    invalid = cp.zeros_like(ids, dtype=cp.bool_)
    for chunk_width in range(1, max_digits // 2 + 1):
        with nvtx.annotate(f"Chunk Width {chunk_width}"):
            mask = digits % chunk_width == 0
            if not bool(mask.any().get()):
                continue

            valid_ids = cp.nonzero(mask)[0]
            selected_ids = ids[valid_ids]
            width = digits[valid_ids]

            reps = width // chunk_width
            min_reps = reps >= 2
            if not bool(min_reps.any().get()):
                break

            active = valid_ids[min_reps]
            current = selected_ids[min_reps]
            reps_rep = reps[min_reps]

            base = cp.int64(10**chunk_width)
            curr_chunk = current % base
            max_reps = int(reps_rep.max().get())
            repeated_chunk = curr_chunk.astype(cp.int64)

            for i in range(1, max_reps):
                has_more = reps_rep > i
                repeated_chunk = cp.where(
                    has_more, repeated_chunk * base + curr_chunk, repeated_chunk
                )

            invalid[active] |= current == repeated_chunk

    return int(ids[invalid].sum().get())


@nvtx.annotate("Day 02")
def main() -> tuple[int, int]:
    data = read_input()
    res1 = part1(data)
    res2 = part2(data)
    return res1, res2


if __name__ == "__main__":
    print(*main())
