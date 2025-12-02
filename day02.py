import cupy as cp
import numpy as np

# Parsing
parts = np.loadtxt("inputs/day02.in", dtype=str, delimiter=",")
split = np.char.partition(parts, "-")
ls, rs = split[:, 0], split[:, 2]
lo, hi = ls.astype(cp.int64), rs.astype(cp.int64)
total_ids = (hi - lo + 1).sum()
ids = cp.empty(total_ids, cp.int64)
off = 0
for a, b in zip(lo, hi):
    count = b - a + 1
    ids[off : off + count] = cp.arange(a, b + 1)
    off += count
digits = cp.floor(cp.log10(ids)).astype(cp.int64) + 1

# Part 1
half = digits // 2
b = cp.power(10, half)
ls, rs = (ids // b), (ids % b)
invalid_mask = ((digits & 1) == 0) & (ls == rs)
r1 = int(cp.sum(ids[invalid_mask]))

# Part 2
max_digits = int(digits.max())
invalid = cp.zeros_like(ids, cp.bool_)
for chunk_width in range(1, max_digits // 2 + 1):
    mask = digits % chunk_width == 0
    if bool(mask.any()):
        valid_ids = cp.nonzero(mask)[0]
        curr_ids = ids[valid_ids]
        width = digits[valid_ids]
        counts = width // chunk_width
        min_counts = counts >= 2
        if bool(min_counts.any()):
            b = cp.int64(10**chunk_width)
            curr = curr_ids[min_counts]
            curr_counts = counts[min_counts]
            curr_chunk = curr % b
            max_counts = int(curr_counts.max())
            dup_chunk = curr_chunk.astype(cp.int64)
            for i in range(1, max_counts):
                dup_chunk = cp.where(
                    curr_counts > i, dup_chunk * b + curr_chunk, dup_chunk
                )
            invalid[valid_ids[min_counts]] |= curr == dup_chunk
r2 = int(ids[invalid].sum())

print(r1, r2)
