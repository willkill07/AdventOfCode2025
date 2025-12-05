from dataclasses import dataclass

import cuda.compute
import cupy as cp
import numpy as np
import nvtx
from cuda.compute import OpKind, SortOrder
from numpy.typing import NDArray


@dataclass
class InputData:
    lo: NDArray
    hi: NDArray
    val: NDArray


@nvtx.annotate("Input Parsing")
def read_input() -> InputData:
    """
    Input format (conceptually):

        RANGES BLOCK                    VALUES BLOCK
        -------------                   -----------
        3-5                             1
        10-14                           5
        16-20                           8
        12-18                           11
                                        17
                                        32

    After parsing and sorting, we have three device arrays:

        lo  = sorted start points of ranges
        hi  = sorted end   points of ranges
        val = sorted available IDs

    Example:

        original ranges:   3-5, 10-14, 16-20, 12-18

        lo (starts):       [ 3, 10, 12, 16 ]
        hi (ends):         [ 5, 14, 18, 20 ]
        val (IDs):         [ 1, 5, 8, 11, 17, 32 ]
    """
    with open("inputs/day05.in") as f:
        all_data = f.read()
        r_block, v_block = [x.strip() for x in all_data.split("\n\n", maxsplit=1)]
        # Replace newlines and dashes with spaces, then parse
        normalized = r_block.replace("-", "\n")
        data = np.fromstring(normalized, sep="\n", dtype=np.int64).reshape(-1, 2)
        u_lo = cp.asarray(data[:, 0])
        u_hi = cp.asarray(data[:, 1])
        u_val = cp.fromstring(v_block, sep="\n", dtype=np.int64)
        lo = cp.empty_like(u_lo)
        hi = cp.empty_like(u_hi)
        val = cp.empty_like(u_val)
        cuda.compute.radix_sort(u_lo, lo, None, None, SortOrder.ASCENDING, lo.size)
        cuda.compute.radix_sort(u_hi, hi, None, None, SortOrder.ASCENDING, hi.size)
        cuda.compute.radix_sort(u_val, val, None, None, SortOrder.ASCENDING, len(val))
        return InputData(lo, hi, val)


def gt_zero(a):
    return 1 if a > 0 else 0


@nvtx.annotate("Part 1")
def part1(data: InputData) -> int:
    """
    Goal: Count how many available IDs are covered by at least one range.

    We use a sweep-line style trick via searchsorted.

    For each value v in val:

        starts(v) = number of ranges with lo <= v
        ends(v)   = number of ranges with hi <  v

        active(v) = starts(v) - ends(v)
                  = how many ranges cover v

        v is fresh if active(v) > 0

    ASCII picture for the example:

        ranges: [3,5], [10,14], [16,20], [12,18]
        values:  1   5   8   11   17   32

        lo:  [ 3, 10, 12, 16 ]
        hi:  [ 5, 14, 18, 20 ]
        val: [ 1, 5, 8, 11, 17, 32 ]

        first = searchsorted(lo, val, side="right")
          counts how many lo <= v

            v:        1  5  8  11  17  32
            first:    0  1  1   2   4   4

        last  = searchsorted(hi, val, side="left")
          counts how many hi < v

            v:        1  5  8  11  17  32
            last:     0  0  1   1   1   4

        diff = first - last:

            diff:     0  1  0   1   3   0
                      ^  ^      ^   ^
                      |  |      |   |
                      |  |      |   +-- covered
                      |  |      +------ covered
                      |  +------------- covered
                      +---------------- spoiled

        fresh_mask = (diff > 0): [0,1,0,1,1,0] → sum = 3
    """
    first = cp.searchsorted(data.lo, data.val, side="right")
    last = cp.searchsorted(data.hi, data.val, side="left")

    diff = cp.empty_like(first, dtype=cp.int64)
    mask = cp.empty_like(diff, cp.int32)
    out = cp.empty(1, dtype=cp.int32)
    init = np.array([0], dtype=np.int32)

    cuda.compute.binary_transform(first, last, diff, OpKind.MINUS, diff.size)
    cuda.compute.unary_transform(diff, mask, gt_zero, diff.size)
    cuda.compute.reduce_into(mask, out, OpKind.PLUS, mask.size, init)

    return int(out[0].item())


@nvtx.annotate("Part 2")
def part2(data: InputData) -> int:
    """
    Goal: Size of the union of all ranges (how many distinct IDs are fresh).

    Trick: Convert inclusive ranges [L,R] to half-open intervals [L,R+1)
    and sweep over "events":

        +1 at position L
        -1 at position R+1

    If we sort all event positions and prefix-sum the deltas, we get the
    active-interval count on each segment between consecutive positions.

    ASCII picture (example ranges: 3-5, 10-14, 16-20, 12-18):

        Original inclusive ranges:
            [3,5], [10,14], [16,20], [12,18]

        Convert to half-open [L,R+1):

            [3,6), [10,15), [16,21), [12,19)

        Events (position, delta):
            3:  +1    (start of [3,6)
            6:  -1    (end   of [3,6)
            10: +1
            15: -1
            16: +1
            21: -1
            12: +1
            19: -1

        After sorting positions:

            pos:   3    6   10   12   15   16   19   21
            dlt:  +1   -1   +1   +1   -1   +1   -1   -1

        Inclusive scan of deltas → active count at each event:

            active: 1    0    1    2    1    2    1    0

        Segments are between consecutive positions:

            segment        active   length  contribution
            [3,6)            1        3          3
            [6,10)           0        4          0
            [10,12)          1        2          2
            [12,15)          2        3          3
            [15,16)          1        1          1
            [16,19)          2        3          3
            [19,21)          1        2          2
                                             ----
                                              14  IDs fresh

        The code below does exactly this on the GPU:

            - p, d:     unsorted positions and deltas
            - pos, delta: sorted by position (radix_sort)
            - active:  inclusive_scan(delta)
            - seg_lengths: pos[k+1] - pos[k]
            - mask:   1 if active[k] > 0 else 0
            - contrib: seg_lengths * mask
            - answer: reduce(sum) of contrib
    """
    lo = data.lo
    hi = data.hi
    n = lo.size
    ns = 2 * n - 1

    p = cp.empty(2 * n, dtype=lo.dtype)
    d = cp.empty(2 * n, dtype=lo.dtype)
    pos = cp.empty_like(p)
    delta = cp.empty_like(d)
    active = cp.empty_like(delta)
    seg_lengths = cp.empty(ns, dtype=pos.dtype)
    mask = cp.empty(ns, dtype=pos.dtype)
    contrib = cp.empty_like(seg_lengths)
    out = cp.empty(1, dtype=contrib.dtype)
    h_init = np.array([0], dtype=delta.dtype)
    h_zero = np.array([0], dtype=contrib.dtype)

    p[:n] = lo
    d[:n] = 1
    p[n:] = hi + 1
    d[n:] = -1

    cuda.compute.radix_sort(p, pos, d, delta, SortOrder.ASCENDING, p.size)
    cuda.compute.inclusive_scan(delta, active, OpKind.PLUS, h_init, delta.size)
    cuda.compute.binary_transform(pos[1:], pos[:-1], seg_lengths, OpKind.MINUS, ns)
    cuda.compute.unary_transform(active[:-1], mask, gt_zero, ns)
    cuda.compute.binary_transform(seg_lengths, mask, contrib, OpKind.MULTIPLIES, ns)
    cuda.compute.reduce_into(contrib, out, OpKind.PLUS, ns, h_zero)

    return int(out[0].item())


@nvtx.annotate("Day 05")
def main() -> tuple[int, int]:
    data = read_input()
    res1 = part1(data)
    res2 = part2(data)
    return res1, res2


if __name__ == "__main__":
    print(*main())
