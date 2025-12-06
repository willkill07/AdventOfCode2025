from dataclasses import dataclass

import cupy as cp
import numpy as np
import nvtx
from cuda.compute import (
    DoubleBuffer,
    OpKind,
    SortOrder,
    TransformIterator,
    ZipIterator,
    inclusive_scan,
    radix_sort,
    reduce_into,
)
from numpy.typing import NDArray


@dataclass
class InputData:
    lo: NDArray
    hi: NDArray
    val: NDArray


@nvtx.annotate("Input Parsing")
def read_input() -> InputData:
    """
    Input structure (two blocks separated by a blank line):

        RANGES BLOCK               VALUES BLOCK
        ------------               ------------
        3-5                        1
        10-14                      5
        16-20                      8
        12-18                      11
                                   17
                                   32

    Parsing turns the left block into pairs (L, R), and we sort L and R
    separately on the GPU.

        Original ranges:
            3-5, 10-14, 16-20, 12-18

        After sorting:

            lo (range starts):   [ 3, 10, 12, 16 ]
            hi (range ends):     [ 5, 14, 18, 20 ]
            val (candidate IDs): [ 1, 5, 8, 11, 17, 32 ]
    """
    with open("inputs/day05.in") as f:
        all_data = f.read()
    r_block, v_block = [x.strip() for x in all_data.split("\n\n", maxsplit=1)]
    normalized = r_block.replace("\n", "-")
    data = np.fromstring(normalized, sep="-", dtype=np.int64).reshape(-1, 2)
    lo = cp.asarray(data[:, 0])
    hi = cp.asarray(data[:, 1])
    val = cp.fromstring(v_block, sep="\n", dtype=np.int64)
    b_lo = DoubleBuffer(lo, cp.empty_like(lo))
    b_hi = DoubleBuffer(hi, cp.empty_like(hi))
    b_val = DoubleBuffer(val, cp.empty_like(val))
    radix_sort(b_lo, None, None, None, SortOrder.ASCENDING, lo.size)
    radix_sort(b_hi, None, None, None, SortOrder.ASCENDING, hi.size)
    radix_sort(b_val, None, None, None, SortOrder.ASCENDING, val.size)
    return InputData(b_lo.current(), b_hi.current(), b_val.current())


@nvtx.annotate("Part 1")
def part1(data: InputData) -> int:
    """
    Goal
    ----
    Count how many IDs in `val` fall inside at least one range [lo, hi].

    Background trick
    ----------------
    If arrays `lo` and `hi` are sorted:

        starts(v) = number of ranges whose L ≤ v
        ends(v)   = number of ranges whose R < v
        active(v) = starts(v) - ends(v)

    An ID v is fresh if active(v) > 0.

    Example
    -------
    Ranges:
        [3,5], [10,14], [16,20], [12,18]

    Values:
        1   5   8   11   17   32

    Sorted arrays:

        lo:  [ 3, 10, 12, 16 ]
        hi:  [ 5, 14, 18, 20 ]
        val: [ 1, 5, 8, 11, 17, 32 ]

    We compute:

        first = searchsorted(lo, val, side="right")
        (how many lo ≤ v)

        last  = searchsorted(hi, val, side="left")
        (how many hi < v)

    Pretty picture:

            v:          1    5    8   11   17   32
            --------------------------------------
            first:      0    1    1    2    4    4
            last:       0    0    1    1    1    4
            --------------------------------------
            diff:       0    1    0    1    3    0
            --------------------------------------
            valid:      0    1    0    1    1    0

    Fresh mask = (diff > 0):
            [0, 1, 0, 1, 1, 0] → total = 3
    """
    out = cp.empty(1, dtype=cp.int32)
    init = np.array([0], dtype=np.int32)
    first = cp.searchsorted(data.lo, data.val, side="right")
    last = cp.searchsorted(data.hi, data.val, side="left")

    def valid(pair):
        return int((pair[0] - pair[1]) > 0)

    reduce_into(
        TransformIterator(ZipIterator(first, last), valid),
        out,
        OpKind.PLUS,
        first.size,
        init,
    )

    return int(out[0].item())


@nvtx.annotate("Part 2")
def part2(data: InputData) -> int:
    """
    Goal
    ----
    Compute the total size of the union of all ranges (how many distinct
    IDs are inside any range).

    Sweep-line method
    -----------------
    Convert inclusive [L, R] ranges into half-open [L, R+1):

        [L, R] → [L, R+1)

    Emit “events”:
        +1 at L
        -1 at R+1

    After sorting all event positions, an inclusive scan of deltas gives the
    number of active intervals over each segment.

    Example
    -------
    Original ranges:
        [3,5], [10,14], [16,20], [12,18]

    Half-open:
        [3,6), [10,15), [16,21), [12,19)

    Events (pos → delta):

        3 : +1
        6 : -1
        10 : +1
        15 : -1
        16 : +1
        21 : -1
        12 : +1
        19 : -1

    Sorted:

        pos:      3    6   10   12   15   16   19   21
        delta:   +1   -1   +1   +1   -1   +1   -1   -1
        active:   1    0    1    2    1    2    1    0
                  ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
             number of simultaneously-active intervals

    Each segment spans consecutive positions:

        segment      width   active   contribution
        ------------------------------------------
        [3,6)          3       1            3
        [6,10)         4       0            0
        [10,12)        2       1            2
        [12,15)        3       2            6
        [15,16)        1       1            1
        [16,19)        3       2            6
        [19,21)        2       1            2
                                     -------------
                                        total = 14

    The GPU code constructs:
        - positions p = [L..., R+1...]
        - deltas    d = [+1..., -1...]
        - radix_sort(p, d)
        - active = inclusive_scan(d)
        - sum(active[i] * (pos[i+1] - pos[i]))
    """
    lo = data.lo
    hi = data.hi
    n = lo.size

    p = cp.empty(2 * n, dtype=lo.dtype)
    d = cp.empty_like(p)
    active = cp.empty_like(p)
    out = cp.empty(1, dtype=cp.int64)
    h_init = np.array([0], dtype=np.int64)
    h_zero = np.array([0], dtype=np.int64)

    p[:n] = lo
    d[:n] = 1
    p[n:] = hi + 1
    d[n:] = -1

    bpos = DoubleBuffer(p, cp.empty_like(p))
    bdelta = DoubleBuffer(d, cp.empty_like(d))
    radix_sort(bpos, None, bdelta, None, SortOrder.ASCENDING, p.size)
    inclusive_scan(bdelta.current(), active, OpKind.PLUS, h_init, d.size)
    cpos = bpos.current()

    def compute(tup):
        return (tup[0] - tup[1]) * (tup[2] > 0)

    reduce_into(
        TransformIterator(ZipIterator(cpos[1:], cpos[:-1], active[:-1]), compute),
        out,
        OpKind.PLUS,
        2 * n - 1,
        h_zero,
    )

    return int(out[0].item())


@nvtx.annotate("Day 05")
def main() -> tuple[int, int]:
    data = read_input()
    res1 = part1(data)
    res2 = part2(data)
    return res1, res2


if __name__ == "__main__":
    print(*main())
