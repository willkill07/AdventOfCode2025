import cudf
import cupy as cp
import numpy as np
import nvtx


@nvtx.annotate("Parse Input")
def parse() -> tuple[cudf.DataFrame, int]:
    with open("inputs/day04.in") as f:
        raw = [line.strip() for line in f]
    grid = np.array(
        [[1 if c == "@" else 0 for c in row] for row in raw], dtype=np.uint8
    )
    padded = np.pad(grid, pad_width=1, mode="constant", constant_values=0)
    _, W = padded.shape
    flat = padded.ravel()
    df = cudf.DataFrame({"v": cp.asarray(flat), "n": cp.zeros_like(flat)})
    return df, W


@nvtx.annotate("Step")
def step(df, W) -> int:
    v = df["v"]
    n = v.shift(-W - 1, fill_value=0)
    n += v.shift(-W + 0, fill_value=0)
    n += v.shift(-W + 1, fill_value=0)
    n += v.shift(0 - 1, fill_value=0)
    n += v.shift(0 + 1, fill_value=0)
    n += v.shift(W - 1, fill_value=0)
    n += v.shift(W + 0, fill_value=0)
    n += v.shift(W + 1, fill_value=0)
    mask = (v == 1) & (n < 4)
    df["v"] = v - mask
    return mask.sum()


@nvtx.annotate("Part 1")
def part1(df: cudf.DataFrame, W: int) -> int:
    return step(df, W)


@nvtx.annotate("Part 2")
def part2(df: cudf.DataFrame, W: int, start: int) -> int:
    total = start
    while True:
        r = step(df, W)
        if r == 0:
            break
        total += r
    return total


@nvtx.annotate("Day 04")
def main():
    df, W = parse()
    p1 = part1(df, W)
    p2 = part2(df, W, p1)
    print(p1, p2)


if __name__ == "__main__":
    main()
