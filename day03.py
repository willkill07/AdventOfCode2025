import numpy as np
import nvtx
import warp as wp

wp.init()


@wp.kernel
def mark(a: wp.array(dtype=wp.int8), o: wp.array(dtype=wp.int8), W: int, K: int):
    r = wp.tid()
    W = wp.int32(W)
    K = wp.int32(K)
    s = r * W
    b = r * K
    d = wp.int32(W - K)
    k = wp.int32(0)
    for i in range(W):
        v = a[s + i]
        while d > 0 and k > 0:
            if o[b + k - 1] >= v:
                break
            k -= 1
            d -= 1
        if k < K:
            o[b + k] = v
            k += 1
        else:
            d -= 1


def run(d: wp.array(dtype=wp.int8), N: int, W: int, K: int) -> int:
    o = wp.zeros(N * K, dtype=wp.int8)
    wp.launch(
        mark,
        dim=N,
        inputs=[d, o, W, K],
    )
    return (
        o.numpy().reshape(N, K).astype(np.int64)
        @ (10 ** np.arange(K - 1, -1, -1, dtype=np.int64))
    ).sum()


@nvtx.annotate("Input Parsing")
def parse() -> tuple[wp.array, int, int]:
    with open("inputs/day03.in", "rb") as f:
        d = f.read()
    W = d.index(b"\n")
    d = d.replace(b"\n", b"")
    return (
        wp.array((np.frombuffer(d, dtype=np.uint8) - 48).astype(np.int8)),
        len(d) // W,
        W,
    )


@nvtx.annotate("Part 1")
def part1(d: wp.array(dtype=wp.int8), N: int, W: int) -> int:
    return run(d, N, W, 2)


@nvtx.annotate("Part 2")
def part2(d: wp.array(dtype=wp.int8), N: int, W: int) -> int:
    return run(d, N, W, 12)


@nvtx.annotate("Day 03")
def main() -> tuple[int, int]:
    d = parse()
    return part1(*d), part2(*d)


if __name__ == "__main__":
    print(*main())
