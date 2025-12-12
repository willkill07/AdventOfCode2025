import cudf
import nvtx


@nvtx.annotate("Parse Input")
def parse() -> tuple[cudf.DataFrame, list[int]]:
    with open("inputs/day12.in", "r") as f:
        content = f.read()

    sections = content.strip().split("\n\n")
    shape_blocks = sections[:-1]
    region_text = sections[-1]

    shapes = [block.count("#") for block in shape_blocks]

    lines = cudf.Series(region_text.strip().split("\n"))
    parts = lines.str.split(": ", expand=True)
    dims = parts[0].str.split("x", expand=True).astype(int)
    counts = parts[1].str.split(" ")
    regions = cudf.DataFrame({"area": dims[0] * dims[1], "counts": counts.list.astype(int)})
    return regions, shapes


@nvtx.annotate("Part 1")
def part1(regions: cudf.DataFrame, shapes: list[int]) -> int:
    required = sum(regions["counts"].list.get(i) * weight for i, weight in enumerate(shapes))
    return int((required < regions["area"]).sum())


@nvtx.annotate("Day 12")
def main():
    print(part1(*parse()))


if __name__ == "__main__":
    main()
