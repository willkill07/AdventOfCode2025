#include <algorithm>
#include <memory>
#include <span>

#include <cstdio>

#include <cuda/std/array>

#include <nvtx3/nvtx3.hpp>

constexpr inline auto NUM_SHAPES{6};

struct alignas(16) Region {
    cuda::std::array<int, NUM_SHAPES> counts;
    int area;
};

struct Data {
    std::unique_ptr<Region[]> regions;
    std::size_t num_regions;
    cuda::std::array<int, NUM_SHAPES> shapes;
};

template <typename T, std::size_t Digits>
auto parse_number(std::span<T, Digits> input) -> int {
  int result{0};
  for (std::size_t i{0}; i < Digits; ++i) {
    result = result * 10 + (input[i] - '0');
  }
  return result;
}

auto parse() -> Data {
  nvtx3::scoped_range _{"Input Parsing"};
  FILE *f = fopen("inputs/day12.in", "r");
  fseek(f, 0, SEEK_END);
  auto const fsize{static_cast<std::size_t>(ftell(f))};
  fseek(f, 0, SEEK_SET);
  auto buf = std::make_unique<char[]>(fsize);
  fread(buf.get(), fsize, 1, f);
  fclose(f);
  std::span<char> input{buf.get(), fsize};
   
  constexpr auto SHAPE_BUFFER_SIZE{16};
  constexpr auto REGION_BUFFER_SIZE{25};
  constexpr auto REGION_OFFSET{7};
  constexpr auto WIDTH_NUM{3};

  nvtx3::scoped_range __{"Parse"};
  cuda::std::array shapes {
    static_cast<int>(std::ranges::count(input.subspan<SHAPE_BUFFER_SIZE * 0, SHAPE_BUFFER_SIZE>(), '#')),
    static_cast<int>(std::ranges::count(input.subspan<SHAPE_BUFFER_SIZE * 1, SHAPE_BUFFER_SIZE>(), '#')),
    static_cast<int>(std::ranges::count(input.subspan<SHAPE_BUFFER_SIZE * 2, SHAPE_BUFFER_SIZE>(), '#')),
    static_cast<int>(std::ranges::count(input.subspan<SHAPE_BUFFER_SIZE * 3, SHAPE_BUFFER_SIZE>(), '#')),
    static_cast<int>(std::ranges::count(input.subspan<SHAPE_BUFFER_SIZE * 4, SHAPE_BUFFER_SIZE>(), '#')),
    static_cast<int>(std::ranges::count(input.subspan<SHAPE_BUFFER_SIZE * 5, SHAPE_BUFFER_SIZE>(), '#')),
  };
  input = input.subspan(SHAPE_BUFFER_SIZE * NUM_SHAPES);
  std::size_t const num_regions{input.size() / REGION_BUFFER_SIZE};
  std::unique_ptr<Region[]> regions{std::make_unique<Region[]>(num_regions)};
  std::ranges::generate(std::span(regions.get(), num_regions), [&input] () {
    auto s{input.subspan<0, REGION_BUFFER_SIZE>()};
    input = input.subspan<REGION_BUFFER_SIZE>();
    int const area{parse_number(s.subspan<0, 2>()) * parse_number(s.subspan<WIDTH_NUM, 2>())};
    return Region{.counts = cuda::std::array{
        parse_number(s.subspan<REGION_OFFSET + WIDTH_NUM * 0, 2>()),
        parse_number(s.subspan<REGION_OFFSET + WIDTH_NUM * 1, 2>()),
        parse_number(s.subspan<REGION_OFFSET + WIDTH_NUM * 2, 2>()),
        parse_number(s.subspan<REGION_OFFSET + WIDTH_NUM * 3, 2>()),
        parse_number(s.subspan<REGION_OFFSET + WIDTH_NUM * 4, 2>()),
        parse_number(s.subspan<REGION_OFFSET + WIDTH_NUM * 5, 2>()),
    }, .area = area};
  });
  return Data{.regions = std::move(regions), .num_regions = num_regions, .shapes = shapes};
}

auto part1(Data const &data) -> int {
  nvtx3::scoped_range _{"Part 1"};
  int total{0};
  auto const regions{data.regions.get()};
  auto const shapes{data.shapes.data()};
  auto const num_regions{data.num_regions};
  #pragma acc parallel loop reduction(+:total) copyin(regions[0:num_regions], shapes[0:NUM_SHAPES])
  for (std::size_t i{0}; i < num_regions; ++i) {
    int required{0};
    #pragma unroll
    for (std::size_t j{0}; j < NUM_SHAPES; ++j) {
        required += regions[i].counts[j] * shapes[j];
    }
    total += (required < regions[i].area);
  }
  return total;
}

auto main() -> int {
  nvtx3::scoped_range _{"Day 12"};
  printf("%d\n", part1(parse()));
  return 0;
}
