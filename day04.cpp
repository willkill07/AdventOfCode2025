#include <algorithm>
#include <print>

#include <cstdio>

#include <nvtx3/nvtx3.hpp>

template <typename T>
struct Grid {
  T *data{nullptr};
  std::size_t stride{0};
  std::size_t width{0};
  std::size_t height{0};
  ~Grid() { delete[] data; }
};

auto parse() -> Grid<char> {
  nvtx3::scoped_range _{"Input Parsing"};
  FILE *file = fopen("inputs/day04.in", "rb");
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  rewind(file);
  auto const buf = new char[size];
  fread(buf, size, 1, file);
  fclose(file);
  std::size_t const width(std::ranges::distance(buf, std::ranges::find(buf, buf + size, '\n')));
  return Grid<char>{.data = buf, .stride = width + 1, .width = width, .height = size / (width + 1)};
}

template <bool Remove>
auto kernel(char* data, int width, int height, int stride) -> int {
  nvtx3::scoped_range _{"Step"};
  int accessible{0};
  #pragma acc data present(data)
  #pragma acc parallel loop collapse(2) reduction(+:accessible)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (int const off{y * stride + x}; data[off] == '@') {
        int count{0};
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx or dy) {
              if (int const nx{x + dx}, ny{y + dy}; 0 <= nx and nx < width and 0 <= ny and ny < height) {
                if (data[ny * stride + nx] == '@') {
                  ++count;
                }
              }
            }
          }
        }
        if (count < 4) {
          ++accessible;
          if constexpr(Remove) {
            data[off] = '.';
          }
        }
      }
    }
  }
  return accessible;
}

auto part1(char* data, int width, int height, int stride) -> int {
  nvtx3::scoped_range _{"Part 1"};
  return kernel<false>(data, width, height, stride);
}

auto part2(char* data, int width, int height, int stride) -> int {
  nvtx3::scoped_range _{"Part 2"};
  int sum{0};
  while (true) {
    if (int result = kernel<true>(data, width, height, stride); result) {
      sum += result;
    } else {
      return sum;
    }
  }
}

auto main() -> int {
  nvtx3::scoped_range _{"Day 04"};
  Grid grid{parse()};
  int const width(grid.width), height(grid.height), stride(grid.stride);
  char* const data{grid.data};
  #pragma acc data copyin(data[0:width * stride])
  {
    int const r1{part1(data, width, height, stride)};
    int const r2{part2(data, width, height, stride)};
    std::println("{} {}", r1, r2);
  }
  return 0;
}
