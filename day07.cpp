#include <algorithm>
#include <execution>
#include <memory>
#include <numeric>
#include <print>
#include <ranges>

#include <cstdio>

#include <nvtx3/nvtx3.hpp>

template <typename T> struct Grid {
  std::unique_ptr<T[]> owned{nullptr};
  std::size_t stride{0};
  std::size_t width{0};
  std::size_t height{0};
  auto data() -> T* { return owned.get(); }
  auto begin() -> T* { return data(); }
  auto end() -> T* { return data() + stride * height; }
};

auto parse() -> Grid<char> {
  nvtx3::scoped_range _{"Input Parsing"};
  FILE *file = fopen("inputs/day07.in", "rb");
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  rewind(file);
  std::unique_ptr owned{std::make_unique<char[]>(size)};
  fread(owned.get(), size, 1, file);
  fclose(file);
  std::size_t const width(std::ranges::distance(owned.get(), std::ranges::find(owned.get(), owned.get() + size, '\n')));
  auto const start{std::ranges::find(owned.get(), owned.get() + size, 'S')};
  // mark the first step
  *(start + width + 1) = '|';
  return Grid<char>{.owned = std::move(owned),
                    .stride = width + 1,
                    .width = width,
                    .height = size / (width + 1)};
}

constexpr inline auto exec_pol = std::execution::par;

auto part1(Grid<char>& grid) -> int {
  nvtx3::scoped_range _{"Part 1"};
  std::size_t const H{grid.height}, W{grid.width}, S{grid.stride};
  std::unique_ptr copy{std::make_unique<char[]>(H * S)};
  std::unique_ptr splits{std::make_unique<int[]>(W)};
  auto split{splits.get()};
  std::copy(exec_pol, grid.begin(), grid.end(), copy.get());
  std::fill_n(exec_pol, split, W, 0);
  for (auto row : std::views::iota(1zu, H / 2)) {
    auto const curr{copy.get() + (2 * row) * S}, above{curr - S}, below{curr + S};
    std::for_each_n(exec_pol, std::views::iota(0).begin(), W, [=](int col) {
          if (auto const cell{above[col]}; cell == '|') {
            if (curr[col] == '^') {
              ++split[col];
              below[col - 1] = below[col + 1] = '|';
            } else {
              below[col] = '|';
            }
          }
        });
  }
  return std::reduce(exec_pol, split, split + W, 0);
}

auto part2(Grid<char>& grid) -> std::uint64_t {
  nvtx3::scoped_range _{"Part 2"};
  std::size_t const H{grid.height}, W{grid.width}, S{grid.stride};
  auto curr_owned{std::make_unique<std::uint64_t[]>(W)}, next_owned{std::make_unique<std::uint64_t[]>(W)};
  auto curr{curr_owned.get()}, next{next_owned.get()};
  std::fill_n(exec_pol, next, W, 0llu);
  std::transform(exec_pol, grid.data(), grid.data() + W, curr, [] (char c) { return (c == 'S') ? 1llu : 0llu; });
  for (auto const level : std::views::iota(1zu, H / 2)) {
    auto const row{grid.data() + 2 * level * S};
    std::for_each_n(exec_pol, std::views::iota(0).begin(), W, [=](int col) {
          next[col] =
            ((row[col] != '^') ? curr[col] : 0) +
            ((row[col + 1] == '^') ? curr[col + 1] : 0) +
            ((row[col - 1] == '^') ? curr[col - 1] : 0);
        });
    std::swap(curr, next);
  }
  return std::reduce(exec_pol, curr, curr + W, 0llu);
}

auto main() -> int {
  nvtx3::scoped_range _{"Day 07"};
  auto grid = parse();
  auto const res1{part1(grid)};
  auto const res2{part2(grid)};
  std::println("{} {}", res1, res2);
  return 0;
}
