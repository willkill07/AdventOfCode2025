#include <vector>

#include <cstdint>
#include <cstdio>

#include <nvtx3/nvtx3.hpp>

struct Data {
    std::vector<int> x;
    std::vector<int> y;
};

auto parse() -> Data {
  nvtx3::scoped_range _{"Parse Input"};
  FILE *in = fopen("inputs/day09.in", "r");
  std::vector<int> x, y, z;
  int xi, yi;
  while (fscanf(in, "%d,%d", &xi, &yi) == 2) {
    x.push_back(xi);
    y.push_back(yi);
  }
  fclose(in);
  return Data{.x = x, .y = y};
}

auto part1(Data const& data) -> int64_t {
  nvtx3::scoped_range _{"Part 1"};
  int64_t largest{0};
  int const n{static_cast<int>(data.x.size())};
  auto const x{data.x.data()}, y{data.y.data()};
  #pragma omp target data map(to: x[0:n], y[0:n])
  {
    #pragma omp target teams distribute parallel for simd collapse(2) reduction(max:largest)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i <= j) {
          continue;
        }
        int const x1{(x[i] < x[j]) ? x[i] : x[j]}, y1{(y[i] < y[j]) ? y[i] : y[j]};
        int const x2{(x[i] > x[j]) ? x[i] : x[j]}, y2{(y[i] > y[j]) ? y[i] : y[j]};
        int64_t const area{static_cast<int64_t>(x2 - x1 + 1) * static_cast<int64_t>(y2 - y1 + 1)};
        if (area > largest) {
          largest = area;
        }
      }
    }
  }
  return largest;
}

auto part2(Data const& data) -> int64_t {
  nvtx3::scoped_range _{"Part 2"};
  int64_t largest{0};
  int const n{static_cast<int>(data.x.size())};
  auto const x{data.x.data()}, y{data.y.data()};
  #pragma omp target data map(to: x[0:n], y[0:n])
  {
    #pragma omp target teams distribute parallel for simd collapse(2) reduction(max:largest)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i <= j) {
          continue;
        }
        int const x1{(x[i] < x[j]) ? x[i] : x[j]}, y1{(y[i] < y[j]) ? y[i] : y[j]};
        int const x2{(x[i] > x[j]) ? x[i] : x[j]}, y2{(y[i] > y[j]) ? y[i] : y[j]};
        int64_t const area{static_cast<int64_t>(x2 - x1 + 1) * static_cast<int64_t>(y2 - y1 + 1)};
        if (area > largest) {
          int k{0};
          while (k < n) {
            int const l{k + 1 + (k + 1 == n ? -n : 0)};
            int const x1a{(x[k] < x[l]) ? x[k] : x[l]}, y1a{(y[k] < y[l]) ? y[k] : y[l]};
            int const x2a{(x[k] > x[l]) ? x[k] : x[l]}, y2a{(y[k] > y[l]) ? y[k] : y[l]};
            if (x1a < x2 and y1a < y2 and x2a > x1 and y2a > y1) {
              break;
            }
            ++k;
          }
          if (k == n) {
            largest = area;
          }
        }
      }
    }
  }
  return largest;
}

auto main() -> int {
  nvtx3::scoped_range _{"Day 09"};
  Data data{parse()};
  int64_t const answer1{part1(data)};
  int64_t const answer2{part2(data)};
  printf("%ld %ld\n", answer1, answer2);
  return 0;
} 
