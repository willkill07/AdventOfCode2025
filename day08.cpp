#include <iostream>
#include <tuple>
#include <vector>

#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

#include <cuda/std/utility>
#include <nvtx3/nvtx3.hpp>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

struct Data {
    thrust::device_vector<int> pair_i;
    thrust::device_vector<int> pair_j;
    thrust::device_vector<int> x;
    int n;
    int num_pairs;
};

auto parse() -> Data {
  nvtx3::scoped_range _{"Parse Input"};
  FILE *in = fopen("inputs/day08.in", "r");
  std::vector<int> x, y, z;
  int xi, yi, zi;
  while (fscanf(in, "%d,%d,%d", &xi, &yi, &zi) == 3) {
    x.push_back(xi);
    y.push_back(yi);
    z.push_back(zi);
  }
  fclose(in);

  int const n{static_cast<int>(x.size())};
  int const num_pairs{n * (n - 1) / 2};

  thrust::device_vector<int> d_pair_i(num_pairs), d_pair_j(num_pairs);
  thrust::transform(thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    thrust::zip_iterator(
                        thrust::make_tuple(d_pair_i.begin(), d_pair_j.begin())),
                    [=] __device__(int idx) {
                      int left{0}, right{n - 1}, i{0};
                      while (left <= right) {
                        int const mid{(left + right) / 2};
                        int const s{mid * n - mid * (mid + 1) / 2};
                        if (s <= idx) {
                          i = mid;
                          left = mid + 1;
                        } else {
                          right = mid - 1;
                        }
                      }
                      int const s{i * n - i * (i + 1) / 2};
                      return thrust::make_tuple(i, i + 1 + idx - s);
                    });

  thrust::device_vector<int> d_x(x), d_y(y), d_z(z);
  thrust::device_vector<int64_t> d_distances(num_pairs);
  auto const px{thrust::raw_pointer_cast(d_x.data())};
  auto const py{thrust::raw_pointer_cast(d_y.data())};
  auto const pz{thrust::raw_pointer_cast(d_z.data())};
  auto const pi{thrust::raw_pointer_cast(d_pair_i.data())};
  auto const pj{thrust::raw_pointer_cast(d_pair_j.data())};

  thrust::transform(thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_pairs),
                    d_distances.begin(), [=] __device__(int idx) {
                      int const i{pi[idx]};
                      int const j{pj[idx]};
                      int64_t const dx{px[i] - px[j]};
                      int64_t const dy{py[i] - py[j]};
                      int64_t const dz{pz[i] - pz[j]};
                      return dx * dx + dy * dy + dz * dz;
                    });

  thrust::device_vector<int> d_sorted_pair_i(num_pairs), d_sorted_pair_j(num_pairs), d_indices(num_pairs);
  thrust::sequence(d_indices.begin(), d_indices.end());
  thrust::sort_by_key(d_distances.begin(), d_distances.end(), d_indices.begin());
  thrust::gather(d_indices.begin(),
                 d_indices.end(),
                 thrust::make_zip_iterator(thrust::make_tuple(d_pair_i.begin(), d_pair_j.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(d_sorted_pair_i.begin(), d_sorted_pair_j.begin())));
  return Data{.pair_i = d_sorted_pair_i, .pair_j = d_sorted_pair_j, .x = d_x, .n = n, .num_pairs = num_pairs};
}

struct DisjointSet {
  int* parent;
  int* rank;
  int* sizes;
  int n;

  __device__ int find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  __device__ bool unite(int x, int y) {
    int rx{find(x)}, ry{find(y)};
    if (rx == ry) {
      return false;
    }
    if (rank[rx] < rank[ry]) {
      cuda::std::swap(rx, ry);
    }
    parent[ry] = rx;
    if (sizes != nullptr) {
      sizes[rx] += sizes[ry];
      sizes[ry] = 0;
    }
    if (rank[rx] == rank[ry]) {
      ++rank[rx];
    }
    return true;
  }

  __device__ int size(int x) {
    if (parent[x] != x) {
      return 0;
    } else {
      return sizes[x];
    }
  }
};

__global__ void kernel_part1(int const* __restrict__ pair_i,
                             int const* __restrict__ pair_j,
                             int num_pairs,
                             DisjointSet dsu,
                             int64_t* __restrict__ answer) {
  if (threadIdx.x != 0 or blockIdx.x != 0) {
    return;
  }

  int const limit{cuda::std::min(1000, num_pairs)};
  for (int k = 0; k < limit; ++k) {
    dsu.unite(pair_i[k], pair_j[k]);
  }

  int top1{0}, top2{0}, top3{0};
  for (int i = 0; i < dsu.n; ++i) {
    if (int const s{dsu.size(i)}; s > top1) {
      top3 = top2;
      top2 = top1;
      top1 = s;
    } else if (s > top2) {
      top3 = top2;
      top2 = s;
    } else if (s > top3) {
      top3 = s;
    }
  }
  *answer = static_cast<int64_t>(top1) * top2 * top3;
}

auto part1(thrust::device_vector<int> &pair_i,
           thrust::device_vector<int> &pair_j,
           int n,
           int num_pairs)
    -> int64_t {
  nvtx3::scoped_range _{"Part 1"};
  thrust::device_vector<int> parent(n), rank(n), sizes(n);
  thrust::device_vector<int64_t> answer(1);
  thrust::sequence(parent.begin(), parent.end());
  thrust::fill(rank.begin(), rank.end(), 0);
  thrust::fill(sizes.begin(), sizes.end(), 1);  // Each node starts as size 1

  DisjointSet dsu{
    thrust::raw_pointer_cast(parent.data()),
    thrust::raw_pointer_cast(rank.data()),
    thrust::raw_pointer_cast(sizes.data()),
    n
  };

  kernel_part1<<<1, 1>>>(thrust::raw_pointer_cast(pair_i.data()),
                         thrust::raw_pointer_cast(pair_j.data()),
                         num_pairs,
                         dsu,
                         thrust::raw_pointer_cast(answer.data()));
  cudaDeviceSynchronize();
  return answer[0];
}

__global__ void kernel_part2(int const* __restrict__ pair_i,
                             int const* __restrict__ pair_j,
                             int const* __restrict__ x_coords,
                             int num_pairs,
                             DisjointSet dsu,
                             int64_t* __restrict__ answer) {
  if (threadIdx.x != 0 or blockIdx.x != 0) {
    return;
  }

  int components{dsu.n};
  for (int k = 0; k < num_pairs; ++k) {
    int const pi{pair_i[k]}, pj{pair_j[k]};
    if (dsu.unite(pi, pj)) {
      if (--components == 1) {
        *answer = static_cast<int64_t>(x_coords[pi]) * x_coords[pj];
        break;
      }
    }
  }
}

auto part2(thrust::device_vector<int> &pair_i,
           thrust::device_vector<int> &pair_j,
           thrust::device_vector<int> &x,
           int n,
           int num_pairs)
    -> int64_t {
  nvtx3::scoped_range _{"Part 2"};
  thrust::device_vector<int> parent(n), rank(n);
  thrust::device_vector<int64_t> answer(1);
  thrust::sequence(parent.begin(), parent.end());
  thrust::fill(rank.begin(), rank.end(), 0);

  DisjointSet dsu{
    thrust::raw_pointer_cast(parent.data()),
    thrust::raw_pointer_cast(rank.data()),
    nullptr,
    n
  };

  kernel_part2<<<1, 1>>>(thrust::raw_pointer_cast(pair_i.data()),
                         thrust::raw_pointer_cast(pair_j.data()),
                         thrust::raw_pointer_cast(x.data()),
                         num_pairs,
                         dsu,
                         thrust::raw_pointer_cast(answer.data()));
  cudaDeviceSynchronize();
  return answer[0];
}

auto main() -> int {
  nvtx3::scoped_range _{"Day 08"};
  Data data{parse()};
  int64_t const answer1{part1(data.pair_i, data.pair_j, data.n, data.num_pairs)};
  int64_t const answer2{part2(data.pair_i, data.pair_j, data.x, data.n, data.num_pairs)};
  printf("%ld %ld\n", answer1, answer2);
  return 0;
} 
