#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda/std/atomic>
#include <cuda/std/span>

#include <nvexec/nvtx.cuh>
#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

#include <thrust/device_vector.h>

#include <nvtx3/nvtx3.hpp>

namespace ex = stdexec;

// Graph in CSR format
struct Graph {
  thrust::device_vector<int> out_degree;      // size = num_nodes
  thrust::device_vector<int> row_offsets_rev; // size = num_nodes + 1
  thrust::device_vector<int> col_indices_rev; // size = num_edges
  int num_nodes{0};
  std::unordered_map<std::string, int> node_ids;
};

auto parse() -> Graph {
  nvtx3::scoped_range _{"Input Parsing"};
  FILE *f = fopen("inputs/day11.in", "r");
  fseek(f, 0, SEEK_END);
  long const fsize = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::string buf(fsize + 1, '\0');
  fread(buf.data(), fsize, 1, f);
  fclose(f);
  std::unordered_map<std::string, std::vector<std::string>> adj;
  std::unordered_map<std::string, int> node_ids;
  char const *p{buf.data()};
  char const *const end{buf.data() + fsize};
  auto skip_ws = [&]() {
    while (p < end and (*p == ' ' or *p == '\n' or *p == '\r'))
      ++p;
  };
  auto read_word = [&]() -> std::string {
    skip_ws();
    std::string word;
    while (p < end and *p != ' ' and *p != ':' and *p != '\n' and *p != '\r') {
      word += *p++;
    }
    return word;
  };
  int next_id{0};
  node_ids["out"] = next_id++;
  adj["out"] = {};
  while (p < end) {
    skip_ws();
    if (p >= end)
      break;
    std::string const src{read_word()};
    if (src.empty())
      break;
    while (p < end and *p == ':')
      ++p;
    if (node_ids.find(src) == node_ids.end()) {
      node_ids[src] = next_id++;
    }
    std::vector<std::string> dests;
    while (p < end and *p != '\n') {
      std::string const dst{read_word()};
      if (!dst.empty()) {
        dests.push_back(dst);
        if (node_ids.find(dst) == node_ids.end()) {
          node_ids[dst] = next_id++;
        }
      }
    }
    adj[src] = std::move(dests);
  }
  auto const num_nodes{node_ids.size()};
  std::vector<int> h_row_offsets(num_nodes + 1, 0);
  for (auto const &[src, dests] : adj) {
    auto const src_id{node_ids.at(src)};
    h_row_offsets[src_id + 1] = static_cast<int>(dests.size());
  }
  std::inclusive_scan(h_row_offsets.begin(), h_row_offsets.end(), h_row_offsets.begin());
  int const num_edges{h_row_offsets[num_nodes]};
  std::vector<int> h_col_indices(num_edges, 0);
  std::vector<int> current_offset(num_nodes, 0);
  for (auto const &[src, dests] : adj) {
    auto const src_id{node_ids.at(src)};
    auto const offset{h_row_offsets[src_id]};
    for (auto const &dst : dests) {
      auto const dst_id{node_ids.at(dst)};
      int const idx = offset + current_offset[src_id]++;
      h_col_indices[idx] = dst_id;
    }
  }
  std::vector<int> h_out_degree(num_nodes);
  std::adjacent_difference(h_row_offsets.begin() + 1, h_row_offsets.end(), h_out_degree.begin());
  std::vector<int> h_in_degree(num_nodes, 0);
  for (int const dst_id : h_col_indices) {
    ++h_in_degree[dst_id];
  }
  std::vector<int> h_row_offsets_rev(num_nodes + 1, 0);
  std::inclusive_scan(h_in_degree.begin(), h_in_degree.end(), h_row_offsets_rev.begin() + 1);
  std::vector<int> h_col_indices_rev(num_edges);
  std::vector<int> rev_offset(num_nodes, 0);
  for (size_t u = 0; u < num_nodes; ++u) {
    int const start{h_row_offsets[u]}, end{h_row_offsets[u + 1]};
    for (int const v : cuda::std::span{h_col_indices}.subspan(start, end - start)) {
      int const idx{h_row_offsets_rev[v] + rev_offset[v]++};
      h_col_indices_rev[idx] = static_cast<int>(u);
    }
  }
  return Graph{
      .out_degree = thrust::device_vector<int>(h_out_degree),
      .row_offsets_rev = thrust::device_vector<int>(h_row_offsets_rev),
      .col_indices_rev = thrust::device_vector<int>(h_col_indices_rev),
      .num_nodes = static_cast<int>(num_nodes),
      .node_ids = std::move(node_ids),
  };
}

auto count(Graph const &graph, int source, int dest, auto scheduler,
           auto policy) -> int64_t {
  nvtx3::scoped_range _{"Count"};
  std::size_t const num_nodes(graph.num_nodes);

  thrust::device_vector<int64_t> paths(num_nodes);
  thrust::device_vector<int> out_degree(graph.out_degree.size());
  thrust::device_vector<int> frontier(num_nodes);
  thrust::device_vector<int> next_frontier(num_nodes);
  thrust::device_vector<int> frontier_size(2);
  thrust::copy(graph.out_degree.begin(), graph.out_degree.end(), out_degree.begin());

  cuda::std::span row_offsets_rev_span{thrust::raw_pointer_cast(graph.row_offsets_rev.data()), graph.row_offsets_rev.size()};
  cuda::std::span col_indices_rev_span{thrust::raw_pointer_cast(graph.col_indices_rev.data()), graph.col_indices_rev.size()};

  cuda::std::span paths_span{thrust::raw_pointer_cast(paths.data()), paths.size()};
  cuda::std::span out_degree_span{thrust::raw_pointer_cast(out_degree.data()), out_degree.size()};
  cuda::std::span frontier_span{thrust::raw_pointer_cast(frontier.data()), frontier.size()};
  cuda::std::span next_frontier_span{thrust::raw_pointer_cast(next_frontier.data()), next_frontier.size()};

  cuda::std::span sizes_span{thrust::raw_pointer_cast(frontier_size.data()), frontier_size.size()};

  int size{0};
  bool curr_active{true};
  auto init =
      ex::transfer_just(scheduler, curr_active) |
      nvexec::nvtx::push("Init") |
      ex::bulk(policy, num_nodes,
               [=](int u, bool curr_active) {
                 cuda::std::atomic_ref frontier_size{sizes_span[curr_active]};
                 paths_span[u] = (u == dest) ? 1 : 0;
                 auto const new_val{(u == dest) ? 0 : out_degree_span[u]};
                 out_degree_span[u] = new_val;
                 if (new_val == 0) {
                   frontier_span[frontier_size.fetch_add(1)] = u;
                 }
               }) |
      ex::then([=](bool curr_active) {
        cuda::std::atomic_ref frontier_size{sizes_span[curr_active]};
        return frontier_size.load();
      }) |
      nvexec::nvtx::pop();
  std::tie(size) = ex::sync_wait(std::move(init)).value();
  for (;size > 0; curr_active = not curr_active) {
    auto process =
        ex::transfer_just(scheduler, curr_active) |
        nvexec::nvtx::push("Process") |
        ex::bulk(
            policy, size,
            [=](int j, bool curr_active) {
              cuda::std::span active_span{curr_active ? frontier_span : next_frontier_span};
              cuda::std::span next_span{curr_active ? next_frontier_span : frontier_span};
              cuda::std::atomic_ref next_frontier_size{sizes_span[not curr_active]};
              int const v{active_span[j]};
              int const start{row_offsets_rev_span[v]}, end{row_offsets_rev_span[v + 1]};
              auto const span{col_indices_rev_span.subspan(start, end - start)};
              for (int const u : span) {
                cuda::std::atomic_ref uref{paths_span[u]}, vref{paths_span[v]};
                uref += vref;
                cuda::std::atomic_ref od_ref{out_degree_span[u]};
                if (--od_ref == 0) {
                  int const idx{next_frontier_size++};
                  next_span[idx] = u;
                }
              }
            }) |
        ex::then([=](bool curr_active) {
          cuda::std::atomic_ref curr_size{sizes_span[curr_active]};
          cuda::std::atomic_ref next_size{sizes_span[not curr_active]};
          curr_size.store(0);
          return next_size.exchange(0);
        }) |
        nvexec::nvtx::pop();
    std::tie(size) = ex::sync_wait(std::move(process)).value();
  }
  return paths[source];
}

auto part1(Graph const &graph, auto scheduler, auto policy) -> int64_t {
  nvtx3::scoped_range _{"Part 1"};
  int const you{graph.node_ids.at("you")};
  int const out{graph.node_ids.at("out")};
  return count(graph, you, out, scheduler, policy);
}

auto part2(Graph const &graph, auto scheduler, auto policy) -> int64_t {
  nvtx3::scoped_range _{"Part 2"};
  int const svr{graph.node_ids.at("svr")};
  int const dac{graph.node_ids.at("dac")};
  int const fft{graph.node_ids.at("fft")};
  int const out{graph.node_ids.at("out")};
  auto check = [&](int source, int dest) -> int64_t {
    return count(graph, source, dest, scheduler, policy);
  };
  int64_t const mid1{check(dac, fft)}, mid2{check(fft, dac)};
  int64_t total{0};
  if (mid1 != 0) {
    total = check(svr, dac) * mid1 * check(fft, out);
  }
  if (mid2 != 0) {
    total += check(svr, fft) * mid2 * check(dac, out);
  }
  return total;
}

auto main() -> int {
  nvtx3::scoped_range _{"Day 11"};
  auto const graph{parse()};
  nvexec::stream_context stream_ctx{};
  auto scheduler{stream_ctx.get_scheduler()};
  auto const policy{std::execution::par};
  auto const answer1{part1(graph, scheduler, policy)};
  auto const answer2{part2(graph, scheduler, policy)};
  printf("%ld %ld\n", answer1, answer2);
  return 0;
}
