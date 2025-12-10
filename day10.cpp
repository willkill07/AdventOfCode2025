#include <vector>

#include <cmath>
#include <cstdio>
#include <climits>

#include <cuda_runtime.h>

#include <nvtx3/nvtx3.hpp>

constexpr int MAX_LIGHTS{12}, MAX_BUTTONS{16}, MAX_CONSTRAINTS{128}, MAX_VARS{16}, MAX_BRANCH_DEPTH{8};
constexpr double INF{1e9}, EPS{1e-2};


struct Machine {
    int n;                                       // number of lights
    int target;                                  // target bitmask
    int num_buttons;                             // number of buttons
    int buttons[MAX_BUTTONS];                    // button toggle masks
    int joltage[MAX_LIGHTS];                     // joltage requirements
    int button_lights[MAX_BUTTONS][MAX_LIGHTS];  // which lights each button affects
    int button_num_lights[MAX_BUTTONS];          // how many lights each button affects
};

template <int N>
using Row = double[N];

template <int N>
using Mat = Row<N>*;

template <typename T, typename F>
__inline__ __device__ void warpReduceBroadcast(T& val, F func) {
    T v{val};
    unsigned int active_mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v = func(v, __shfl_down_sync(active_mask, v, offset));
    }
    val = __shfl_sync(active_mask, v, 0);
}

template <typename T, typename I, typename TB = I>
__inline__ __device__ void warpMinLocBroadcast(T& val, I& idx, TB tie_breaker = 0) {
    T v{val};
    I i{idx};
    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T other_val = __shfl_down_sync(mask, v, offset);
        TB other_tb = __shfl_down_sync(mask, tie_breaker, offset);
        I other_idx = __shfl_down_sync(mask, i, offset);
        if (other_val < v or (other_val == v and other_tb < tie_breaker)) {
            v = other_val;
            tie_breaker = other_tb;
            i = other_idx;
        }
    }
    val = __shfl_sync(mask, v, 0);
    idx = __shfl_sync(mask, i, 0);
}

template <typename Idx, typename Fn>
__inline__ __device__ void threadLoop(Idx start, Idx end, Fn fn) {
    for (auto idx{static_cast<Idx>(start + threadIdx.x)}; idx < end; idx += blockDim.x) {
        fn(idx);
    }
}

struct SimplexWorkspace {
    Mat<MAX_VARS + 2> D;
    int* N;
    int* B;

    __device__ void pivot(int m, int n, int r, int s) {
        double const k{1.0 / D[r][s]};
        for (int i{0}; i < m + 2; ++i) {
            if (i == r) {
                continue;
            }
            threadLoop(0, n + 2, [&] (int j) {
                if (j != s) {
                    D[i][j] -= D[r][j] * D[i][s] * k;
                }
            });
        }
        threadLoop(0, n + 2, [&] (int i) {
            D[r][i] *= k;
        });
        threadLoop(0, m + 2, [&] (int i) {
            D[i][s] *= -k;
        });
        if (threadIdx.x == 0) {
            D[r][s] = k;
            int const tmp{B[r]};
            B[r] = N[s];
            N[s] = tmp;
        }
    }

    __device__ int find_pivot(int m, int n, int p) {
        while (true) {
            int s{-1};
            {
                double min_val{INF};
                int min_idx{INT_MAX};
                threadLoop(0, n + 1, [&] (int i) {
                    if (p or N[i] != -1) {
                        double const val{D[m + p][i]};
                        int const idx{N[i]};
                        if (val < min_val or (val == min_val and idx < min_idx)) {
                            min_val = val;
                            min_idx = idx;
                            s = i;
                        }
                    }
                });
                warpMinLocBroadcast(min_val, s, min_idx);
                if (s == -1 or D[m + p][s] > -EPS) {
                    return 1;
                }
            }
            {
                int r{-1};
                double min_ratio{INF};
                int min_idx{INT_MAX};
                threadLoop(0, m, [&] (int i) {
                    if (D[i][s] > EPS) {
                        double const ratio{D[i][n + 1] / D[i][s]};
                        int const idx{B[i]};
                        if (ratio < min_ratio or (ratio == min_ratio and idx < min_idx)) {
                            min_ratio = ratio;
                            min_idx = idx;
                            r = i;
                        }
                    }
                });
                warpMinLocBroadcast(min_ratio, r, min_idx);
                if (r == -1) {
                    return 0;
                }
                pivot(m, n, r, s);
            }
        }
    }

    __device__ double simplex(Mat<MAX_VARS + 1> A, int m, int n, double* x) {
        threadLoop(0, n, [&] (int i) {
            N[i] = i;
        });
        threadLoop(0, m, [&] (int i) {
            B[i] = n + i;
        });
        for (int i{0}; i < m; ++i) {
            threadLoop(0, n, [&] (int j) {
                D[i][j] = A[i][j];
            });
            if (threadIdx.x == 0) {
                D[i][n] = -1;
                D[i][n + 1] = A[i][n];
            }
        }
        threadLoop(0, n, [&] (int j) {
            D[m][j] = 1;
        });
        threadLoop(0, n + 2, [&] (int j) {
            D[m + 1][j] = 0;
        });
        if (threadIdx.x == 0) {
            N[n] = -1;
            D[m][n] = 0;
            D[m][n + 1] = 0;
            D[m + 1][n] = 1;
        }
        __syncwarp();
        int r{0};
        double min_val{D[r][n + 1]};
        threadLoop(1, m, [&] (int i) {
            if (double const val{D[i][n + 1]}; val < min_val) {
                min_val = val;
                r = i;
            }
        });
        warpMinLocBroadcast(min_val, r);
        if (D[r][n + 1] < -EPS) {
            pivot(m, n, r, n);
            if (not find_pivot(m, n, 1) or D[m + 1][n + 1] < -EPS) {
                return INF;
            }
        }
        for (int i{0}; i < m; ++i) {
            if (B[i] == -1) {
                int s{-1};
                double min_val{INF};
                int min_idx{INT_MAX};
                threadLoop(0, n, [&] (int j) {
                    double const val{D[i][j]};
                    int const idx{N[j]};
                    if (val < min_val or (val - min_val < EPS and idx < min_idx)) {
                        min_val = val;
                        min_idx = idx;
                        s = j;
                    }
                });
                warpMinLocBroadcast(min_val, s, min_idx);
                if (s != -1) {
                    pivot(m, n, i, s);
                }
            }
        }
        if (find_pivot(m, n, 0)) {
            threadLoop(0, n, [&] (int i) {
                x[i] = 0;
            });
            threadLoop(0, m, [&] (int i) {
                if (B[i] >= 0 and B[i] < n) {
                    x[B[i]] = D[i][n + 1];
                }
            });
            __syncwarp();
            double obj{0};
            threadLoop(0, n, [&] (int i) {
                obj += x[i];
            });
            warpReduceBroadcast(obj, [] (double a, double b) { return a + b; });
            return obj;
        }
        return INF;
    }
};

struct BranchNode {
    int depth;
    int branch_var;
    int branch_val;
    bool is_upper;
};

struct ILPWorkspace {
    SimplexWorkspace simplex_ws;
    BranchNode* stack;
    double* x;
    Mat<MAX_VARS + 1> A;
    int* constraint_counts;

    __device__ double solve_ilp(int base_m, int n) {
        double best_val{INF};
        int stack_top{0};
        if (threadIdx.x == 0) {
            constraint_counts[0] = base_m;
        }
        __syncwarp();
        double const val{simplex_ws.simplex(A, base_m, n, x)};
        if (val >= best_val - EPS) {
            return best_val;
        }
        int frac_var{-1};
        threadLoop(0, n, [&] (int i) {
            if (fabs(x[i] - round(x[i])) > EPS) {
                frac_var = i;
            }
        });
        warpReduceBroadcast(frac_var, [] (int a, int b) { return (a > b) ? a : b; });
        if (frac_var == -1) {
            return val;
        }
        if (threadIdx.x == 0) {
            int const frac_floor{static_cast<int>(floor(x[frac_var]))};
            stack[stack_top + 0] = BranchNode{1, frac_var, frac_floor, false};
            stack[stack_top + 1] = BranchNode{1, frac_var, frac_floor, true};
        }
        stack_top += 2;
        while (stack_top > 0) {
            --stack_top;
            __syncwarp();
            BranchNode const node{stack[stack_top]};
            int const m{constraint_counts[node.depth - 1]};
            if (m >= MAX_CONSTRAINTS - 1) {
                continue;
            }
            threadLoop(0, n + 1, [&] (int j) {
                A[m][j] = 0;
            });
            if (threadIdx.x == 0) {
                A[m][node.branch_var] = node.is_upper ? -1 : 1;
                A[m][n] = node.is_upper ? -(node.branch_val + 1) : node.branch_val;
                constraint_counts[node.depth] = m + 1;
            }
            __syncwarp();
            double const v{simplex_ws.simplex(A, m + 1, n, x)};
            if (v >= best_val - EPS) {
                continue;
            }
            frac_var = -1;
            for (unsigned int i{threadIdx.x}; i < n; i += blockDim.x) {
                if (fabs(x[i] - round(x[i])) > EPS) {
                    frac_var = i;
                }
            }
            warpReduceBroadcast(frac_var, [] (int a, int b) { return (a > b) ? a : b; });
            if (frac_var == -1) {
                if (v < best_val - EPS) {
                    best_val = v;
                }
            } else if (node.depth < MAX_BRANCH_DEPTH and stack_top + 2 < MAX_BRANCH_DEPTH * 2) {
                int const frac_floor{static_cast<int>(floor(x[frac_var]))};
                if (threadIdx.x == 0) {
                    stack[stack_top + 0] = BranchNode{node.depth + 1, frac_var, frac_floor, false};
                    stack[stack_top + 1] = BranchNode{node.depth + 1, frac_var, frac_floor, true};
                }
                stack_top += 2;
            }
        }
        return best_val;
    }
};


__global__ void part1_kernel(Machine const* machines, int num_machines, int* result) {
    extern __shared__ int shared_mem[];
    __shared__ int front, back, level_end, next_back;
    int const machine_idx{static_cast<int>(blockIdx.x)};
    if (machine_idx >= num_machines) {
        return;
    }
    Machine const& m{machines[machine_idx]};
    int const states{1 << m.n};
    int* const dist{shared_mem};
    int* const queue{shared_mem + states};
    threadLoop(0, states, [&] (int i) {
        dist[i] = -1;
    });
    if (threadIdx.x == 0) {
        dist[0] = 0;
        queue[0] = 0;
        front = 0;
        back = 1;
    }
    while (true) {
        if (threadIdx.x == 0) {
            level_end = back;
            next_back = back;
        }
        __syncthreads();
        if (front >= level_end) {
            break;
        }
        threadLoop(front, level_end, [&] (int idx) {
            int const u{queue[idx]};
            int const d{dist[u]};
            for (int b{0}; b < m.num_buttons; ++b) {
                int const v{u ^ m.buttons[b]};
                if (atomicCAS(&dist[v], -1, d + 1) == -1) {
                    int const pos{atomicAdd(&next_back, 1)};
                    queue[pos] = v;
                }
            }
        });
        if (threadIdx.x == 0) {
            front = level_end;
            back = next_back;
        }
    }
    if (threadIdx.x == 0) {
        atomicAdd(result, dist[m.target]);
    }
}

__global__ void part2_kernel(Machine const* machines, int num_machines, int* result) {
    __shared__ double D[MAX_CONSTRAINTS + 2][MAX_VARS + 2];
    __shared__ int N[MAX_VARS + 2];
    __shared__ int B[MAX_CONSTRAINTS];

    __shared__ BranchNode stack[MAX_BRANCH_DEPTH * 2];
    __shared__ double x[MAX_VARS];
    __shared__ double A[MAX_CONSTRAINTS][MAX_VARS + 1];
    __shared__ int constraint_counts[MAX_BRANCH_DEPTH + 1];

    int const machine_idx{static_cast<int>(blockIdx.x)};
    if (machine_idx >= num_machines) {
        return;
    }
    auto const& m{machines[machine_idx]};
    ILPWorkspace ws{SimplexWorkspace{D, N, B}, stack, x, A, constraint_counts};
    int const num_buttons{m.num_buttons};
    int const n_lights{m.n};
    int const num_constraints{2 * n_lights + num_buttons};
    for (int i{0}; i < num_constraints; ++i) {
        threadLoop(0, num_buttons + 1, [&] (int j) {
            ws.A[i][j] = 0;
        });
    }
    threadLoop(0, num_buttons, [&] (int i) {
        for (int k{0}; k < m.button_num_lights[i]; ++k) {
            int const light{m.button_lights[i][k]};
            ws.A[light][i] = 1;
            ws.A[light + n_lights][i] = -1;
        }
        ws.A[2 * n_lights + num_buttons - 1 - i][i] = -1;
        ws.A[2 * n_lights + num_buttons - 1 - i][num_buttons] = 0;
    });
    threadLoop(0, n_lights, [&] (int i) {
        ws.A[i][num_buttons] = m.joltage[i];
        ws.A[i + n_lights][num_buttons] = -m.joltage[i];
    });
    __syncwarp();
    double const dbl_ans{ws.solve_ilp(num_constraints, num_buttons)};
    if (threadIdx.x == 0) {
        int const ans{static_cast<int>(round(dbl_ans))};
        atomicAdd(result, ans);
    }
}


auto parse() -> std::vector<Machine> {
    nvtx3::scoped_range _{"Parse Input"};
    FILE* f{fopen("inputs/day10.in", "r")};
    fseek(f, 0, SEEK_END);
    long const fsize{ftell(f)};
    fseek(f, 0, SEEK_SET);
    char* const buf{new char[fsize + 1]};
    fread(buf, fsize, 1, f);
    buf[fsize] = '\0';
    fclose(f);
    std::vector<Machine> machines;
    char const* p{buf};
    char const* const end{buf + fsize};
    while (p < end) {
        while (p < end and *p != '[') ++p;
        if (p >= end) {
            break;
        }
        ++p;
        Machine m{};
        m.target = 0;
        m.n = 0;
        while (p < end and *p != ']') {
            if (*p == '#') m.target |= (1 << m.n);
            ++m.n;
            ++p;
        }
        if (p >= end) {
            break;
        }
        ++p;
        m.num_buttons = 0;
        while (p < end and *p != '{') {
            if (*p == '(') {
                ++p;
                int mask{0};
                int num_lights{0};
                while (p < end and *p != ')') {
                    int light{0};
                    while (p < end and *p >= '0' and *p <= '9') {
                        light = light * 10 + (*p - '0');
                        ++p;
                    }
                    mask |= (1 << light);
                    m.button_lights[m.num_buttons][num_lights++] = light;
                    if (p < end and *p == ',') {
                        ++p;
                    }
                }
                m.buttons[m.num_buttons] = mask;
                m.button_num_lights[m.num_buttons] = num_lights;
                ++m.num_buttons;
            }
            if (p < end) {
                ++p;
            }
        }
        if (p < end and *p == '{') {
            ++p;
            for (int i{0}; i < m.n and p < end and *p != '}'; ++i) {
                int val{0};
                while (p < end and *p >= '0' and *p <= '9') {
                    val = val * 10 + (*p - '0');
                    ++p;
                }
                m.joltage[i] = val;
                if (p < end and *p == ',') {
                    ++p;
                }
            }
        }
        machines.push_back(m);
        while (p < end and *p != '\n') {
            ++p;
        }
        if (p < end) {
            ++p;
        }
    }
    delete[] buf;
    return machines;
}


auto part1(Machine* d_machines, int num_machines) -> int {
    nvtx3::scoped_range _{"Part 1"};
    int res{0};
    int* d_res{nullptr};
    cudaMalloc(&d_res, sizeof(int));
    cudaMemset(d_res, 0, sizeof(int));
    part1_kernel<<<num_machines, 128, 32'768>>>(d_machines, num_machines, d_res);
    cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_res);
    return res;
}


auto part2(Machine* d_machines, int num_machines) -> int {
    nvtx3::scoped_range _{"Part 2"};
    int res{0};
    int* d_res{nullptr};
    cudaMalloc(&d_res, sizeof(int));
    cudaMemset(d_res, 0, sizeof(int));
    part2_kernel<<<num_machines, 32>>>(d_machines, num_machines, d_res);
    cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_res);
    return res;
}


int main() {
    cudaSetDevice(0);
    nvtx3::scoped_range _{"Day 10"};
    std::vector<Machine> const machines{parse()};
    auto const num_machines{machines.size()};
    Machine* d_machines{nullptr};
    cudaMalloc(&d_machines, num_machines * sizeof(Machine));
    cudaMemcpy(d_machines, machines.data(), num_machines * sizeof(Machine), cudaMemcpyHostToDevice);
    int result1{part1(d_machines, num_machines)};
    int result2{part2(d_machines, num_machines)};
    cudaFree(d_machines);
    printf("%d %d\n", result1, result2);
    return 0;
}
