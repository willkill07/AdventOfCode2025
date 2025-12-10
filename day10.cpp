#include <vector>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <climits>

#include <cuda_runtime.h>

#include <nvtx3/nvtx3.hpp>

constexpr int MAX_LIGHTS{12}, MAX_BUTTONS{16}, MAX_CONSTRAINTS{128}, MAX_VARS{20}, MAX_BRANCH_DEPTH{32};
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

struct SimplexWorkspace {
    double D[MAX_CONSTRAINTS + 2][MAX_VARS + 2];
    int N[MAX_VARS + 2];
    int B[MAX_CONSTRAINTS];

    __device__ void pivot(int m, int n, int r, int s) {
        double const k{1.0 / D[r][s]};
        for (int i{0}; i < m + 2; ++i) {
            if (i == r) {
                continue;
            }
            for (int j{0}; j < n + 2; ++j) {
                if (j != s) {
                    D[i][j] -= D[r][j] * D[i][s] * k;
                }
            }
        }
        for (int i{0}; i < n + 2; ++i) {
            D[r][i] *= k;
        }
        for (int i{0}; i < m + 2; ++i) {
            D[i][s] *= -k;
        }
        D[r][s] = k;
        int const tmp{B[r]};
        B[r] = N[s];
        N[s] = tmp;
    }

    __device__ int find_pivot(int m, int n, int p) {
        while (true) {
            int s{-1};
            {
                double min_val{INF};
                int min_idx{INT_MAX};
                for (int i{0}; i <= n; ++i) {
                    if (p or N[i] != -1) {
                        double const val{D[m + p][i]};
                        int const idx{N[i]};
                        if (val < min_val or (val == min_val and idx < min_idx)) {
                            min_val = val;
                            min_idx = idx;
                            s = i;
                        }
                    }
                }
                if (s == -1 or D[m + p][s] > -EPS) {
                    return 1;
                }
            }
            {
                int r{-1};
                double min_ratio{INF};
                int min_idx{INT_MAX};
                for (int i{0}; i < m; ++i) {
                    if (D[i][s] > EPS) {
                        double const ratio{D[i][n + 1] / D[i][s]};
                        int const idx{B[i]};
                        if (ratio < min_ratio or (ratio == min_ratio and idx < min_idx)) {
                            min_ratio = ratio;
                            min_idx = idx;
                            r = i;
                        }
                    }
                }
                if (r == -1) {
                    return 0;
                }
                pivot(m, n, r, s);
            }
        }
    }

    __device__ double simplex(double A[][MAX_VARS + 1], int m, int n, double x[]) {
        for (int i{0}; i < n; ++i) {
            N[i] = i;
        }
        N[n] = -1;
        for (int i{0}; i < m; ++i) {
            B[i] = n + i;
        }
        for (int i{0}; i < m; ++i) {
            for (int j{0}; j < n; ++j) {
                D[i][j] = A[i][j];
            }
            D[i][n] = -1;
            D[i][n + 1] = A[i][n];
        }
        for (int j{0}; j < n; ++j) {
            D[m][j] = 1;
        }
        D[m][n] = 0;
        D[m][n + 1] = 0;
        for (int j{0}; j <= n + 1; ++j) {
            D[m + 1][j] = 0;
        }
        D[m + 1][n] = 1;
        int r{0};
        for (int i{1}; i < m; ++i) {
            if (D[i][n + 1] < D[r][n + 1]) {
                r = i;
            }
        }
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
                for (int j{0}; j < n; ++j) {
                    double const val{D[i][j]};
                    int const idx{N[j]};
                    if (val < min_val or (fabs(val - min_val) < EPS and idx < min_idx)) {
                        min_val = val;
                        min_idx = idx;
                        s = j;
                    }
                }
                if (s != -1) {
                    pivot(m, n, i, s);
                }
            }
        }
        if (find_pivot(m, n, 0)) {
            for (int i{0}; i < n; ++i) {
                x[i] = 0;
            }
            for (int i{0}; i < m; ++i) {
                if (B[i] >= 0 and B[i] < n) {
                    x[B[i]] = D[i][n + 1];
                }
            }
            double obj{0};
            for (int i{0}; i < n; ++i) {
                obj += x[i];
            }
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
    BranchNode stack[MAX_BRANCH_DEPTH * 2];
    double A[MAX_CONSTRAINTS][MAX_VARS + 1];
    int constraint_counts[MAX_BRANCH_DEPTH + 1];
    double x[MAX_VARS];

    __device__ double solve_ilp(int base_m, int n) {
        double best_val{INF};
        int stack_top{0};
        constraint_counts[0] = base_m;
        double const val{simplex_ws.simplex(A, base_m, n, x)};
        if (val >= best_val - EPS or val >= INF - EPS) {
            return best_val;
        }
        int frac_var{-1}, frac_floor{0};
        for (int i{0}; i < n; ++i) {
            if (fabs(x[i] - round(x[i])) > EPS) {
                frac_var = i;
                frac_floor = static_cast<int>(floor(x[i]));
                break;
            }
        }
        if (frac_var == -1) {
            return val;
        }
        stack[stack_top++] = BranchNode{1, frac_var, frac_floor, false};
        stack[stack_top++] = BranchNode{1, frac_var, frac_floor, true};
        while (stack_top > 0) {
            BranchNode const node{stack[--stack_top]};
            int const m{constraint_counts[node.depth - 1]};
            if (m >= MAX_CONSTRAINTS - 1) {
                continue;
            }
            for (int j{0}; j <= n; ++j) {
                A[m][j] = 0;
            }
            A[m][node.branch_var] = node.is_upper ? -1 : 1;
            A[m][n] = node.is_upper ? -(node.branch_val + 1) : node.branch_val;
            constraint_counts[node.depth] = m + 1;
            double const v{simplex_ws.simplex(A, m + 1, n, x)};
            if (v >= best_val - EPS or v >= INF - EPS) {
                continue;
            }
            frac_var = -1;
            for (int i{0}; i < n; ++i) {
                if (fabs(x[i] - round(x[i])) > EPS) {
                    frac_var = i;
                    frac_floor = static_cast<int>(floor(x[i]));
                    break;
                }
            }
            if (frac_var == -1) {
                if (v < best_val - EPS) {
                    best_val = v;
                }
            } else if (node.depth < MAX_BRANCH_DEPTH and stack_top + 2 < MAX_BRANCH_DEPTH * 2) {
                stack[stack_top++] = BranchNode{node.depth + 1, frac_var, frac_floor, false};
                stack[stack_top++] = BranchNode{node.depth + 1, frac_var, frac_floor, true};
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
    for (int i{static_cast<int>(threadIdx.x)}; i < states; i += static_cast<int>(blockDim.x)) {
        dist[i] = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        dist[0] = 0;
        queue[0] = 0;
        front = 0;
        back = 1;
    }
    __syncthreads();
    while (true) {
        if (threadIdx.x == 0) {
            level_end = back;
            next_back = back;
        }
        __syncthreads();
        if (front >= level_end) {
            break;
        }
        for (int idx = front + threadIdx.x; idx < level_end; idx += blockDim.x) {
            int const u{queue[idx]};
            int const d{dist[u]};
            for (int b{0}; b < m.num_buttons; ++b) {
                int const v{u ^ m.buttons[b]};
                if (atomicCAS(&dist[v], -1, d + 1) == -1) {
                    int const pos{atomicAdd(&next_back, 1)};
                    queue[pos] = v;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            front = level_end;
            back = next_back;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(result, dist[m.target]);
    }
}


__global__ void part2_kernel(Machine const* machines, int num_machines, ILPWorkspace* workspaces, int* result) {
    int const machine_idx{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    if (machine_idx >= num_machines) {
        return;
    }
    auto const& m{machines[machine_idx]};
    auto& ws{workspaces[machine_idx]};
    int const num_buttons{m.num_buttons};
    int const n_lights{m.n};
    int const num_constraints{2 * n_lights + num_buttons};
    for (int i{0}; i < num_constraints; ++i) {
        for (int j{0}; j <= num_buttons; ++j) {
            ws.A[i][j] = 0;
        }
    }
    for (int i{0}; i < num_buttons; ++i) {
        for (int k{0}; k < m.button_num_lights[i]; ++k) {
            int const light{m.button_lights[i][k]};
            ws.A[light][i] = 1;
            ws.A[light + n_lights][i] = -1;
        }
    }
    for (int i{0}; i < n_lights; ++i) {
        ws.A[i][num_buttons] = m.joltage[i];
        ws.A[i + n_lights][num_buttons] = -m.joltage[i];
    }
    for (int i{0}; i < num_buttons; ++i) {
        ws.A[2 * n_lights + num_buttons - 1 - i][i] = -1;
        ws.A[2 * n_lights + num_buttons - 1 - i][num_buttons] = 0;
    }
    double const dbl_ans{ws.solve_ilp(num_constraints, num_buttons)};
    int const ans{static_cast<int>(round(dbl_ans))};
    atomicAdd(result, ans);
}


auto parse() -> std::vector<Machine> {
    FILE* f{fopen("inputs/day10.in", "r")};
    std::vector<Machine> machines;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' or line[0] == '\0') {
            continue;
        }
        Machine m{};
        char* p{strchr(line, '[')};
        if (not p) {
            continue;
        }
        ++p;
        char* end{strchr(p, ']')};
        if (not end) {
            continue;
        }
        m.n = static_cast<int>(end - p);
        m.target = 0;
        for (int i{0}; i < m.n; ++i) {
            if (p[i] == '#') {
                m.target |= (1 << i);
            }
        }
        p = end + 1;
        m.num_buttons = 0;
        while ((p = strchr(p, '(')) != nullptr) {
            ++p;
            end = strchr(p, ')');
            if (not end) {
                break;
            }
            int mask{0};
            m.button_num_lights[m.num_buttons] = 0;

            for (char* comma{p}; comma < end; ++comma) {
                int const light{atoi(comma)};
                mask |= (1 << light);
                m.button_lights[m.num_buttons][m.button_num_lights[m.num_buttons]++] = light;
                comma = strchr(comma, ',');
                if (not comma or comma >= end) {
                    break;
                }
            }
            m.buttons[m.num_buttons] = mask;
            m.num_buttons++;
            p = end + 1;
        }
        p = strchr(line, '{');
        if (p++) {
            for (int i{0}; i < m.n and *p and *p != '}'; ++i, ++p) {
                m.joltage[i] = atoi(p);
                p = strchr(p, ',');
                if (not p) {
                    break;
                }
            }
        }
        machines.push_back(m);
    }
    fclose(f);
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
    ILPWorkspace* d_workspaces{nullptr};
    int res{0};
    int* d_res{nullptr};
    cudaMalloc(&d_res, sizeof(int));
    cudaMemset(d_res, 0, sizeof(int));
    cudaMalloc(&d_workspaces, num_machines * sizeof(ILPWorkspace));
    part2_kernel<<<num_machines, 1>>>(d_machines, num_machines, d_workspaces, d_res);
    cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_res);
    cudaFree(d_workspaces);
    return res;
}


int main() {
    nvtx3::scoped_range _{"Day 10"};
    std::vector<Machine> const machines{parse()};
    auto const num_machines{machines.size()};
    Machine* d_machines{nullptr};
    cudaMalloc(&d_machines, num_machines * sizeof(Machine));
    cudaMemcpy(d_machines, machines.data(), num_machines * sizeof(Machine), cudaMemcpyHostToDevice);
    int const result1{part1(d_machines, num_machines)};
    int const result2{part2(d_machines, num_machines)};
    cudaFree(d_machines);
    printf("%d %d\n", result1, result2);
    return 0;
}
