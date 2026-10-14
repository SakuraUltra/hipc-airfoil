#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#include "data.h"
#include "vtk.h"
#include "setup.h"
#include "boundary.h"
#include "args.h"

#ifdef __cplusplus
extern "C" {
    void set_defaults(void);
    void parse_args(int argc, char **argv);
    void setup(void);
    void allocate_arrays(void);
    void problem_set_up(void);
    void apply_boundary_conditions(void);
    int  write_checkpoint(int iter, double t);
    void free_arrays(void);
    void print_opts(void);
    int  write_result(int iter, double t);
}
#endif

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                            \
                    __FILE__, __LINE__, cudaGetErrorString(err__));              \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define BLOCK_X 32
#define BLOCK_Y 16
#define THREADS_PER_BLOCK (BLOCK_X * BLOCK_Y)
#define offset(i, j, pitch) ((i) * (pitch) + (j))

typedef struct {
    double *u;
    double *v;
    double *f;
    double *g;
    double *p;
    double *rhs;
    char *flag;
} DeviceView;

static inline DeviceView device_view(void) {
    // CUDA is compiled as C++17; avoid C99 designated initializers here.
    return DeviceView{u_dev, v_dev, f_dev, g_dev, p_dev, rhs_dev, flag_dev};
}

// Array offset is i * pitch + j: adjacent warp lanes vary j, not i.
// Keep logical extents in (i, j) order at every call site.
static inline dim3 make_grid(int span_i, int span_j) {
    if (span_i < 1) span_i = 1;
    if (span_j < 1) span_j = 1;
    return dim3((span_j + BLOCK_X - 1) / BLOCK_X,
                (span_i + BLOCK_Y - 1) / BLOCK_Y);
}

static double *reduction_buffer = NULL;
static double reduction_host = 0.0;
static double *residual_partials_dev = NULL;
static size_t residual_partials_capacity = 0;
static int cuda_device = 0;
static int *obstacle_indices_dev = NULL;
static size_t obstacle_count = 0;

static void prepare_obstacle_boundary(void) {
    // Store only active obstacle cells, preserving reference i-major/j-minor
    // order. A single GPU lane can then apply the original scatter rules
    // without scanning the entire fluid domain or racing adjacent cells.
    std::vector<int> indices;
    const int pitch = jmax + 2;
    for (int i = 1; i <= imax; ++i) {
        for (int j = 1; j <= jmax; ++j) {
            if (flag[i][j] & B_NSEW) indices.push_back(offset(i, j, pitch));
        }
    }
    obstacle_count = indices.size();
    if (obstacle_count != 0) {
        CUDA_CHECK(cudaMalloc(&obstacle_indices_dev, obstacle_count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(obstacle_indices_dev, indices.data(),
                              obstacle_count * sizeof(int), cudaMemcpyHostToDevice));
    }
}

static void ensure_reduction_buffer(void) {
    if (!reduction_buffer) {
        CUDA_CHECK(cudaMalloc(&reduction_buffer, sizeof(double)));
    }
}

static void set_reduction_value(double value) {
    ensure_reduction_buffer();
    CUDA_CHECK(cudaMemcpy(reduction_buffer, &value, sizeof(double), cudaMemcpyHostToDevice));
}

static double read_reduction_buffer(void) {
    ensure_reduction_buffer();
    CUDA_CHECK(cudaMemcpy(&reduction_host, reduction_buffer, sizeof(double), cudaMemcpyDeviceToHost));
    return reduction_host;
}

static void ensure_residual_partials(size_t required) {
    if (required <= residual_partials_capacity) {
        return;
    }

    if (residual_partials_dev) {
        CUDA_CHECK(cudaFree(residual_partials_dev));
        residual_partials_dev = NULL;
    }
    CUDA_CHECK(cudaMalloc(&residual_partials_dev, required * sizeof(double)));
    residual_partials_capacity = required;
}

static void release_reduction_helpers(void) {
    if (reduction_buffer) {
        CUDA_CHECK(cudaFree(reduction_buffer));
        reduction_buffer = NULL;
    }
    if (residual_partials_dev) {
        CUDA_CHECK(cudaFree(residual_partials_dev));
        residual_partials_dev = NULL;
    }
    residual_partials_capacity = 0;
    if (obstacle_indices_dev) {
        CUDA_CHECK(cudaFree(obstacle_indices_dev));
        obstacle_indices_dev = NULL;
    }
    obstacle_count = 0;
}

__device__ inline void atomicMaxDouble(double *address, double val) {
    atomicMax((unsigned long long int*)address, __double_as_longlong(val));
}

__global__ void compute_f_kernel(const double *u, const double *v, double *f,
                                 const char *flag, double del_t, double Re,
                                 double gamma, double rdx4, double rdy4,
                                 double rdx2, double rdy2, int imax, int jmax,
                                 int pitch) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i >= imax || j >= jmax + 1) return;

    int idx_ij = offset(i, j, pitch);
    if ((flag[idx_ij] & C_F) && (flag[offset(i + 1, j, pitch)] & C_F)) {
        double uij = u[idx_ij];
        double uipj = u[offset(i + 1, j, pitch)];
        double uimj = u[offset(i - 1, j, pitch)];
        double uijp = u[offset(i, j + 1, pitch)];
        double uijm = u[offset(i, j - 1, pitch)];

        double vij = v[offset(i, j, pitch)];
        double vipj = v[offset(i + 1, j, pitch)];
        double vijm = v[offset(i, j - 1, pitch)];
        double vipjm = v[offset(i + 1, j - 1, pitch)];

        double du2dx = ((uij + uipj) * (uij + uipj) +
                        gamma * fabs(uij + uipj) * (uij - uipj) -
                        (uimj + uij) * (uimj + uij) -
                        gamma * fabs(uimj + uij) * (uimj - uij)) * rdx4;

        double duvdy = ((vij + vipj) * (uij + uijp) +
                        gamma * fabs(vij + vipj) * (uij - uijp) -
                        (vijm + vipjm) * (uijm + uij) -
                        gamma * fabs(vijm + vipjm) * (uijm - uij)) * rdy4;

        double laplu = (uipj - 2.0 * uij + uimj) * rdx2 +
                       (uijp - 2.0 * uij + uijm) * rdy2;

        f[idx_ij] = uij + del_t * (laplu / Re - du2dx - duvdy);
    } else {
        f[idx_ij] = u[idx_ij];
    }
}

__global__ void compute_g_kernel(const double *u, const double *v, double *g,
                                 const char *flag, double del_t, double Re,
                                 double gamma, double rdx4, double rdy4,
                                 double rdx2, double rdy2, int imax, int jmax,
                                 int pitch) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i >= imax + 1 || j >= jmax) return;

    int idx_ij = offset(i, j, pitch);
    if ((flag[idx_ij] & C_F) && (flag[offset(i, j + 1, pitch)] & C_F)) {
        double vij = v[idx_ij];
        double vipj = v[offset(i + 1, j, pitch)];
        double vimj = v[offset(i - 1, j, pitch)];
        double vijp = v[offset(i, j + 1, pitch)];
        double vijm = v[offset(i, j - 1, pitch)];

        double uij = u[offset(i, j, pitch)];
        double uijp = u[offset(i, j + 1, pitch)];
        double uimj = u[offset(i - 1, j, pitch)];
        double uimjp = u[offset(i - 1, j + 1, pitch)];

        double duvdx = ((uij + uijp) * (vij + vipj) +
                        gamma * fabs(uij + uijp) * (vij - vipj) -
                        (uimj + uimjp) * (vimj + vij) -
                        gamma * fabs(uimj + uimjp) * (vimj - vij)) * rdx4;

        double dv2dy = ((vij + vijp) * (vij + vijp) +
                        gamma * fabs(vij + vijp) * (vij - vijp) -
                        (vijm + vij) * (vijm + vij) -
                        gamma * fabs(vijm + vij) * (vijm - vij)) * rdy4;

        double laplv = (vipj - 2.0 * vij + vimj) * rdx2 +
                       (vijp - 2.0 * vij + vijm) * rdy2;

        g[idx_ij] = vij + del_t * (laplv / Re - duvdx - dv2dy);
    } else {
        g[idx_ij] = v[idx_ij];
    }
}

__global__ void set_f_boundaries_kernel(const double *u, double *f,
                                        int imax, int jmax, int pitch) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > jmax) return;

    f[offset(0, j, pitch)] = u[offset(0, j, pitch)];
    f[offset(imax, j, pitch)] = u[offset(imax, j, pitch)];
}

__global__ void set_g_boundaries_kernel(const double *v, double *g,
                                        int imax, int jmax, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > imax) return;

    g[offset(i, 0, pitch)] = v[offset(i, 0, pitch)];
    g[offset(i, jmax, pitch)] = v[offset(i, jmax, pitch)];
}

__global__ void compute_rhs_kernel(const double *f, const double *g,
                                   double *rhs, const char *flag, double rdx,
                                   double rdy, double rdt, int imax, int jmax,
                                   int pitch) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i >= imax + 1 || j >= jmax + 1) return;

    int idx_ij = offset(i, j, pitch);
    if (flag[idx_ij] & C_F) {
        double dfdx = (f[idx_ij] - f[offset(i - 1, j, pitch)]) * rdx;
        double dgdy = (g[idx_ij] - g[offset(i, j - 1, pitch)]) * rdy;
        rhs[idx_ij] = (dfdx + dgdy) * rdt;
    }
}

__global__ void p_norm_kernel(const double *p_vals, const char *flag,
                              int imax, int jmax, int pitch, double *partials) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    __shared__ double shbuf[THREADS_PER_BLOCK];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Out-of-domain lanes still participate in every block barrier.
    double contrib = 0.0;
    if (i <= imax && j <= jmax) {
        int idx_ij = offset(i, j, pitch);
        if (flag[idx_ij] & C_F) {
            double val = p_vals[idx_ij];
            contrib = val * val;
        }
    }
    shbuf[tid] = contrib;
    __syncthreads();

    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shbuf[tid] += shbuf[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        partials[(size_t)blockIdx.y * gridDim.x + blockIdx.x] = shbuf[0];
    }
}

// Second-stage device reduction: transfer one scalar per convergence check,
// rather than copying every block partial to the host. The tree changes the
// floating-point summation order, so validation uses numerical tolerances.
__global__ void sum_partials_kernel(const double *partials, size_t count,
                                    double *sum_out) {
    __shared__ double shbuf[THREADS_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    double sum = 0.0;
    for (size_t k = tid; k < count; k += blockDim.x) sum += partials[k];
    shbuf[tid] = sum;
    __syncthreads();
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < (unsigned int)stride) shbuf[tid] += shbuf[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *sum_out = shbuf[0];
}

static double sum_partials_gpu(size_t count) {
    ensure_reduction_buffer();
    sum_partials_kernel<<<1, THREADS_PER_BLOCK>>>(residual_partials_dev, count,
                                                reduction_buffer);
    CUDA_CHECK(cudaGetLastError());
    return read_reduction_buffer();
}

__global__ void red_black_sor_kernel(double *p_vals, const double *rhs,
                                     const char *flag, double beta_2,
                                     double omega_val, double rdx2,
                                     double rdy2, int color,
                                     int imax, int jmax, int pitch) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i >= imax + 1 || j >= jmax + 1 || ((i + j) & 1) != color) return;

    int idx_ij = offset(i, j, pitch);
    char cell_f = flag[idx_ij];
    if (!(cell_f & C_F)) return;

    double rhs_val = rhs[idx_ij];
    double pi = p_vals[idx_ij];

    if (cell_f == (C_F | B_NSEW)) {
        double star = (p_vals[offset(i + 1, j, pitch)] +
                       p_vals[offset(i - 1, j, pitch)]) * rdx2 +
                      (p_vals[offset(i, j + 1, pitch)] +
                       p_vals[offset(i, j - 1, pitch)]) * rdy2 -
                      rhs_val;
        p_vals[idx_ij] = (1.0 - omega_val) * pi - beta_2 * star;
    } else {
        double eps_E = (flag[offset(i + 1, j, pitch)] & C_F) ? 1.0 : 0.0;
        double eps_W = (flag[offset(i - 1, j, pitch)] & C_F) ? 1.0 : 0.0;
        double eps_N = (flag[offset(i, j + 1, pitch)] & C_F) ? 1.0 : 0.0;
        double eps_S = (flag[offset(i, j - 1, pitch)] & C_F) ? 1.0 : 0.0;

        double denom = (eps_E + eps_W) * rdx2 + (eps_N + eps_S) * rdy2;
        if (denom > 0.0) {
            double beta_mod = -omega_val / denom;
            double star = ((eps_E * p_vals[offset(i + 1, j, pitch)] +
                            eps_W * p_vals[offset(i - 1, j, pitch)]) * rdx2 +
                           (eps_N * p_vals[offset(i, j + 1, pitch)] +
                            eps_S * p_vals[offset(i, j - 1, pitch)]) * rdy2 -
                           rhs_val);
            p_vals[idx_ij] = (1.0 - omega_val) * pi - beta_mod * star;
        }
    }
}

__global__ void residual_kernel(const double *p_vals, const double *rhs,
                                const char *flag, double rdx2, double rdy2,
                                int imax, int jmax, int pitch,
                                double *partials) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    __shared__ double shbuf[THREADS_PER_BLOCK];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    // Zero-fill inactive lanes before the shared-memory reduction.
    if (i <= imax && j <= jmax && (flag[offset(i, j, pitch)] & C_F)) {
        int idx_ij = offset(i, j, pitch);
        double eps_E = (flag[offset(i + 1, j, pitch)] & C_F) ? 1.0 : 0.0;
        double eps_W = (flag[offset(i - 1, j, pitch)] & C_F) ? 1.0 : 0.0;
        double eps_N = (flag[offset(i, j + 1, pitch)] & C_F) ? 1.0 : 0.0;
        double eps_S = (flag[offset(i, j - 1, pitch)] & C_F) ? 1.0 : 0.0;

        double add = (eps_E * (p_vals[offset(i + 1, j, pitch)] - p_vals[idx_ij]) -
                      eps_W * (p_vals[idx_ij] - p_vals[offset(i - 1, j, pitch)])) * rdx2 +
                     (eps_N * (p_vals[offset(i, j + 1, pitch)] - p_vals[idx_ij]) -
                      eps_S * (p_vals[idx_ij] - p_vals[offset(i, j - 1, pitch)])) * rdy2 -
                     rhs[idx_ij];
        contrib = add * add;
    }

    shbuf[tid] = contrib;
    __syncthreads();

    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shbuf[tid] += shbuf[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        partials[block_idx] = shbuf[0];
    }
}

__global__ void update_u_kernel(double *u_vals, const double *f_vals,
                                const double *p_vals, const char *flag,
                                double del_t, double delx,
                                int imax, int jmax, int pitch) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i >= imax - 2 || j >= jmax - 1) return;

    int idx_ij = offset(i, j, pitch);
    if ((flag[idx_ij] & C_F) && (flag[offset(i + 1, j, pitch)] & C_F)) {
        double correction = (p_vals[offset(i + 1, j, pitch)] - p_vals[idx_ij]) * del_t / delx;
        u_vals[idx_ij] = f_vals[idx_ij] - correction;
    }
}

__global__ void update_v_kernel(double *v_vals, const double *g_vals,
                                const double *p_vals, const char *flag,
                                double del_t, double dely,
                                int imax, int jmax, int pitch) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i >= imax - 1 || j >= jmax - 2) return;

    int idx_ij = offset(i, j, pitch);
    if ((flag[idx_ij] & C_F) && (flag[offset(i, j + 1, pitch)] & C_F)) {
        double correction = (p_vals[offset(i, j + 1, pitch)] - p_vals[idx_ij]) * del_t / dely;
        v_vals[idx_ij] = g_vals[idx_ij] - correction;
    }
}

__global__ void apply_bc_west_east_kernel(double *u_vals, double *v_vals,
                                          int imax, int jmax, int pitch) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= jmax + 2) return;

    u_vals[offset(0, j, pitch)] = u_vals[offset(1, j, pitch)];
    v_vals[offset(0, j, pitch)] = v_vals[offset(1, j, pitch)];
    u_vals[offset(imax, j, pitch)] = u_vals[offset(imax - 1, j, pitch)];
    v_vals[offset(imax + 1, j, pitch)] = v_vals[offset(imax, j, pitch)];
}

__global__ void apply_bc_north_south_kernel(double *u_vals, double *v_vals,
                                            int imax, int jmax, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= imax + 2) return;

    v_vals[offset(i, jmax, pitch)] = 0.0;
    u_vals[offset(i, jmax + 1, pitch)] = u_vals[offset(i, jmax, pitch)];

    v_vals[offset(i, 0, pitch)] = 0.0;
    u_vals[offset(i, 0, pitch)] = u_vals[offset(i, 1, pitch)];
}

__global__ void apply_bc_obstacle_kernel(double *u_vals, double *v_vals,
                                         const char *flag_vals,
                                         const int *obstacle_indices,
                                         size_t count, int pitch) {
    // The reference obstacle rule scatters to neighboring velocity locations.
    // Adjacent obstacle cells can write the same location. Preserve its serial
    // lexicographic order instead of introducing an inter-block data race.
    // This intentionally conservative pass is a future optimization target.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (size_t k = 0; k < count; ++k) {
        int idx = obstacle_indices[k];
        int i = idx / pitch;
        int j = idx % pitch;
        char cell_flag = flag_vals[idx];
        if (!(cell_flag & B_NSEW)) {
            continue;
        }

        switch (cell_flag) {
            case B_N:
                v_vals[idx] = 0.0;
                u_vals[idx] = -u_vals[offset(i, j + 1, pitch)];
                u_vals[offset(i - 1, j, pitch)] = -u_vals[offset(i - 1, j + 1, pitch)];
                break;
            case B_E:
                u_vals[idx] = 0.0;
                v_vals[idx] = -v_vals[offset(i + 1, j, pitch)];
                v_vals[offset(i, j - 1, pitch)] = -v_vals[offset(i + 1, j - 1, pitch)];
                break;
            case B_S:
                v_vals[offset(i, j - 1, pitch)] = 0.0;
                u_vals[idx] = -u_vals[offset(i, j - 1, pitch)];
                u_vals[offset(i - 1, j, pitch)] = -u_vals[offset(i - 1, j - 1, pitch)];
                break;
            case B_W:
                u_vals[offset(i - 1, j, pitch)] = 0.0;
                v_vals[idx] = -v_vals[offset(i - 1, j, pitch)];
                v_vals[offset(i, j - 1, pitch)] = -v_vals[offset(i - 1, j - 1, pitch)];
                break;
            case B_NE:
                v_vals[idx] = 0.0;
                u_vals[idx] = 0.0;
                v_vals[offset(i, j - 1, pitch)] = -v_vals[offset(i + 1, j - 1, pitch)];
                u_vals[offset(i - 1, j, pitch)] = -u_vals[offset(i - 1, j + 1, pitch)];
                break;
            case B_SE:
                v_vals[offset(i, j - 1, pitch)] = 0.0;
                u_vals[idx] = 0.0;
                v_vals[idx] = -v_vals[offset(i + 1, j, pitch)];
                u_vals[offset(i - 1, j, pitch)] = -u_vals[offset(i - 1, j - 1, pitch)];
                break;
            case B_SW:
                v_vals[offset(i, j - 1, pitch)] = 0.0;
                u_vals[offset(i - 1, j, pitch)] = 0.0;
                v_vals[idx] = -v_vals[offset(i - 1, j, pitch)];
                u_vals[idx] = -u_vals[offset(i, j - 1, pitch)];
                break;
            case B_NW:
                v_vals[idx] = 0.0;
                u_vals[offset(i - 1, j, pitch)] = 0.0;
                v_vals[offset(i, j - 1, pitch)] = -v_vals[offset(i - 1, j - 1, pitch)];
                u_vals[idx] = -u_vals[offset(i, j + 1, pitch)];
                break;
            default:
                break;
        }
    }
}

__global__ void apply_bc_inlet_kernel(double *u_vals, double *v_vals,
                                      double ui_val, double vi_val,
                                      int jmax, int pitch) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > jmax) return;

    if (j == 0) {
        v_vals[offset(0, 0, pitch)] = 2.0 * vi_val - v_vals[offset(1, 0, pitch)];
        return;
    }

    u_vals[offset(0, j, pitch)] = ui_val;
    v_vals[offset(0, j, pitch)] = 2.0 * vi_val - v_vals[offset(1, j, pitch)];
}

__global__ void max_abs_kernel(const double *vals, int pitch, int i_start,
                               int i_end, int j_start, int j_end,
                               double *max_out) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    double local_max = 0.0;

    int i_stride = blockDim.y * gridDim.y;
    int j_stride = blockDim.x * gridDim.x;

    for (int ii = i_start + blockIdx.y * blockDim.y + threadIdx.y; ii < i_end; ii += i_stride) {
        for (int jj = j_start + blockIdx.x * blockDim.x + threadIdx.x; jj < j_end; jj += j_stride) {
            double val = fabs(vals[offset(ii, jj, pitch)]);
            if (val > local_max) {
                local_max = val;
            }
        }
    }

    __shared__ double shbuf[BLOCK_X * BLOCK_Y];
    shbuf[tid] = local_max;
    __syncthreads();

    for (int stride = BLOCK_X * BLOCK_Y / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shbuf[tid] = fmax(shbuf[tid], shbuf[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxDouble(max_out, shbuf[0]);
    }
}

static void apply_boundary_conditions_gpu(void) {
    DeviceView dev = device_view();
    double *u_dev = dev.u;
    double *v_dev = dev.v;
    char *flag_dev = dev.flag;
    int pitch = jmax + 2;
    int threads_line = 256;
    int blocks_j = (jmax + 2 + threads_line - 1) / threads_line;
    int blocks_i = (imax + 2 + threads_line - 1) / threads_line;

    apply_bc_west_east_kernel<<<blocks_j, threads_line>>>(u_dev, v_dev, imax, jmax, pitch);
    CUDA_CHECK(cudaGetLastError());

    apply_bc_north_south_kernel<<<blocks_i, threads_line>>>(u_dev, v_dev, imax, jmax, pitch);
    CUDA_CHECK(cudaGetLastError());

    if (obstacle_count != 0) {
        apply_bc_obstacle_kernel<<<1, 1>>>(u_dev, v_dev, flag_dev,
                                         obstacle_indices_dev, obstacle_count, pitch);
        CUDA_CHECK(cudaGetLastError());
    }

    apply_bc_inlet_kernel<<<blocks_j, threads_line>>>(u_dev, v_dev, ui, vi, jmax, pitch);
    CUDA_CHECK(cudaGetLastError());
}

static double gpu_max_abs(double *vals, int pitch, int i_start, int i_end,
                          int j_start, int j_end) {
    set_reduction_value(1.0e-10);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid = make_grid(i_end - i_start, j_end - j_start);
    max_abs_kernel<<<grid, block>>>(vals, pitch, i_start, i_end, j_start, j_end, reduction_buffer);
    CUDA_CHECK(cudaGetLastError());

    return read_reduction_buffer();
}

static double compute_residual_gpu(double p0, double rdx2, double rdy2) {
    DeviceView dev = device_view();
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid = make_grid(imax, jmax);

    size_t partial_count = (size_t)grid.x * grid.y;
    ensure_residual_partials(partial_count);

    residual_kernel<<<grid, block>>>(dev.p, dev.rhs, dev.flag, rdx2, rdy2, imax, jmax, jmax + 2, residual_partials_dev);
    CUDA_CHECK(cudaGetLastError());

    double accum = sum_partials_gpu(partial_count);

    if (accum <= 0.0) {
        return 0.0;
    }

    return sqrt(accum / fluid_cells) / p0;
}

void compute_tentative_velocity(void) {
    DeviceView dev = device_view();
    double *u_dev = dev.u;
    double *v_dev = dev.v;
    double *f_dev = dev.f;
    double *g_dev = dev.g;
    char *flag_dev = dev.flag;
    double rdx = 1.0 / delx;
    double rdy = 1.0 / dely;
    double rdx4 = 0.25 * rdx;
    double rdy4 = 0.25 * rdy;
    double rdx2 = rdx * rdx;
    double rdy2 = rdy * rdy;

    int pitch = jmax + 2;

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid_f = make_grid(imax - 1, jmax);
    dim3 grid_g = make_grid(imax, jmax - 1);

    compute_f_kernel<<<grid_f, block>>>(u_dev, v_dev, f_dev, flag_dev, del_t, Re, y, rdx4, rdy4, rdx2, rdy2, imax, jmax, pitch);
    CUDA_CHECK(cudaGetLastError());

    compute_g_kernel<<<grid_g, block>>>(u_dev, v_dev, g_dev, flag_dev, del_t, Re, y, rdx4, rdy4, rdx2, rdy2, imax, jmax, pitch);
    CUDA_CHECK(cudaGetLastError());

    int threads_fb = 256;
    int grid_fb = (jmax + threads_fb - 1) / threads_fb;
    set_f_boundaries_kernel<<<grid_fb, threads_fb>>>(u_dev, f_dev, imax, jmax, pitch);
    CUDA_CHECK(cudaGetLastError());

    int grid_gb = (imax + threads_fb - 1) / threads_fb;
    set_g_boundaries_kernel<<<grid_gb, threads_fb>>>(v_dev, g_dev, imax, jmax, pitch);
    CUDA_CHECK(cudaGetLastError());

}

void compute_rhs(void) {
    DeviceView dev = device_view();
    double rdx = 1.0 / delx;
    double rdy = 1.0 / dely;
    double rdt = 1.0 / del_t;

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid = make_grid(imax, jmax);

    compute_rhs_kernel<<<grid, block>>>(dev.f, dev.g, dev.rhs, dev.flag, rdx, rdy, rdt, imax, jmax, jmax + 2);
    CUDA_CHECK(cudaGetLastError());

}

double poisson(void) {
    DeviceView dev = device_view();
    double *p_dev = dev.p;
    double *rhs_dev = dev.rhs;
    char *flag_dev = dev.flag;
    double rdx2 = 1.0 / (delx * delx);
    double rdy2 = 1.0 / (dely * dely);
    double beta_2 = -omega / (2.0 * (rdx2 + rdy2));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid = make_grid(imax, jmax);

    size_t partial_count = (size_t)grid.x * grid.y;
    ensure_residual_partials(partial_count);
    p_norm_kernel<<<grid, block>>>(p_dev, flag_dev, imax, jmax, jmax + 2,
                                  residual_partials_dev);
    CUDA_CHECK(cudaGetLastError());

    double p0 = sqrt(sum_partials_gpu(partial_count) / fluid_cells);
    if (p0 < 1.0e-4) p0 = 1.0;

    double res = 0.0;
    int iter = 0;

    while (iter < itermax) {
        for (int color = 0; color < 2; color++) {
            red_black_sor_kernel<<<grid, block>>>(p_dev, rhs_dev, flag_dev, beta_2, omega, rdx2, rdy2, color, imax, jmax, jmax + 2);
            CUDA_CHECK(cudaGetLastError());
        }
        iter++;
        res = compute_residual_gpu(p0, rdx2, rdy2);
        if (res < eps) {
            break;
        }
    }

    if (res == 0.0) {
        res = compute_residual_gpu(p0, rdx2, rdy2);
    }

    return res;
}

void update_velocity(void) {
    DeviceView dev = device_view();
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid_u = make_grid(imax - 3, jmax - 2);
    dim3 grid_v = make_grid(imax - 2, jmax - 3);

    update_u_kernel<<<grid_u, block>>>(dev.u, dev.f, dev.p, dev.flag, del_t, delx, imax, jmax, jmax + 2);
    CUDA_CHECK(cudaGetLastError());

    update_v_kernel<<<grid_v, block>>>(dev.v, dev.g, dev.p, dev.flag, del_t, dely, imax, jmax, jmax + 2);
    CUDA_CHECK(cudaGetLastError());

}

void set_timestep_interval(void) {
    if (tau < 1.0e-10) return;

    int pitch = jmax + 2;
    DeviceView dev = device_view();
    double umax = gpu_max_abs(dev.u, pitch, 0, imax + 2, 1, jmax + 2);
    double vmax = gpu_max_abs(dev.v, pitch, 1, imax + 2, 0, jmax + 2);
    double deltu = delx / umax;
    double deltv = dely / vmax;
    double deltRe = 1.0 / (1.0 / (delx * delx) + 1.0 / (dely * dely)) * Re / 2.0;

    double dt_candidate = fmin(fmin(deltu, deltv), deltRe);
    del_t = tau * dt_candidate;
}

int main(int argc, char *argv[]) {
    set_defaults();
    parse_args(argc, argv);
    setup();
    // Argument help/validation does not require a working CUDA device.
    CUDA_CHECK(cudaGetDevice(&cuda_device));

    if (verbose) print_opts();
    allocate_arrays();
    problem_set_up();
    upload_all_to_device();
    prepare_obstacle_boundary();

    // Allocate reduction workspaces before the timed simulation loop.
    dim3 grid = make_grid(imax, jmax);
    ensure_residual_partials((size_t)grid.x * grid.y);
    ensure_reduction_buffer();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));
    double cuda_solve_ms = 0.0;
    double res = 0.0;
    int iters = 0;
    double t;
    const auto wall_start = std::chrono::steady_clock::now();

    for (t = 0.0; t < t_end; t += del_t, iters++) {
        // Per-step stream elapsed time excludes console and VTK output. It
        // includes stream idle time during host-driven convergence checks;
        // this is solve elapsed time, not the sum of individual kernel times.
        CUDA_CHECK(cudaEventRecord(start_ev));
        if (!fixed_dt) set_timestep_interval();
        compute_tentative_velocity();
        compute_rhs();
        res = poisson();
        update_velocity();
        apply_boundary_conditions_gpu();
        CUDA_CHECK(cudaEventRecord(stop_ev));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));
        float step_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&step_ms, start_ev, stop_ev));
        cuda_solve_ms += step_ms;

        if ((iters % output_freq) == 0) {
            printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n",
                   iters, t + del_t, del_t, res);
            if (!no_output && enable_checkpoints) {
                download_uvp_to_host();
                if (write_checkpoint(iters, t + del_t) != 0) {
                    fprintf(stderr, "Failed to write CUDA checkpoint.\n");
                    return EXIT_FAILURE;
                }
            }
        }
    }

    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start).count();
    printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, res);
    printf("CUDA solve elapsed time: %.3f ms\n", cuda_solve_ms);
    printf("Simulation loop wall time: %.6f seconds\n", wall_seconds);
    if (!no_output) {
        download_uvp_to_host();
        if (write_result(iters, t) != 0) {
            fprintf(stderr, "Failed to write CUDA result.\n");
            return EXIT_FAILURE;
        }
    }

    release_reduction_helpers();
    CUDA_CHECK(cudaEventDestroy(start_ev));
    CUDA_CHECK(cudaEventDestroy(stop_ev));
    free_arrays();
    return EXIT_SUCCESS;
}
