#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <cuda_runtime.h>

#include "vtk.h"
#include "data.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

double xlength = 4.0;      /* Width of simulated domain */
double ylength = 1.0;      /* Height of simulated domain */
int imax = 1024;           /* Number of cells horizontally */
int jmax = 256;            /* Number of cells vertically */

int airfoil = 2412;        /* NACA 4-digit Airfoil Spec */

double t_end = 2.0;        /* Simulation runtime */
double del_t = 0.003;      /* Duration of each timestep */
double tau = 0.5;          /* Safety factor for timestep control */

int itermax = 100;         /* Maximum number of iterations in SOR */
double eps = 0.001;        /* Stopping error threshold for SOR */
double omega = 1.7;        /* Relaxation parameter for SOR */
double y = 0.9;            /* Gamma, Upwind differencing factor in PDE discretisation */

double Re = 500.0;         /* Reynolds number */
double ui = 1.0;           /* Initial X velocity */
double vi = 0.0;           /* Initial Y velocity */

double delx, dely;

int fluid_cells = 0;

// Grids used for veclocities, pressure, rhs, flag and temporary f and g arrays
int u_size_x, u_size_y;
double ** u;
int v_size_x, v_size_y;
double ** v;
int p_size_x, p_size_y;
double ** p;
int rhs_size_x, rhs_size_y;
double ** rhs;
int f_size_x, f_size_y;
double ** f;
int g_size_x, g_size_y;
double ** g;
int flag_size_x, flag_size_y;
char ** flag;

double *u_dev = NULL;
double *v_dev = NULL;
double *p_dev = NULL;
double *rhs_dev = NULL;
double *f_dev = NULL;
double *g_dev = NULL;
char   *flag_dev = NULL;

static size_t double_bytes(int m, int n) {
    return (size_t)m * n * sizeof(double);
}

static size_t char_bytes(int m, int n) {
    return (size_t)m * n * sizeof(char);
}

static void alloc_device_buffer(void **ptr, size_t bytes) {
    CUDA_CHECK(cudaMalloc(ptr, bytes));
    CUDA_CHECK(cudaMemset(*ptr, 0, bytes));
}

static void free_device_buffer(void **ptr) {
    if (*ptr) {
        CUDA_CHECK(cudaFree(*ptr));
        *ptr = NULL;
    }
}

/**
 * @brief Allocate a 2D array that is addressable using square brackets
 *
 * @param m The first dimension of the array
 * @param n The second dimension of the array
 * @return double** A 2D array
 */
double **alloc_2d_array(int m, int n) {
    double **rows = (double **)malloc((size_t)m * sizeof(double *));
    if (!rows) {
        fprintf(stderr, "Host allocation failed for row pointers.\n");
        exit(EXIT_FAILURE);
    }

    double *data = (double *)calloc((size_t)m * n, sizeof(double));
    if (!data) {
        fprintf(stderr, "Host allocation failed for double grid.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m; i++) {
        rows[i] = data + (size_t)i * n;
    }

    return rows;
}


/**
 * @brief Allocate a 2D char array that is addressable using square brackets
 *
 * @param m The first dimension of the array
 * @param n The second dimension of the array
 * @return char** A 2D array
 */
char **alloc_2d_char_array(int m, int n) {
	char **rows = (char **)malloc((size_t)m * sizeof(char *));
	if (!rows) {
	    fprintf(stderr, "Host allocation failed for char row pointers.\n");
	    exit(EXIT_FAILURE);
	}

    char *data = (char *)calloc((size_t)m * n, sizeof(char));
    if (!data) {
        fprintf(stderr, "Host allocation failed for char grid.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m; i++) {
        rows[i] = data + (size_t)i * n;
    }

    return rows;
}

/**
 * @brief Free a 2D array
 *
 * @param array The 2D array to free
 */
void free_2d_array(void ** array) {
	if (!array) return;
	free(array[0]);
	free(array);
}

void init_device_storage(void) {
    size_t dbl_bytes = double_bytes(imax + 2, jmax + 2);
    size_t char_by = char_bytes(imax + 2, jmax + 2);
    alloc_device_buffer((void **)&u_dev, dbl_bytes);
    alloc_device_buffer((void **)&v_dev, dbl_bytes);
    alloc_device_buffer((void **)&p_dev, dbl_bytes);
    alloc_device_buffer((void **)&rhs_dev, dbl_bytes);
    alloc_device_buffer((void **)&f_dev, dbl_bytes);
    alloc_device_buffer((void **)&g_dev, dbl_bytes);
    alloc_device_buffer((void **)&flag_dev, char_by);
}

void free_device_storage(void) {
    free_device_buffer((void **)&u_dev);
    free_device_buffer((void **)&v_dev);
    free_device_buffer((void **)&p_dev);
    free_device_buffer((void **)&rhs_dev);
    free_device_buffer((void **)&f_dev);
    free_device_buffer((void **)&g_dev);
    free_device_buffer((void **)&flag_dev);
}

void upload_all_to_device(void) {
    size_t dbl_bytes = double_bytes(imax + 2, jmax + 2);
    size_t char_by = char_bytes(imax + 2, jmax + 2);
    CUDA_CHECK(cudaMemcpy(u_dev, u[0], dbl_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v_dev, v[0], dbl_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(f_dev, f[0], dbl_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_dev, g[0], dbl_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p_dev, p[0], dbl_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rhs_dev, rhs[0], dbl_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(flag_dev, flag[0], char_by, cudaMemcpyHostToDevice));
}

void download_uvp_to_host(void) {
    size_t dbl_bytes = double_bytes(imax + 2, jmax + 2);
    CUDA_CHECK(cudaMemcpy(u[0], u_dev, dbl_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v[0], v_dev, dbl_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(p[0], p_dev, dbl_bytes, cudaMemcpyDeviceToHost));
}
