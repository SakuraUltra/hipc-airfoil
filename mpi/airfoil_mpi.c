#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>

#include "data.h"
#include "vtk.h"
#include "setup.h"
#include "boundary.h"
#include "args.h"
double rdx, rdy, rdt;

/* MPI Globals */
int rank, size;
int local_imax;         // Number of columns per process (X-axis)
int global_imax_mpi;    // Global grid width (for VTK output)
int left_neighbor, right_neighbor; // Neighbor ranks

void halo_exchange(double **array) {
    MPI_Status status;
    int count = jmax + 2;

    MPI_Sendrecv(array[local_imax], count, MPI_DOUBLE, right_neighbor, 0,
                 array[0],          count, MPI_DOUBLE, left_neighbor,  0,
                 MPI_COMM_WORLD, &status);

    MPI_Sendrecv(array[1],            count, MPI_DOUBLE, left_neighbor,  1,
                 array[local_imax+1], count, MPI_DOUBLE, right_neighbor, 1,
                 MPI_COMM_WORLD, &status);
}

void halo_exchange_char(char **array) {
    MPI_Status status;
    int count = jmax + 2;

    MPI_Sendrecv(array[local_imax], count, MPI_CHAR, right_neighbor, 0,
                 array[0],          count, MPI_CHAR, left_neighbor,  0,
                 MPI_COMM_WORLD, &status);

    MPI_Sendrecv(array[1],            count, MPI_CHAR, left_neighbor,  1,
                 array[local_imax+1], count, MPI_CHAR, right_neighbor, 1,
                 MPI_COMM_WORLD, &status);
}

/**
 * @brief Computation of tentative velocity field (f, g)
 *
 */
void compute_tentative_velocity() {
    halo_exchange(u);
    halo_exchange(v);

    double rdx4 = 0.25 * rdx;
    double rdy4 = 0.25 * rdy;
    double rdx2 = rdx * rdx;
    double rdy2 = rdy * rdy;

    for (int i = 1; i <= imax; i++) {
        for (int j = 1; j < jmax+1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                double du2dx = ((u[i][j] + u[i+1][j]) * (u[i][j] + u[i+1][j]) +
                                y * fabs(u[i][j] + u[i+1][j]) * (u[i][j] - u[i+1][j]) -
                                (u[i-1][j] + u[i][j]) * (u[i-1][j] + u[i][j]) -
                                y * fabs(u[i-1][j] + u[i][j]) * (u[i-1][j]-u[i][j]))
                                * rdx4;
                double duvdy = ((v[i][j] + v[i+1][j]) * (u[i][j] + u[i][j+1]) +
                                y * fabs(v[i][j] + v[i+1][j]) * (u[i][j] - u[i][j+1]) -
                                (v[i][j-1] + v[i+1][j-1]) * (u[i][j-1] + u[i][j]) -
                                y * fabs(v[i][j-1] + v[i+1][j-1]) * (u[i][j-1] - u[i][j]))
                                * rdy4;
                double laplu = (u[i+1][j] - 2.0 * u[i][j] + u[i-1][j]) * rdx2 +
                                (u[i][j+1] - 2.0 * u[i][j] + u[i][j-1]) * rdy2;

                f[i][j] = u[i][j] + del_t * (laplu / Re - du2dx - duvdy);
            } else {
                f[i][j] = u[i][j];
            }
        }
    }
    for (int i = 1; i < imax+1; i++) {
        for (int j = 1; j < jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                double duvdx = ((u[i][j] + u[i][j+1]) * (v[i][j] + v[i+1][j]) +
                                y * fabs(u[i][j] + u[i][j+1]) * (v[i][j] - v[i+1][j]) -
                                (u[i-1][j] + u[i-1][j+1]) * (v[i-1][j] + v[i][j]) -
                                y * fabs(u[i-1][j] + u[i-1][j+1]) * (v[i-1][j]-v[i][j]))
                                * rdx4;
                double dv2dy = ((v[i][j] + v[i][j+1]) * (v[i][j] + v[i][j+1]) +
                                y * fabs(v[i][j] + v[i][j+1]) * (v[i][j] - v[i][j+1]) -
                                (v[i][j-1] + v[i][j]) * (v[i][j-1] + v[i][j]) -
                                y * fabs(v[i][j-1] + v[i][j]) * (v[i][j-1] - v[i][j]))
                                * rdy4;
                double laplv = (v[i+1][j] - 2.0 * v[i][j] + v[i-1][j]) * rdx2 +
                                (v[i][j+1] - 2.0 * v[i][j] + v[i][j-1]) * rdy2;

                g[i][j] = v[i][j] + del_t * (laplv / Re - duvdx - dv2dy);
            } else {
                g[i][j] = v[i][j];
            }
        }
    }

    if (left_neighbor == MPI_PROC_NULL) {
        for (int j = 1; j < jmax+1; j++) {
            f[0][j] = u[0][j];
        }
    }

    if (right_neighbor == MPI_PROC_NULL) {
        for (int j = 1; j < jmax+1; j++) {
            f[imax][j] = u[imax][j];
        }
    }

    for (int i = 1; i <= imax+1; i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }

    halo_exchange(f);
    halo_exchange(g);
}


/**
 * @brief Calculate the right hand side of the pressure equation
 *
 */
void compute_rhs() {
    for (int i = 1; i < imax+1; i++) {
        for (int j = 1;j < jmax+1; j++) {
            if (flag[i][j] & C_F) {
                /* only for fluid and non-surface cells */
                rhs[i][j] = ((f[i][j] - f[i-1][j]) * rdx +
                             (g[i][j] - g[i][j-1]) * rdy)
                             * rdt;
            }
        }
    }
}


/**
 * @brief Red/Black SOR to solve the poisson equation.
 *
 * @return Calculated residual of the computation
 *
 */
double poisson() {
    double rdx2 = 1.0 / (delx * delx);
    double rdy2 = 1.0 / (dely * dely);
    double beta_2 = -omega / (2.0 * (rdx2 + rdy2));

    double p0_local = 0.0;
    /* Calculate sum of squares */
    for (int i = 1; i < imax+1; i++) {
        for (int j = 1; j < jmax+1; j++) {
            if (flag[i][j] & C_F) { p0_local += p[i][j] * p[i][j]; }
        }
    }

    double p0 = 0.0;
    MPI_Allreduce(&p0_local, &p0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    p0 = sqrt(p0 / fluid_cells);
    if (p0 < 0.0001) { p0 = 1.0; }

    /* Red/Black SOR-iteration */
    int iter;
    double res = 0.0;
    for (iter = 0; iter < itermax; iter++) {
        for (int rb = 0; rb < 2; rb++) {
            for (int i = 1; i < imax+1; i++) {
                /* All ranks must use the same global checkerboard colouring. */
                int j_start = ((global_i_offset + i + rb) % 2 == 0) ? 2 : 1;

                for (int j = j_start; j < jmax+1; j += 2) {

                    if (flag[i][j] == (C_F | B_NSEW)) {
                        /* five point star for interior fluid cells */
                        p[i][j] = (1.0 - omega) * p[i][j] -
                              beta_2 * ((p[i+1][j] + p[i-1][j] ) *rdx2
                                  + (p[i][j+1] + p[i][j-1]) * rdy2
                                  - rhs[i][j]);
                    } else if (flag[i][j] & C_F) {
                        /* modified star near boundary */

                        double eps_E = ((flag[i+1][j] & C_F) ? 1.0 : 0.0);
                        double eps_W = ((flag[i-1][j] & C_F) ? 1.0 : 0.0);
                        double eps_N = ((flag[i][j+1] & C_F) ? 1.0 : 0.0);
                        double eps_S = ((flag[i][j-1] & C_F) ? 1.0 : 0.0);

                        double beta_mod = -omega / ((eps_E + eps_W) * rdx2 + (eps_N + eps_S) * rdy2);
                        p[i][j] = (1.0 - omega) * p[i][j] -
                            beta_mod * ((eps_E * p[i+1][j] + eps_W * p[i-1][j]) * rdx2
                                + (eps_N * p[i][j+1] + eps_S * p[i][j-1]) * rdy2
                                - rhs[i][j]);
                    }
                }
            }
            halo_exchange(p);
        }

        double res_local = 0.0;
        /* computation of residual */
        for (int i = 1; i < imax+1; i++) {
            for (int j = 1; j < jmax+1; j++) {
                if (flag[i][j] & C_F) {
                    double eps_E = ((flag[i+1][j] & C_F) ? 1.0 : 0.0);
                    double eps_W = ((flag[i-1][j] & C_F) ? 1.0 : 0.0);
                    double eps_N = ((flag[i][j+1] & C_F) ? 1.0 : 0.0);
                    double eps_S = ((flag[i][j-1] & C_F) ? 1.0 : 0.0);

                    /* only fluid cells */
                    double add = (eps_E * (p[i+1][j] - p[i][j]) -
                        eps_W * (p[i][j] - p[i-1][j])) * rdx2  +
                        (eps_N * (p[i][j+1] - p[i][j]) -
                        eps_S * (p[i][j] - p[i][j-1])) * rdy2  -  rhs[i][j];
                    res_local += add * add;
                }
            }
        }

        res = 0.0;
        MPI_Allreduce(&res_local, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        res = sqrt(res / fluid_cells) / p0;

        /* convergence? */
        if (res < eps) break;
    }

    return res;
}


/**
 * @brief Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void update_velocity() {
    /* Preserve the supplied solver's bounds even for one-column partitions. */
    for (int i = 1; i <= imax; i++) {
        if (global_i_offset + i >= global_imax_mpi - 2) continue;
        for (int j = 1; j < jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j] - (p[i+1][j] - p[i][j]) * del_t / delx;
            }
        }
    }

    for (int i = 1; i <= imax; i++) {
        if (global_i_offset + i >= global_imax_mpi - 1) continue;
        for (int j = 1; j < jmax-2; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                v[i][j] = g[i][j] - (p[i][j+1] - p[i][j]) * del_t / dely;
            }
        }
    }
}

/**
 * @brief Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions. Otherwise the simulation becomes unstable.
 */
void set_timestep_interval() {
    /* del_t satisfying CFL conditions */
    if (tau >= 1.0e-10) { /* else no time stepsize control */
        double umax = 1.0e-10;
        double vmax = 1.0e-10;

        for (int i = 0; i < imax+2; i++) {
            for (int j = 1; j < jmax+2; j++) {
                umax = fmax(fabs(u[i][j]), umax);
            }
        }

        for (int i = 1; i < imax+2; i++) {
            for (int j = 0; j < jmax+2; j++) {
                vmax = fmax(fabs(v[i][j]), vmax);
            }
        }

        double deltu = delx / umax;
        double deltv = dely / vmax;
        double deltRe = 1.0 / (1.0 / (delx * delx) + 1 / (dely * dely)) * Re / 2.0;

        if (deltu < deltv) {
            del_t = fmin(deltu, deltRe);
        } else {
            del_t = fmin(deltv, deltRe);
        }
        del_t = tau * del_t; /* multiply by safety factor */

        double global_del_t;
        MPI_Allreduce(&del_t, &global_del_t, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        del_t = global_del_t;
    }
}

/**
 * @brief The main routine that sets up the problem and executes the solving routines routines
 *
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    set_defaults();
    parse_args(argc, argv);
    if (size > imax) {
        if (rank == 0) fprintf(stderr, "Error: MPI ranks must not exceed X cells.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    setup();

    global_imax_mpi = imax;  // Save for VTK output

    local_imax = global_imax_mpi / size;
    int remainder = global_imax_mpi % size;
    if (rank < remainder) {
        local_imax++;
    }

    left_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    right_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    if (rank == 0 && verbose) print_opts();
    imax = local_imax;

    if (rank == 0) {
        printf("Running with %d MPI processes\n", size);
        printf("Global domain: %d x %d\n", global_imax_mpi, jmax);
        printf("Local domain per process: %d x %d (plus 2 halo columns)\n", local_imax, jmax);
        printf("Domain decomposition: X-axis split\n");
    }

    allocate_arrays();
    problem_set_up();


    double res = 0.0;

    /* Main loop */
    int iters = 0;
    double t;
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        if (!fixed_dt)
            set_timestep_interval();

        rdx = 1.0 / delx;
        rdy = 1.0 / dely;
        rdt = 1.0 / del_t;

        compute_tentative_velocity();

        compute_rhs();

        res = poisson();

        update_velocity();

        halo_exchange(u);
        halo_exchange(v);

        apply_boundary_conditions();
        /* Boundary updates also change values shared across partitions. */
        halo_exchange(u);
        halo_exchange(v);

        if ((iters % output_freq == 0)) {
            if (rank == 0) {
                printf("Step %8d, Time: %14.8e (del_t: %14.8e), Residual: %14.8e\n", iters, t+del_t, del_t, res);
            }

            if ((!no_output) && (enable_checkpoints) &&
                write_checkpoint(iters, t+del_t) != 0) {
                free_arrays();
                MPI_Finalize();
                return EXIT_FAILURE;
            }
        }
    } /* End of main loop */

    if(rank == 0) {
        printf("Step %8d, Time: %14.8e, Residual: %14.8e\n", iters, t, res);
        printf("Simulation complete.\n");
    }

    int output_status = no_output ? 0 : write_result(iters, t);
    free_arrays();
    MPI_Finalize();

    return output_status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
