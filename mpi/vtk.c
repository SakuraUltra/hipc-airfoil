#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vtk.h"
#include "data.h"
#include "../common/output_path.h"

extern int rank, size, local_imax, global_imax_mpi;

static char output_basename[1000];

void set_default_base() {
    set_basename("out/airfoil");
}

void set_basename(char *base) {
    if (!*base || strlen(base) >= sizeof(output_basename)) {
        if (rank == 0) fprintf(stderr, "Error: output basename must contain 1-999 characters.\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    strcpy(output_basename, base);
}

char *get_basename() {
    return output_basename;
}

int write_checkpoint(int iters, double t) {
    char filename[1024];
    snprintf(filename, sizeof(filename), "%s-%d.vtr", output_basename, iters);
    return write_vtk(filename, iters, t);
}

int write_result(int iters, double t) {
    char filename[1024];
    snprintf(filename, sizeof(filename), "%s.vtr", output_basename);
    return write_vtk(filename, iters, t);
}

/* Collect the owned columns, including only the two physical ghost columns. */
int write_vtk(char *filename, int iters, double t) {
    double *u_global = NULL, *v_global = NULL, *p_global = NULL;
    int *recvcounts = NULL, *displs = NULL;
    int status = 0;
    const int stride = jmax + 2;
    const size_t total_points = (size_t)(global_imax_mpi + 2) * (size_t)stride;

    if (rank == 0) {
        u_global = malloc(total_points * sizeof(*u_global));
        v_global = malloc(total_points * sizeof(*v_global));
        p_global = malloc(total_points * sizeof(*p_global));
        recvcounts = malloc((size_t)size * sizeof(*recvcounts));
        displs = malloc((size_t)size * sizeof(*displs));
        if (!u_global || !v_global || !p_global || !recvcounts || !displs) {
            fprintf(stderr, "Error: could not allocate gathered output arrays.\n");
            status = -1;
        } else {
            int displacement = 0;
            for (int r = 0; r < size; r++) {
                int columns = global_imax_mpi / size + (r < global_imax_mpi % size);
                columns += (r == 0) + (r == size - 1);
                recvcounts[r] = columns * stride;
                displs[r] = displacement;
                displacement += recvcounts[r];
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0) goto cleanup;

    const int first_column = (rank == 0) ? 0 : 1;
    const int columns = local_imax + (rank == 0) + (rank == size - 1);
    MPI_Gatherv(u[first_column], columns * stride, MPI_DOUBLE,
                u_global, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(v[first_column], columns * stride, MPI_DOUBLE,
                v_global, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(p[first_column], columns * stride, MPI_DOUBLE,
                p_global, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *file = NULL;
        if (hipc_make_parent_directories(filename) != 0) {
            status = -1;
        } else if ((file = fopen(filename, "w")) == NULL) {
            perror(filename);
            status = -1;
        } else {
            fprintf(file, "<?xml version=\"1.0\"?>\n");
            fprintf(file, "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
            fprintf(file, "<RectilinearGrid WholeExtent=\"0 %d 0 %d 0 0\">\n", global_imax_mpi + 1, jmax + 1);
            fprintf(file, "<FieldData>\n");
            fprintf(file, "<DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">%.17g</DataArray>\n", t);
            fprintf(file, "<DataArray type=\"Int32\" Name=\"CYCLE\" NumberOfTuples=\"1\" format=\"ascii\">%d</DataArray>\n", iters);
            fprintf(file, "</FieldData>\n");
            fprintf(file, "<Piece Extent=\"0 %d 0 %d 0 0\">\n", global_imax_mpi + 1, jmax + 1);
            fprintf(file, "<Coordinates>\n");
            /* Raw solver arrays include ghost layers; do not shrink their spacing. */
            fprintf(file, "<DataArray type=\"Float64\" Name=\"X\" format=\"ascii\">\n");
            for (int i = 0; i < global_imax_mpi + 2; i++) fprintf(file, "%.17g ", delx * i);
            fprintf(file, "\n</DataArray>\n");
            fprintf(file, "<DataArray type=\"Float64\" Name=\"Y\" format=\"ascii\">\n");
            for (int j = 0; j < jmax + 2; j++) fprintf(file, "%.17g ", dely * j);
            fprintf(file, "\n</DataArray>\n");
            fprintf(file, "<DataArray type=\"Float64\" Name=\"Z\" format=\"ascii\">0</DataArray>\n");
            fprintf(file, "</Coordinates>\n");
            fprintf(file, "<PointData Vectors=\"uv\" Scalars=\"p\">\n");
            fprintf(file, "<DataArray type=\"Float64\" Name=\"uv\" NumberOfComponents=\"3\" format=\"ascii\">\n");
            for (int j = 0; j < jmax + 2; j++) {
                for (int i = 0; i < global_imax_mpi + 2; i++) {
                    size_t index = (size_t)i * (size_t)stride + (size_t)j;
                    fprintf(file, "%.17g %.17g 0\n", u_global[index], v_global[index]);
                }
            }
            fprintf(file, "</DataArray>\n");
            fprintf(file, "<DataArray type=\"Float64\" Name=\"p\" format=\"ascii\">\n");
            for (int j = 0; j < jmax + 2; j++) {
                for (int i = 0; i < global_imax_mpi + 2; i++) {
                    size_t index = (size_t)i * (size_t)stride + (size_t)j;
                    fprintf(file, "%.17g ", p_global[index]);
                }
                fprintf(file, "\n");
            }
            fprintf(file, "</DataArray>\n</PointData>\n");
            fprintf(file, "</Piece>\n</RectilinearGrid>\n</VTKFile>\n");
            int write_failed = ferror(file);
            if (fclose(file) != 0) write_failed = 1;
            if (write_failed) { perror(filename); status = -1; }
        }
    }
    /* Every rank must agree whether output succeeded before the next collective. */
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
cleanup:
    free(u_global); free(v_global); free(p_global);
    free(recvcounts); free(displs);
    return status;
}
