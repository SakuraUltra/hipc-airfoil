#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vtk.h"
#include "data.h"
#include "output_path.h"

static char output_basename[1000];

void set_default_base() { set_basename("out/airfoil"); }

void set_basename(char *base) {
    if (!*base || strlen(base) >= sizeof(output_basename)) {
        fprintf(stderr, "Error: output basename must contain 1-999 characters.\n");
        exit(EXIT_FAILURE);
    }
    strcpy(output_basename, base);
}

char *get_basename() { return output_basename; }

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

/* Diagnostic export of raw staggered-grid storage, including ghost layers.
 * Values are not interpolated onto a physically co-located velocity grid. */
int write_vtk(char *filename, int iters, double t) {
    if (hipc_make_parent_directories(filename) != 0) return -1;
    FILE *file = fopen(filename, "w");
    if (!file) { perror(filename); return -1; }

    fprintf(file, "<?xml version=\"1.0\"?>\n");
    fprintf(file, "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(file, "<RectilinearGrid WholeExtent=\"0 %d 0 %d 0 0\">\n", imax + 1, jmax + 1);
    fprintf(file, "<FieldData>\n");
    fprintf(file, "<DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">%.17g</DataArray>\n", t);
    fprintf(file, "<DataArray type=\"Int32\" Name=\"CYCLE\" NumberOfTuples=\"1\" format=\"ascii\">%d</DataArray>\n", iters);
    fprintf(file, "</FieldData>\n");
    fprintf(file, "<Piece Extent=\"0 %d 0 %d 0 0\">\n", imax + 1, jmax + 1);
    fprintf(file, "<Coordinates>\n");
    fprintf(file, "<DataArray type=\"Float64\" Name=\"X\" format=\"ascii\">\n");
    for (int i = 0; i < imax + 2; i++) fprintf(file, "%.17g ", delx * i);
    fprintf(file, "\n</DataArray>\n");
    fprintf(file, "<DataArray type=\"Float64\" Name=\"Y\" format=\"ascii\">\n");
    for (int j = 0; j < jmax + 2; j++) fprintf(file, "%.17g ", dely * j);
    fprintf(file, "\n</DataArray>\n");
    fprintf(file, "<DataArray type=\"Float64\" Name=\"Z\" format=\"ascii\">0</DataArray>\n");
    fprintf(file, "</Coordinates>\n");
    fprintf(file, "<PointData Vectors=\"uv\" Scalars=\"p\">\n");
    fprintf(file, "<DataArray type=\"Float64\" Name=\"uv\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int j = 0; j < jmax + 2; j++)
        for (int i = 0; i < imax + 2; i++)
            fprintf(file, "%.17g %.17g 0\n", u[i][j], v[i][j]);
    fprintf(file, "</DataArray>\n");
    fprintf(file, "<DataArray type=\"Float64\" Name=\"p\" format=\"ascii\">\n");
    for (int j = 0; j < jmax + 2; j++) {
        for (int i = 0; i < imax + 2; i++) fprintf(file, "%.17g ", p[i][j]);
        fprintf(file, "\n");
    }
    fprintf(file, "</DataArray>\n</PointData>\n");
    fprintf(file, "</Piece>\n</RectilinearGrid>\n</VTKFile>\n");
    int failed = ferror(file);
    if (fclose(file) != 0) failed = 1;
    if (failed) { perror(filename); return -1; }
    return 0;
}
