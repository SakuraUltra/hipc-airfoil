#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <getopt.h>

#include "args.h"
#include "data.h"
#include "vtk.h"

int verbose = 0;
int no_output = 0;
int output_freq = 100;
int enable_checkpoints = 0;
int fixed_dt = 0;

static struct option long_options[] = {
    {"airfoil", required_argument, 0, 'a'},
    {"del-t", required_argument, 0, 'd'},
    {"cellx", required_argument, 0, 'x'},
    {"celly", required_argument, 0, 'y'},
    {"freq", required_argument, 0, 'f'},
    {"endtime", required_argument, 0, 't'},
    {"noio", no_argument, 0, 'n'},
    {"output", required_argument, 0, 'o'},
    {"checkpoint", no_argument, 0, 'c'},
    {"verbose", no_argument, 0, 'v'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
};
#define GETOPTS "a:d:x:y:f:t:no:cvh"

static void finish(int status) {
    exit(status);
}

static void invalid(const char *name, const char *value) {
    fprintf(stderr, "Error: invalid %s: %s\n", name, value);
    finish(EXIT_FAILURE);
}

static int positive_integer(const char *value, const char *name) {
    char *end;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || end == value || *end || parsed <= 0 || parsed > INT_MAX - 2)
        invalid(name, value);
    return (int)parsed;
}

static double positive_real(const char *value, const char *name) {
    char *end;
    errno = 0;
    double parsed = strtod(value, &end);
    if (errno || end == value || *end || !isfinite(parsed) || parsed <= 0.0)
        invalid(name, value);
    return parsed;
}

void print_help(char *progname) {
    fprintf(stderr, "A simple computational fluid dynamics solver for NACA 4-digit airfoils.\n\n");
    fprintf(stderr, "Usage: %s [options]\n", progname);
    fprintf(stderr, "  -a N, --airfoil=N       NACA four-digit airfoil\n");
    fprintf(stderr, "  -x N, --cellx=N         Cells in X-dimension (at least 4)\n");
    fprintf(stderr, "  -y N, --celly=N         Cells in Y-dimension (at least 4)\n");
    fprintf(stderr, "  -t N, --endtime=N       Positive simulation end time\n");
    fprintf(stderr, "  -d N, --del-t=N         Positive fixed timestep size\n");
    fprintf(stderr, "  -f N, --freq=N          Positive output frequency in steps\n");
    fprintf(stderr, "  -n, --noio             Disable file I/O\n");
    fprintf(stderr, "  -o BASE, --output=BASE  Output basename (BASE.vtr)\n");
    fprintf(stderr, "  -c, --checkpoint       Enable BASE-ITERATION.vtr checkpoints\n");
    fprintf(stderr, "  -v, --verbose          Print parameters\n");
    fprintf(stderr, "  -h, --help             Print this message and exit\n");
}

void parse_args(int argc, char *argv[]) {
    int option_index = 0;
    int c;
    opterr = 0;
    while ((c = getopt_long(argc, argv, GETOPTS, long_options, &option_index)) != -1) {
        switch (c) {
            case 'a':
                if (strlen(optarg) != 4 || strspn(optarg, "0123456789") != 4)
                    invalid("NACA airfoil (four digits required)", optarg);
                airfoil = (int)strtol(optarg, NULL, 10);
                if (airfoil / 1000 != 0 && (airfoil / 100) % 10 == 0)
                    invalid("cambered airfoil (camber position must be nonzero)", optarg);
                break;
            case 'x': imax = positive_integer(optarg, "X cells"); break;
            case 'y': jmax = positive_integer(optarg, "Y cells"); break;
            case 't': t_end = positive_real(optarg, "end time"); break;
            case 'd': fixed_dt = 1; del_t = positive_real(optarg, "timestep"); break;
            case 'f': output_freq = positive_integer(optarg, "output frequency"); break;
            case 'n': no_output = 1; break;
            case 'o': set_basename(optarg); break;
            case 'c': enable_checkpoints = 1; break;
            case 'v': verbose = 1; break;
            case 'h': print_help(argv[0]); finish(EXIT_SUCCESS); break;
            default: print_help(argv[0]); finish(EXIT_FAILURE);
        }
    }
    if (optind != argc) invalid("argument", argv[optind]);
    if (imax < 4 || jmax < 4) invalid("grid", "both dimensions must be at least 4");
    /* Keep dimensions within the indexing/count range shared by all backends. */
    if ((long long)(imax + 2) * (jmax + 2) > INT_MAX)
        invalid("grid", "too large for supported indexing");
}

void print_opts() {
    printf("=======================================\n");
    printf("Started with the following options\n");
    printf("=======================================\n");
    printf("  airfoil          = %04d\n", airfoil);
    printf("  cellx            = %14d\n", imax);
    printf("  celly            = %14d\n", jmax);
    printf("  endtime          = %14.12lf\n", t_end);
    printf("  del-t            = %14lf\n", del_t);
    printf("  freq             = %14d\n", output_freq);
    printf("  noio             = %14d\n", no_output);
    printf("  output           = %s\n", get_basename());
    printf("  checkpoint       = %14d\n", enable_checkpoints);
    printf("=======================================\n");
}
