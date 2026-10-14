# Source provenance and measurement scope

## Serial reference and parallel implementations

This project builds on an existing serial NACA-airfoil CFD implementation.
The archived README attributes its computational scheme to Griebel,
Dornseifer and Neunhoeffer, *Numerical Simulation in Fluid Dynamics*
(SIAM, 1998). The underlying solver and numerical method are not claimed as
original research or as a solver developed from scratch for this repository.

The OpenMP, MPI and CUDA implementations extend that serial implementation.
`reference/serial/` retains the archived serial source, including `airfoil.c`,
`airfoil_opt.c`, support files and README. The numerical implementation is
preserved; an original support contact is omitted for privacy. The reference
is historical evidence, not the corrected executable comparison baseline.
The source archive, PDFs, executables, object files and large generated VTK
files are not distributed here.

## Maintained implementation

The runnable `serial/`, `openmp/`, `mpi/`, `cuda/` and `common/` code includes
subsequent fixes and improvements:

- Corrected the SOR residual reset in maintained serial code. In the archived
  baseline, `res` is initialised once before the SOR loop, then accumulates onto
  the previous normalised residual in later iterations. Regression comparisons
  use the maintained serial solver.
- Combined independent OpenMP loops into shared parallel regions with loop
  collapse and appropriate `nowait` clauses; retained the barriers between
  pressure colours and residual evaluation.
- Corrected MPI global colour parity, narrow-slab velocity bounds, obstacle
  interface updates, halo synchronization and single-rank/final output.
- Revised CUDA access mapping, reductions and obstacle-boundary ordering.
  Compilation with the CUDA 12.8.1 NVCC toolchain passes in CI; current GPU execution and numerical
  behaviour remain unverified.
- Added consistent builds, argument/allocation checks, valid diagnostic VTK
  output, regression tools, benchmark collection, CI and documentation.

The inherited velocity-update ranges remain unchanged; their exact bounds are
listed in [architecture](ARCHITECTURE.md). Numerical agreement with that solver
is an implementation check, not independent validation of its CFD model.

## Historical performance evidence

The logs in `benchmarks/historical/` come from archived implementations.
They are single observations of earlier code, not measurements of the corrected
current code. CPU logs describe a fixed **1024 × 256** grid and a single-node
16-core allocation.

| Historical series | One worker | 16 workers | Speedup within that series |
| --- | ---: | ---: | ---: |
| OpenMP | 332.667 s | 34.389 s | 9.674× |
| MPI | 417.738 s | 31.535 s | 13.247× |

Each speedup divides its own one-worker elapsed time by its 16-worker elapsed
time. The two one-worker baselines differ; these ratios do not establish a
common serial-baseline comparison. There are no recorded repeated-run
confidence intervals, and the MPI measurements do not demonstrate multi-node
scaling. Historical CSV summaries are derived from the retained logs rather
than new runs.

The archived CUDA log reports `Total GPU time: 46290.664 ms`, approximately
**46.291 s**. That event-timed interval excludes initialisation and final output;
it is not a complete application wall-clock measurement or a sum of kernel
execution times. It cannot establish a directly comparable CPU/GPU speedup.
No new GPU performance result is claimed for the revised source. The maintained
CUDA implementation separately labels per-step event solve time and simulation
loop wall time; their scopes are documented in [benchmarks](BENCHMARKS.md).

Fresh CPU/MPI checks and later measurements retain their own commands,
environment and results. They must not overwrite the historical observations
or imply that those observations benchmark the current commit.

## Distribution and rights

The existing serial implementation retains its original authorship. No upstream
licence granting redistribution was found in the available source files, and
this repository does not grant a new licence over that material. Parallelisation
and subsequent maintenance do not establish ownership of the underlying code.

Public release of material without a verified licence requires confirmation of
the applicable licence or permission from its rights holder. Repository
visibility should remain private until that is established. This provenance
record is not a licensing grant or a claim that permission has already been
obtained.
