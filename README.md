# Airfoil CFD: parallel computing on CPUs and GPUs

[![Build and validate](https://github.com/SakuraUltra/hipc-airfoil/actions/workflows/ci.yml/badge.svg)](https://github.com/SakuraUltra/hipc-airfoil/actions/workflows/ci.yml)

A two-dimensional NACA-airfoil CFD application, parallelised using
**OpenMP, MPI and custom CUDA kernels**. This repository focuses on work
decomposition, data dependencies, numerical agreement and reproducible performance
measurement.

The project extends an existing serial finite-difference solver with CPU and GPU
parallel implementations, numerical validation tools and inspectable benchmark
evidence. [Provenance](docs/PROVENANCE.md) distinguishes the upstream serial
reference, archived implementations and maintained code.

![Actual horizontal velocity output from the maintained serial solver](docs/assets/velocity.png)

*Actual maintained-solver output: NACA 2412, 256×64 cells, fixed timestep 0.001,
simulation time 0.200. This is a diagnostic illustration, not physical validation
of the CFD model.*

## Implementations

| Executable | Main implementation |
| --- | --- |
| `build/airfoil_serial` | Corrected serial comparison baseline |
| `build/airfoil_serial_opt` | Precomputed reciprocals and direct checkerboard traversal |
| `build/airfoil_openmp` | Parallel loops with sum/max reductions |
| `build/airfoil_openmp_opt` | Shared parallel regions, collapsed loops and static work sharing |
| `build/airfoil_mpi` | X-slab decomposition, halo exchange and global reductions |
| `build/airfoil_cuda` | Device-resident stencil kernels and two-stage reductions |

All maintained variants retain the same solver stages and inherited velocity
update ranges. See [architecture](docs/ARCHITECTURE.md) for synchronization,
boundary treatment and limitations.

## Build and run

Core requirements: a C11 compiler, GNU Make and **Python 3.10+**. Python validation
and benchmark tools use only the standard library. OpenMP requires a supporting
compiler/runtime; MPI requires an MPI implementation. CUDA builds require a
Linux CUDA toolkit with `nvcc`; executing them requires a compatible NVIDIA GPU.

Ubuntu:

```sh
sudo apt-get install build-essential python3 openmpi-bin libopenmpi-dev
make -j4 cpu mpi
make info
```

macOS with Apple Clang and Homebrew:

```sh
brew install libomp open-mpi
make -j4 cpu mpi
make info
```

The Makefile detects Apple Clang and the Homebrew `libomp` path. Linux defaults
to `-fopenmp`. Override `CC`, `MPICC`, `CFLAGS`, `BUILD_DIR` or OpenMP flags as
needed; use a fresh `BUILD_DIR` when changing toolchain options.

```sh
# A short, inspectable simulation; output directories are created automatically.
./build/airfoil_serial -x 128 -y 32 -t .02 -d .001 -o out/serial
OMP_NUM_THREADS=4 ./build/airfoil_openmp_opt -x 128 -y 32 -t .02 -d .001 -o out/openmp
mpirun -n 4 ./build/airfoil_mpi -x 128 -y 32 -t .02 -d .001 -o out/mpi

# Compile on a CUDA-equipped Linux host; set the actual target capability.
make cuda CUDA_ARCH=75
./build/airfoil_cuda -x 128 -y 32 -t .02 -d .001 -o out/cuda
```

`--help` lists all options. `-d` selects a fixed timestep; omitting it enables
CFL control. `-c -f 2` enables checkpoints; `-n` / `--noio` disables file output.
Choosing a stable fixed timestep remains the caller's responsibility.

## Validate first

```sh
make check
# Require MPI rather than silently skipping an unbuilt executable:
python3 scripts/validate.py --mpi required --cuda skip
# On an NVIDIA GPU, require actual CUDA execution too:
python3 scripts/validate.py --mpi required --cuda required

python3 scripts/compare_vtk.py out/serial.vtr out/mpi.vtr --strict
```

Validation compares **complete finite velocity and pressure arrays**, matching
grid coordinates, simulation time and cycle. It includes checkpoints, odd and
uneven decompositions, single-column MPI slabs, symmetric airfoils, adaptive
timesteps and invalid arguments. Optional backends are visibly skipped when
their executables are absent; an existing executable with a broken runtime fails.
Logs and field comparisons are retained for failed runs.

Local CPU/MPI regression and tool-test results are recorded in
[validation](docs/VALIDATION.md). CI builds/tests CPU and MPI with GCC and Clang,
runs Linux sanitizers, and compiles CUDA separately. **The CUDA CI job has no GPU:
successful compilation does not establish GPU runtime correctness or speedup.**

## Performance evidence

Archived measurements, one run per configuration on a single CPU node:

| 1024×256 workload | 1 worker | 16 workers | Speedup vs own 1-worker run |
| --- | ---: | ---: | ---: |
| OpenMP | 332.667 s | 34.389 s | 9.674× |
| MPI | 417.738 s | 31.535 s | 13.247× |

These measurements belong to the **earlier implementation**. They do
not benchmark the revised repository, share a common serial baseline, establish
multi-node scaling, or include repeated-run uncertainty. The archived CUDA event
time is retained separately and is not used to calculate a CPU/GPU speedup.

Collect fresh repeated full-process wall times with an unchanged workload:

```sh
make info
python3 scripts/benchmark.py --backend serial serial-opt openmp openmp-opt mpi \
  --workers 1 2 4 --nx 256 --ny 64 --end-time .02 --dt .001 \
  --warmups 1 --repeats 3 --output benchmarks/results/scaling-256x64
```

CSV, JSON, toolchain/source hashes and per-run logs keep the measurement scope
inspectable. Short smoke runs verify the harness; meaningful scaling results
need representative workloads and controlled hardware. See
[benchmark methodology](docs/BENCHMARKS.md) before interpreting the numbers.

## Repository map

```text
common/                shared CLI, CPU allocation and diagnostic VTK writer
serial/                maintained serial solvers
openmp/                basic and loop-optimised OpenMP solvers
mpi/                   decomposition, communication and collective output
cuda/                  CUDA kernels and device storage
reference/serial/      archived serial reference (numerical code preserved)
scripts/               validation, comparisons, benchmarks and optional plotting
tests/                 negative controls for the verification tools
benchmarks/historical/ retained legacy logs and derived CSV summaries
docs/                  architecture, validation, provenance and methodology
```

The VTR export intentionally contains raw staggered storage with ghost layers;
it is not a collocated physical-field reconstruction. Prebuilt binaries, large
field dumps, input archives and personal machine paths are omitted.

## Attribution and distribution

The supplied CFD scheme follows Griebel, Dornseifer and Neunhoeffer,
*Numerical Simulation in Fluid Dynamics* (SIAM, 1998). Upstream code retains
its original authorship. No redistribution licence was included in the supplied
archive; this repository makes no new licence grant over that material. Public
release requires confirmation of the applicable upstream code permissions. See [provenance](docs/PROVENANCE.md).
