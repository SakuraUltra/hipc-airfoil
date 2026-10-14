# Benchmark methodology

## Two separate evidence sets

`benchmarks/historical/` contains archival logs, original run/build configurations,
profiles and CSV summaries. Its [index](../benchmarks/historical/README.md)
retains all four available CPU scaling campaigns, including the earlier runs
with different elapsed times. They describe earlier archived implementation
code, not the maintained commit. Each campaign remains separate; the two CSV
files must not be pooled into repeated-run statistics.
The OpenMP and MPI one-worker baselines differ. No confidence intervals or
multi-node result can be inferred from these single observations.

The manifest records original archive hashes, retained hashes and explicit
privacy edits. Numerical timing and solver-output values are preserved.
`python3 scripts/check_historical.py` checks the retained files and recalculates
CSV values from their source logs. It never updates hashes or rewrites evidence.
Historical files are an append-only evidence set: preserve existing observations
when adding a new campaign or correcting a derived interpretation.

OpenMP's retained primary launch script includes final VTK output, while the
archived MPI script requests `-n` and pipes stdout through `tee`. Exact compiler
versions, hardware models, binary hashes and original placement were not
recorded. Retained Makefiles describe intended builds and do not prove which
binary generated a historical log. These gaps prevent treating the archived
CPU/GPU runs as one controlled cross-backend benchmark.

The archived CUDA `46290.664 ms` interval begins after initial upload and ends
before final download/output. It includes host-driven convergence work and is
neither whole-program wall time nor a pure kernel-time sum. It is not directly
divided into CPU `real` times.

`benchmarks/results/` is ignored generated evidence for fresh runs. The runner
refuses an existing output prefix, preserves failure logs and stops on a failed
or timed-out process. It never fabricates a result for a missing backend.

## Fresh measurements

1. Build in a fresh directory with the intended compiler and flags. Run `make
   info` with the same overrides; this records toolchain versions and hashes of
   source files and executable binaries.
2. Validate numerical results first. For a new benchmark grid/airfoil/timestep,
   generate matching serial and candidate VTR files and compare them explicitly;
   passing the included small regression grids does not validate every workload.
3. Keep grid dimensions, geometry, timestep, end time, output settings and
   compiler policy identical. Benchmarking uses a fixed timestep and `--noio`.
4. Run one campaign at a time, with an idle host and controlled thread/rank
   placement. Specify OpenMP binding/places when applicable; record affinity,
   physical core count, NUMA layout and scheduler allocation for serious HPC
   measurements.
5. Keep warmups separate, collect repeated observations, and report the median
   plus variation. Increase workload duration until process startup and timers
   are a small fraction of runtime. MPI launcher startup can dominate small runs.
6. Report strong scaling against the same backend's one-worker run; for a
   cross-backend comparison explicitly select one common maintained serial run.
   Do not mix the historical and current implementations or combine runs from
   different hardware to estimate a new “measured” speedup.

For cluster optimisation decisions, run both implementations on the same target
cluster node with the same workload and allocation. Mac correctness and short
timing checks do not determine which implementation is faster on that cluster.
When comparing machines, also match solver version, timestep policy, end time,
worker count and output settings; changing the compiler or runtime must be
reported as part of the configuration. Preserve the historical 1024×256,
2098-step observations alongside any new campaign, even when new results differ.

```sh
python3 scripts/benchmark.py --backend openmp openmp-opt --workers 1 2 4 8 \
  --nx 512 --ny 128 --end-time .1 --dt .001 --repeats 5 --warmups 1 \
  --omp-proc-bind close --omp-places cores --label 'controlled single-node run' \
  --output benchmarks/results/omp-512x128
```

Each JSON stores the actual commands, selected environment, run status, platform,
revision/dirty state, build metadata when available, and individual observations.
CSV contains each warmup and measured subprocess wall time. The summary includes
median/min/max/standard deviation and within-backend speedup/efficiency. Bounds
on worker counts, campaign length and per-run timeouts guard accidental long runs.

## Timing boundaries

The Python runner measures full subprocess elapsed time through `communicate`,
including process/launcher startup, allocation, solver work, communication,
normal stdout handling and teardown. VTK output is disabled. It does not compare
internal CUDA-event timing against CPU process wall time.

The maintained CUDA executable additionally reports:

- **CUDA solve elapsed time:** sum of per-timestep event intervals. Includes
  device execution, transfers and stream idle time during host convergence
  decisions inside the solve. Excludes console/checkpoint writes, initial upload
  and final output. It is not the sum of individual kernel durations.
- **Simulation loop wall time:** host steady-clock interval across all timesteps,
  including requested checkpoint output. Excludes initialisation and final output.

For a fair CPU/GPU end-to-end comparison, use the Python runner's process wall
times for every backend and record the GPU model/driver/toolkit. Revised CUDA
runtime performance remains unmeasured until an NVIDIA device is used.

## Optional illustration

The README's figure is generated from an actual maintained serial run:

```sh
./build/airfoil_serial_opt -x 256 -y 64 -t .2 -d .001 -f 200 -o out/demo
python3 -m venv .venv
.venv/bin/pip install -r requirements-visualisation.txt
.venv/bin/python scripts/plot_field.py out/demo.vtr --airfoil 2412 \
  --output docs/assets/velocity.png
```

Only plotting needs NumPy/Matplotlib. It selects one staggered component and
plots its own coordinate locations; the optional outline follows the archived
geometry. This image is illustrative and is not a CFD accuracy claim.
