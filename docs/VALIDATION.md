# Validation record

## Maintained-code checks

Local Apple Silicon build: Apple Clang 15.0.0, Homebrew libomp 21.1.8 and
Open MPI 5.0.11, C11, `-O3 -g -Wall -Wextra`, without fast-math.

- **28 verification-tool tests passed.** Deliberately invalid field values,
  shapes, metadata, legacy associations and comparisons must fail. Runtime
  crashes and launcher failures cannot count as successful argument rejection;
  alternate build directories must use their own provenance.
- The demo's five pipeline tests launch real fixture subprocesses and use the
  strict field reader/comparator. They check timeout, corrupted output, a
  mismatch in ghost storage, partial-render cleanup and preservation of earlier
  successful runs. These fixtures test the harness, not CFD physics.
- **84 solver regression cases passed** in the local maintained CPU/MPI suite.
  It compares 100 complete checkpoint/final output pairs with the corrected
  serial solver. Maximum recorded absolute field difference was approximately
  `2.10e-13`, and maximum relative L2 difference approximately `1.21e-14`.
- A further MPI study compared **24** full outputs on `8×8 / 8 ranks`,
  `9×9 / 4 ranks` and `33×17 / 4 ranks`, including adaptive time control.
  Maximum absolute pressure difference was approximately `1.30e-11`.
- **28 serial UBSan cases passed** with `-O1 -g -fsanitize=undefined`.
- The benchmark harness completed **15** short actual subprocess runs, including
  serial, OpenMP and MPI warmups/repeats. These only validate collection and
  logging, not performance gains.

The main suite covers `48×16`, irregular `55×17`, uneven/odd-offset partitions,
one-column MPI slabs, NACA 0012, fixed and adaptive timesteps, single-process MPI
checkpoints, automatic output directories, full-precision VTR metadata, `--noio`,
help and invalid inputs. It rejects excessive rank counts. Default field
tolerances are `atol=1e-8`, `rtol=1e-7`, and RMS/L2 relative tolerance `1e-7`;
metadata and coordinates must match exactly.

Raw local reports remain in ignored `benchmarks/results/`; a compact redacted
summary is committed as `docs/validation-summary.json`. Re-run rather than
treating these small workloads as exhaustive validation.

## Sanitizer environment boundary

The local Apple Clang 15 AddressSanitizer runtime aborts during initialization
on this macOS installation. An independent malloc/free-only C probe reproduced
the same `sanitizer_malloc_mac.inc` failure before application execution. That
run is recorded as unavailable, not a passing memory-safety check. Local UBSan
was run successfully; Linux CI separately requires AddressSanitizer plus UBSan.

## CUDA status

The revised kernels received source and host-translated C++17 checks; extracted
GPU boundary logic matched the C boundary function over 20 randomized CPU cases,
including nonzero inlet values. Those checks do **not** compile device code or
validate GPU synchronization.

The `CUDA compile only (no GPU runtime)` CI job uses CUDA 12.8.1 and C++17 to
compile and link the actual CUDA sources, then checks help without a GPU. Its
live result is visible in Actions. GPU execution and revised GPU speedup remain
unverified, regardless of whether that build is green.

On a CUDA host:

```sh
make cuda CUDA_ARCH=75
python3 scripts/validate.py --cuda required --mpi skip
./build/airfoil_serial -x 33 -y 17 -t .005 -d .001 -o out/cpu-odd
./build/airfoil_cuda -x 33 -y 17 -t .005 -d .001 -o out/gpu-odd
python3 scripts/compare_vtk.py out/cpu-odd.vtr out/gpu-odd.vtr --strict
compute-sanitizer --tool memcheck ./build/airfoil_cuda -x 33 -y 17 -t .005 -d .001 -n
compute-sanitizer --tool initcheck ./build/airfoil_cuda -x 33 -y 17 -t .005 -d .001 -n
compute-sanitizer --tool synccheck ./build/airfoil_cuda -x 33 -y 17 -t .005 -d .001 -n
```

Also test longer runs, other block-edge sizes, multiple airfoils, adaptive timesteps
and more than one GPU architecture. Changed reduction order can alter low bits
and occasionally threshold-adjacent iteration counts.

## Meaning of a pass

A pass establishes numerical agreement on the tested workloads and catches
specific implementation regressions. It does not establish physical correctness,
mesh independence, converged pressure at every step, production suitability,
multi-node scaling or reproducibility of earlier archived implementation timings.
