# One-command demo

```sh
make demo
```

This builds the maintained serial and serial-optimised executables, runs both on
the same small NACA 2412 case, compares their full output and creates an offline
result page. It does not build or run the OpenMP, MPI or CUDA backends.

## Prerequisites

- macOS or Linux with a C11 compiler and Make.
- Python 3.10 or newer, using only its standard library.

On macOS, install the Apple command-line developer tools if `cc` or `make` is
missing. On Ubuntu, `sudo apt-get install build-essential python3` supplies the
requirements. No Python package installation is necessary.

If needed, choose the compiler or Python command with `make demo CC=clang
PYTHON=python3`. Use a fresh `BUILD_DIR` after changing compiler options.

## What you get

The terminal prints the absolute path of a new `out/demo/run-*/index.html`.
Open it in a browser directly from your file manager. The page works offline;
all assets and evidence links are local to its run directory.

| File | Contents |
| --- | --- |
| `index.html` | Field viewer, actual final simulation time and cycle, and numerical comparison |
| `velocity.svg`, `pressure.svg` | Horizontal velocity and pressure figures generated from the actual optimised-serial output |
| `serial.vtr`, `serial-opt.vtr` | Complete final rectilinear-grid fields |
| `serial.stdout.txt`, `serial.stderr.txt` | Reference process logs |
| `serial-opt.stdout.txt`, `serial-opt.stderr.txt` | Optimised-serial process logs |
| `report.json` | Settings, process outcomes and comparison errors/tolerances |

Each invocation creates its own directory and preserves earlier results. To
share a result page, copy its **whole run directory** so figures and evidence
links continue to work. Reports and logs record local process paths; review
these before publishing a result directory.

## What the check establishes

The default case uses 128×32 cells, a fixed timestep of 0.001 and a requested
simulation end time of 0.2. Each solver has a 30-second process timeout. The page
shows the `TIME` and `CYCLE` read from the generated fields, rather than inferring
them from filenames.

The strict reader rejects malformed or non-finite fields. Validation requires
matching coordinates, extent, time and cycle, then compares every velocity and
pressure value, including ghost and obstacle storage. Both componentwise
absolute/relative error and relative L2 error must pass the recorded tolerances.
Plots show only the interior grid; horizontal velocity and pressure retain their
different staggered-grid positions.

An unsuccessful process, timeout, invalid output, failed comparison or rendering
error exits with failure, keeps the report and available logs, and leaves no
completed `index.html` in that run directory.

Numerical agreement between related implementations is a regression check. This
short transient is not a converged-flow study or independent physical validation.
The demo calculates no speedup. Historical performance measurements remain in the
[separate evidence archive](../benchmarks/historical/README.md).

## Options and troubleshooting

```sh
# Put generated runs somewhere else; this directory is not removed by make clean.
make demo DEMO_DIR=out/my-demo

# Build a separate serial pair with a different compiler.
make demo CC=clang BUILD_DIR=build/demo-clang

# After building serial, customise the bounded demonstration.
python3 scripts/demo.py --nx 64 --ny 16 --end-time .05 --dt .001

# Inspect all runner options.
python3 scripts/demo.py --help
```

The runner limits grid size and timestep count to keep this a small demonstration.
Changing the fixed timestep can affect numerical stability; a finite result is
not evidence that a chosen timestep is physically appropriate. For larger cases
or backend performance work, use the solver and
[benchmark harness](BENCHMARKS.md) directly.

If the compiler is missing, install the platform tools above. If the runner
fails, open the printed `report.json` and the corresponding stderr/stdout files.
The timeout is configurable with `--timeout` up to 120 seconds per solver for
slower machines. A failed run does not delete earlier successful pages.
