# One-command demo

```sh
make demo
```

This builds the maintained serial and serial-optimised executables, runs both on
the same NACA 2412 case, compares their full output at each saved time and creates
an offline animation. It does not build or run the OpenMP, MPI or CUDA backends.

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
| `index.html` | Play/pause, time slider, playback speed, velocity/pressure and close-up/full-domain controls |
| `velocity.svg`, `pressure.svg` | Final horizontal velocity and pressure over the full domain |
| `velocity-detail.svg`, `pressure-detail.svg` | Final fields in the airfoil close-up |
| `frames/` | Compact animation images generated from actual saved solver fields |
| `serial.vtr`, `serial-opt.vtr` | Complete final rectilinear-grid fields |
| `serial-*.vtr`, `serial-opt-*.vtr` | Checkpoints, retained for every displayed time |
| `serial.stdout.txt`, `serial.stderr.txt` | Reference process logs |
| `serial-opt.stdout.txt`, `serial-opt.stderr.txt` | Optimised-serial process logs |
| `report.json` | Settings, process outcomes and comparison errors/tolerances |

Each invocation creates its own directory and preserves earlier results. To
share a result page, copy its **whole run directory** so figures and evidence
links continue to work. Reports and logs record local process paths; review
these before publishing a result directory.

## What the check establishes

The default case uses 256×64 cells, a fixed timestep of 0.001 and a requested
simulation end time of 0.4. It saves every 10 iterations plus the final result,
giving 41 actual playback frames for the default case.
Each solver has a 30-second process timeout. `make demo-hd` uses 512×128 cells
and a 60-second timeout per solver; it computes more spatial detail rather than
smoothing or enlarging the coarse solution.

Playback uses the `TIME` and `CYCLE` read from each generated field. The first
checkpoint is after the first update (normally TIME 0.001, CYCLE 0), not an
invented time-zero frame. The slider advances between saved states; there is no
temporal interpolation. Playback speed controls the viewing speed rather than
the solver timestep. Reduced-motion preferences start the viewer paused, and
the final static images remain accessible without JavaScript.

The strict reader rejects malformed or non-finite fields. Validation requires
matching coordinates, extent, time and cycle, then compares every velocity and
pressure value, including ghost and obstacle storage. Both componentwise
absolute/relative error and relative L2 error must pass the recorded tolerances.
Plots show only the interior grid; horizontal velocity and pressure retain their
different staggered-grid positions. Geometry marking is reconstructed from the
solver's setup rules, with a test against actual C-generated flags. Grey marks
solid or excluded boundary samples, whose stored values are not physical fluid
measurements. All stored values still participate in the numerical comparison.
Colour scales remain fixed over time and between the close-up and full view.

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

# More actual spatial samples, with the same playback controls.
make demo-hd

# Build a separate serial pair with a different compiler.
make demo CC=clang BUILD_DIR=build/demo-clang

# After building serial, customise the bounded demonstration.
python3 scripts/demo.py --nx 64 --ny 16 --end-time .05 --dt .001

# Save more intermediate states (at most 64 displayed frames).
python3 scripts/demo.py --frame-stride 8

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
