#!/usr/bin/env python3
"""Measure repeated full-process strong scaling on a fixed solver workload.

Example: python3 scripts/benchmark.py --backend openmp --workers 1 2 4 \
    --nx 256 --ny 64 --end-time .02 --dt .001 --output benchmarks/results/openmp

Every timing disables VTK output and uses the same grid, fixed timestep and end
time. Startup, allocation, communication and normal stdout overhead ARE included.
Run scripts/validate.py separately before interpreting performance results.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import sys

from common import EXECUTABLES, metadata, run_process, save_json


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=EXECUTABLES, nargs="+", default=["openmp"],
                        help="Serial/CUDA use one worker; OpenMP/MPI use --workers")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--end-time", type=float, default=.02)
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--airfoil", default="2412")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-runs", type=int, default=128, help="Reject larger accidental campaigns")
    parser.add_argument("--build-dir", type=Path, default=root / "build")
    parser.add_argument("--mpirun", default="mpirun", help="MPI launcher command, e.g. 'srun --mpi=pmix'")
    parser.add_argument("--mpirun-arg", action="append", default=[], help="Extra launcher argument; use = for leading -")
    parser.add_argument("--omp-proc-bind", choices=["false", "true", "close", "spread", "master", "primary"])
    parser.add_argument("--omp-places", help="For example cores or threads; otherwise inherit environment")
    parser.add_argument("--label", default="", help="Human label such as compiler/build configuration")
    parser.add_argument("--output", type=Path, default=root / "benchmarks" / "results" / "scaling",
                        help="Prefix for .csv, .json and .logs/ outputs; existing reports are refused")
    args = parser.parse_args()
    for key in ("end_time", "dt", "timeout"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.nx < 4 or args.ny < 4 or args.repeats < 1 or args.warmups < 0 or args.max_runs < 1:
        parser.error("Grid dimensions must be >=4, repeats/max-runs positive and warmups nonnegative")
    if not args.airfoil.isdigit() or len(args.airfoil) != 4 or args.airfoil[2:] == "00":
        parser.error("--airfoil must be a four-digit NACA code with nonzero thickness")
    if any(worker < 1 or worker > 256 for worker in args.workers):
        parser.error("--workers must be between 1 and 256")
    workers = sorted(set(args.workers))
    backends = list(dict.fromkeys(args.backend))
    configurations = [(backend, worker) for backend in backends
                      for worker in (workers if backend.startswith("openmp") or backend == "mpi" else [1])]
    if len(configurations) * (args.warmups + args.repeats) > args.max_runs:
        parser.error("Campaign exceeds --max-runs; reduce the configuration or explicitly raise the cap")
    if "mpi" in backends and max(workers) > args.nx:
        parser.error("MPI workers cannot exceed --nx")
    executables = {backend: (args.build_dir / EXECUTABLES[backend]).resolve() for backend in backends}
    for backend, path in executables.items():
        if not path.is_file() or not os.access(path, os.X_OK):
            parser.error(f"{backend} executable missing or not executable: {path}")
    launcher = shlex.split(args.mpirun) + args.mpirun_arg
    if "mpi" in backends and (not launcher or not shutil.which(launcher[0])):
        parser.error(f"MPI launcher not found: {args.mpirun}")

    prefix = args.output.resolve()
    json_path, csv_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".csv")
    logs_dir = Path(str(prefix) + ".logs")
    if json_path.exists() or csv_path.exists() or logs_dir.exists():
        parser.error(f"Output already exists for {prefix}; choose a fresh --output")
    logs_dir.mkdir(parents=True)
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "new local full-subprocess wall time; not legacy archived implementation timings",
        "numerical_validation": "Not performed by this tool. Run scripts/validate.py first.",
        "label": args.label, "host": metadata(root, args.build_dir),
        "configuration": {"nx": args.nx, "ny": args.ny, "end_time": args.end_time,
                          "fixed_dt": args.dt, "airfoil": args.airfoil, "vtk_enabled": False,
                          "repeats": args.repeats, "warmups": args.warmups,
                          "timeout_seconds": args.timeout},
        "runs": [], "summary": [], "completed": False,
    }
    if "cuda" in backends and shutil.which("nvidia-smi"):
        import subprocess
        try:
            info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=5)
            report["host"]["gpus"] = info.stdout.strip().splitlines() if info.returncode == 0 else "query failed"
        except (OSError, subprocess.TimeoutExpired):
            report["host"]["gpus"] = "query failed"
    columns = ["backend", "workers", "phase", "repeat", "wall_seconds", "returncode", "status"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for backend, worker in configurations:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(worker if backend.startswith("openmp") else 1)
            env["OMP_DYNAMIC"] = "FALSE"
            if args.omp_proc_bind:
                env["OMP_PROC_BIND"] = args.omp_proc_bind
            if args.omp_places:
                env["OMP_PLACES"] = args.omp_places
            command = [str(executables[backend]), "-x", str(args.nx), "-y", str(args.ny),
                       "-a", args.airfoil, "-t", str(args.end_time), "-d", str(args.dt),
                       "-f", "1000000000", "-n"]
            if backend == "mpi":
                command = launcher + ["-n", str(worker)] + command
            for index in range(args.warmups + args.repeats):
                phase = "warmup" if index < args.warmups else "measured"
                repeat = index + 1 if phase == "warmup" else index - args.warmups + 1
                tag = f"{backend}-{worker}-{phase}-{repeat}"
                result = run_process(command, root, env, args.timeout, logs_dir / tag)
                result.update(backend=backend, workers=worker, phase=phase, repeat=repeat,
                              environment={key: env.get(key) for key in ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES")})
                report["runs"].append(result)
                writer.writerow({key: result[key] for key in columns})
                stream.flush()
                save_json(json_path, report)
                print(f"{tag}: {result['wall_seconds']:.6f} s ({result['status']}, exit {result['returncode']})", flush=True)
                if result["status"] != "completed" or result["returncode"] != 0:
                    print(f"Benchmark stopped; inspect {json_path} and {logs_dir}", file=sys.stderr)
                    return 1
    for backend, worker in configurations:
        values = [run["wall_seconds"] for run in report["runs"]
                  if run["backend"] == backend and run["workers"] == worker and run["phase"] == "measured"]
        report["summary"].append({"backend": backend, "workers": worker, "samples": len(values),
                                  "median_seconds": statistics.median(values),
                                  "min_seconds": min(values), "max_seconds": max(values),
                                  "stdev_seconds": statistics.stdev(values) if len(values) > 1 else None})
    for entry in report["summary"]:
        baseline = next((value for value in report["summary"] if value["backend"] == entry["backend"] and value["workers"] == 1), None)
        entry["speedup_vs_own_one_worker"] = baseline["median_seconds"] / entry["median_seconds"] if baseline else None
        entry["parallel_efficiency"] = entry["speedup_vs_own_one_worker"] / entry["workers"] if baseline else None
    report["completed"] = True
    save_json(json_path, report)
    print(f"Saved {csv_path} and {json_path}. Timings include process startup; short runs are smoke tests, not credible speedup claims.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
