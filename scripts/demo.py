#!/usr/bin/env python3
"""Run a small verified serial demonstration and create an offline result page.

Only a C11-built serial pair and Python's standard library are required. Each
invocation retains a fresh run directory, its real checkpoint/final VTR fields,
process logs and report.json. Every stored frame is checked before playback.
This functional demonstration does not calculate performance
speedups or provide benchmark evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
import tempfile

from common import EXECUTABLES, run_process, save_json
from compare_vtk import compare_grids, read_grid


def fixed_step_budget(end_time: float, dt: float, limit: int = 2000) -> tuple[int, float]:
    """Bound work using the solver's repeated floating-point timestep addition."""
    time = 0.0
    steps = 0
    while time < end_time:
        advanced = time + dt
        if not math.isfinite(advanced) or advanced <= time:
            raise ValueError("The fixed timestep must advance finite simulation time")
        time = advanced
        steps += 1
        if steps > limit:
            raise ValueError(f"This demonstration is limited to {limit} fixed timesteps")
    return steps, time


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, default=root / "build")
    parser.add_argument("--output-dir", type=Path, default=root / "out" / "demo",
                        help="Parent directory; every invocation creates its own run-* directory")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--airfoil", default="2412", help="Four-digit NACA profile")
    parser.add_argument("--end-time", type=float, default=.4)
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--frame-stride", type=int, default=10,
                        help="Save a real checkpoint every N steps; at most 64 distinct playback frames")
    parser.add_argument("--timeout", type=float, default=30,
                        help="Per-process timeout in seconds, positive and at most 120 (default: 30)")
    args = parser.parse_args()
    if not (4 <= args.nx <= 512 and 4 <= args.ny <= 512) or args.nx * args.ny > 65536:
        parser.error("Each grid dimension must be 4..512 and total cells must not exceed 65,536")
    if len(args.airfoil) != 4 or not args.airfoil.isascii() or not args.airfoil.isdigit():
        parser.error("--airfoil must contain four ASCII digits")
    if args.airfoil[0] != "0" and args.airfoil[1] == "0":
        parser.error("A cambered NACA profile requires a nonzero camber position")
    if any(not math.isfinite(value) or value <= 0 for value in (args.end_time, args.dt, args.timeout)):
        parser.error("--end-time, --dt and --timeout must be finite and positive")
    if args.timeout > 120:
        parser.error("--timeout must not exceed 120 seconds for this bounded demonstration")
    if not 1 <= args.frame_stride <= 2147483645:
        parser.error("--frame-stride must be a positive integer no greater than 2,147,483,645")
    try:
        expected_steps, expected_time = fixed_step_budget(args.end_time, args.dt)
    except ValueError as error:
        parser.error(str(error))
    expected_checkpoints = []
    accumulated_time = 0.0
    for iteration in range(expected_steps):
        accumulated_time += args.dt
        if iteration % args.frame_stride == 0:
            expected_checkpoints.append((iteration, accumulated_time))
    duplicate_final = expected_checkpoints[-1][0] == expected_steps - 1
    expected_frames = len(expected_checkpoints) + (0 if duplicate_final else 1)
    if expected_frames > 64:
        parser.error("Playback is limited to 64 distinct frames; increase --frame-stride")

    settings = {"nx": args.nx, "ny": args.ny, "dt": args.dt,
                "end_time": args.end_time, "airfoil": args.airfoil,
                "frame_stride": args.frame_stride}
    try:
        parent = args.output_dir.resolve()
        parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="run-", dir=parent)).resolve()
    except OSError as error:
        print(f"Demo could not create its evidence directory: {error}", file=sys.stderr)
        return 1
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "stage": "initialization", "settings": settings,
        "limits": {"estimated_fixed_steps": expected_steps, "per_process_timeout_seconds": args.timeout,
                   "max_cells": 65536, "max_fixed_steps": 2000,
                   "expected_playback_frames": expected_frames, "max_playback_frames": 64},
        "scope": "Functional numerical demonstration, not a performance benchmark; no speedups are calculated.",
        "runs": [], "validation": None, "frame_validations": [],
    }
    report_path = directory / "report.json"
    stage = "initialization"
    try:
        save_json(report_path, report)
        build = args.build_dir.resolve()
        for backend in ("serial", "serial-opt"):
            stage = f"running {backend}"
            report.update(stage=stage)
            executable = build / EXECUTABLES[backend]
            options = ["-x", str(args.nx), "-y", str(args.ny), "-a", args.airfoil,
                       "-t", str(args.end_time), "-d", str(args.dt),
                       "-c", "-f", str(args.frame_stride), "-o", backend]
            # run_process also records launch errors and retains empty logs when
            # the executable is missing; do not drop that failure evidence.
            run = run_process([str(executable)] + options, directory, os.environ.copy(),
                              args.timeout, directory / backend)
            run.update(backend=backend, portable_command=[executable.name] + options)
            report["runs"].append(run)
            save_json(report_path, report)
            if run["status"] != "completed" or run["returncode"] != 0:
                raise RuntimeError(f"{backend}: {run['status']}, exit {run['returncode']}; see retained stdout/stderr")

        stage = "validating numerical output"
        report.update(stage=stage)
        expected_extent = (0, args.nx + 1, 0, args.ny + 1, 0, 0)

        def validate_pair(reference_name: str, candidate_name: str, cycle: int,
                          time: float, kind: str) -> dict:
            entry = {"kind": kind, "reference_file": reference_name,
                     "candidate_file": candidate_name, "time": None, "cycle": None,
                     "comparison": None, "status": "validating"}
            report["frame_validations"].append(entry)
            save_json(report_path, report)
            try:
                reference = read_grid(directory / reference_name, strict=True)
                candidate = read_grid(directory / candidate_name, strict=True)
                entry.update(time=candidate.time, cycle=candidate.cycle)
                for name, grid in ((reference_name, reference), (candidate_name, candidate)):
                    if grid.extent != expected_extent:
                        raise ValueError(f"{name}: output extent does not match the requested grid")
                    if grid.cycle != cycle or not math.isclose(grid.time, time, rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError(f"{name}: TIME/CYCLE does not match the requested fixed-step simulation")
                comparison = compare_grids(reference, candidate, atol=1e-8, rtol=1e-7, l2_rtol=1e-7)
                entry["comparison"] = comparison
                if kind == "final":
                    # Preserve the existing report contract for final-state checks.
                    report["validation"] = comparison
                if not comparison["passed"]:
                    raise ValueError(f"{candidate_name}: complete serial and serial-opt fields do not match within the recorded tolerances")
                entry["status"] = "passed"
                save_json(report_path, report)
                return {"grid": candidate, "comparison": comparison}
            except Exception as error:
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                save_json(report_path, report)
                raise

        final_frame = validate_pair("serial.vtr", "serial-opt.vtr", expected_steps, expected_time, "final")
        frames = []
        for iteration, checkpoint_time in expected_checkpoints:
            reference_name = f"serial-{iteration}.vtr"
            candidate_name = f"serial-opt-{iteration}.vtr"
            stage = f"validating checkpoint {reference_name} / {candidate_name}"
            report.update(stage=stage)
            frame = validate_pair(reference_name, candidate_name, iteration, checkpoint_time, "checkpoint")
            if frames and frame["grid"].time <= frames[-1]["grid"].time:
                raise ValueError(f"{candidate_name}: checkpoint TIME must increase strictly")
            frames.append(frame)
        if final_frame["grid"].time == frames[-1]["grid"].time:
            # The last checkpoint can describe the same state with its loop
            # index CYCLE; retain final metadata without duplicating playback.
            frames[-1] = final_frame
        elif final_frame["grid"].time > frames[-1]["grid"].time:
            frames.append(final_frame)
        else:
            raise ValueError("Final TIME precedes the last checkpoint")
        if len(frames) != expected_frames:
            raise ValueError("The verified playback frame count differs from the bounded request")
        report["frame_validations"].sort(key=lambda entry: (entry["time"], entry["cycle"]))
        comparison = final_frame["comparison"]
        final_grid = final_frame["grid"]

        stage = "rendering offline page"
        report.update(stage=stage)
        save_json(report_path, report)
        from demo_view import write_demo_page
        page = Path(write_demo_page(directory, final_grid, comparison, report["runs"], settings, frames)).resolve()
        if page != directory / "index.html" or not page.is_file():
            raise ValueError("The renderer did not produce index.html in the current run directory")
        report.update(status="success", stage="complete", page="index.html",
                      actual={"time": final_grid.time, "cycle": final_grid.cycle,
                              "playback_frames": len(frames), "stored_frame_pairs": len(report["frame_validations"])})
        save_json(report_path, report)
    except Exception as error:
        # A partly rendered page must never look like a completed demonstration.
        try:
            (directory / "index.html").unlink(missing_ok=True)
        except OSError:
            pass
        report.update(status="failed", stage=stage, error=f"{type(error).__name__}: {error}")
        try:
            save_json(report_path, report)
        except OSError as report_error:
            print(f"Failure report could not be saved: {report_error}", file=sys.stderr)
        print(f"Demo failed during {stage}: {error}\nEvidence retained: {directory}", file=sys.stderr)
        return 1
    print(f"Verified {len(report['frame_validations'])} complete frame pairs for {len(frames)} real playback frames; this is a functional demonstration, not a benchmark.")
    print(f"Open: {page}")
    print(f"Evidence: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
