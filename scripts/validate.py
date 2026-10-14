#!/usr/bin/env python3
"""Run short real solver regressions and compare full finite output fields.

Build CPU targets first, then run: python3 scripts/validate.py
MPI/CUDA are tested automatically only when their executables exist. A missing
runtime for an existing executable is a failure, not a silently passing skip.
All logs, VTK files and JSON evidence are retained, including failed runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

from common import EXECUTABLES, metadata, run_process, save_json
from compare_vtk import VTKError, compare_grids, read_grid


def require_solver_failure(run: dict, diagnostic: str) -> None:
    """Distinguish deliberate input rejection from a crash/runtime/launcher error."""
    assert run.get("returncode") is not None and run["returncode"] > 0, "input rejection must be a normal nonzero exit, not a signal"
    stderr = Path(run["stderr"]).read_text(errors="replace")
    assert diagnostic in stderr, f"expected solver diagnostic {diagnostic!r} missing; inspect {run['stderr']}"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, default=root / "build")
    parser.add_argument("--backends", nargs="+", choices=["serial", "serial-opt", "openmp", "openmp-opt"],
                        default=["serial", "serial-opt", "openmp", "openmp-opt"])
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--mpi", choices=["auto", "required", "skip"], default="auto")
    parser.add_argument("--cuda", choices=["auto", "required", "skip"], default="auto")
    parser.add_argument("--mpi-workers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--mpirun-arg", action="append", default=[], help="Use = for arguments starting with -")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--l2-rtol", type=float, default=1e-7)
    parser.add_argument("--output-dir", type=Path, help="Fresh evidence directory; default: a retained system temporary directory")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol, args.l2_rtol)):
        parser.error("Tolerances must be finite and nonnegative")
    if any(value < 1 or value > 64 for value in args.workers + args.mpi_workers):
        parser.error("Worker counts must be between 1 and 64")
    # The unoptimised serial solver is always the numerical reference.
    backends = list(dict.fromkeys(["serial"] + args.backends))
    executables = {name: (args.build_dir / filename).resolve() for name, filename in EXECUTABLES.items()}
    for name in backends:
        if not executables[name].is_file() or not os.access(executables[name], os.X_OK):
            parser.error(f"Missing executable {executables[name]}; build CPU targets first")
    launcher = shlex.split(args.mpirun) + args.mpirun_arg
    skips = []
    for name in ("mpi", "cuda"):
        mode = getattr(args, name)
        available = executables[name].is_file() and os.access(executables[name], os.X_OK)
        if mode == "required" and not available:
            parser.error(f"--{name}=required but {executables[name]} is missing")
        if mode != "skip" and available:
            if name == "mpi" and (not launcher or not shutil.which(launcher[0])):
                parser.error(f"MPI executable exists but launcher is unavailable: {args.mpirun}")
            backends.append(name)
        else:
            skips.append({"backend": name, "reason": "explicitly skipped" if mode == "skip" else "executable not built"})
    if args.output_dir:
        output = args.output_dir.resolve()
        if output.exists() and any(output.iterdir()):
            parser.error("--output-dir must be absent or empty to preserve previous evidence")
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = Path(tempfile.mkdtemp(prefix="hipc-validation-"))
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "host": metadata(root, args.build_dir), "scope": "Small-grid numerical and CLI regression, not a performance benchmark or physical validation",
              "evidence_directory": str(output), "skipped": skips, "cases": [], "passed": False}
    for skipped in skips:
        print(f"SKIP {skipped['backend']}: {skipped['reason']}", flush=True)

    def record(name: str, action) -> None:
        case = {"name": name, "passed": False}
        try:
            case.update(action())
            case["passed"] = True
        except (AssertionError, OSError, VTKError, ValueError) as error:
            case["error"] = str(error)
        report["cases"].append(case)
        save_json(output / "report.json", report)
        print(f"{'PASS' if case['passed'] else 'FAIL'} {name}" + (f": {case['error']}" if "error" in case else ""), flush=True)

    def launch(name: str, backend: str, workers: int, options: list[str], expected_success: bool = True) -> tuple[dict, Path]:
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        command = [str(executables[backend])] + options
        if backend == "mpi":
            command = launcher + ["-n", str(workers)] + command
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS=str(workers), OMP_DYNAMIC="FALSE")
        run = run_process(command, directory, env, args.timeout, directory / "process")
        # Save the complete run before assertions so even a timeout retains its command.
        save_json(directory / "run.json", run)
        assert run["status"] == "completed", f"{run['status']}; inspect {directory}"
        assert (run["returncode"] == 0) == expected_success, f"unexpected exit {run['returncode']}; inspect {directory}"
        if not expected_success:
            assert run["returncode"] > 0, f"solver terminated by signal {run['returncode']}; inspect {directory}"
        return run, directory

    def read_outputs(directory: Path, base: str = "field", checkpoints: bool = True) -> dict:
        expected = {f"{base}.vtr"}
        if checkpoints:
            expected.update(f"{base}-{iteration}.vtr" for iteration in (0, 2, 4))
        actual = {path.name for path in directory.glob("*.vtr")}
        assert actual == expected, f"output files {sorted(actual)} != {sorted(expected)} in {directory}"
        grids = {path.name: read_grid(path, strict=True) for path in directory.glob("*.vtr")}
        final = grids[f"{base}.vtr"]
        assert final.cycle == 5 and math.isclose(final.time, .005, rel_tol=0, abs_tol=1e-15), "final TIME/CYCLE incorrect"
        if checkpoints:
            for iteration in (0, 2, 4):
                checkpoint = grids[f"{base}-{iteration}.vtr"]
                assert checkpoint.cycle == iteration, f"checkpoint CYCLE incorrect at {iteration}"
                assert math.isclose(checkpoint.time, (iteration + 1) * .001, rel_tol=0, abs_tol=1e-15), "checkpoint TIME incorrect"
        return grids

    # Both normal and odd/nonuniform global decompositions are covered. With 55
    # columns and 3 ranks, global offsets are 0, 19, 37: red-black parity matters.
    reference_grids: dict[tuple[int, int], dict] = {}
    for nx, ny in ((48, 16), (55, 17)):
        options = ["-x", str(nx), "-y", str(ny), "-t", ".005", "-d", ".001", "-o", "field", "-c", "-f", "2"]
        name = f"serial-reference-{nx}x{ny}"

        def reference_action(name=name, options=options, shape=(nx, ny)):
            run, directory = launch(name, "serial", 1, options)
            grids = read_outputs(directory)
            assert grids["field.vtr"].extent == (0, shape[0] + 1, 0, shape[1] + 1, 0, 0), "output extent incorrect"
            reference_grids[shape] = grids
            return {"run": run, "outputs": list(grids)}

        record(name, reference_action)
        for backend in backends:
            workers_list = sorted(set(args.workers)) if backend.startswith("openmp") else (sorted(set(args.mpi_workers)) if backend == "mpi" else [1])
            for workers in workers_list:
                if backend == "serial":
                    continue
                name = f"{backend}-{workers}-{nx}x{ny}"

                def numerical_action(name=name, backend=backend, workers=workers, options=options, shape=(nx, ny)):
                    assert shape in reference_grids, "serial reference failed; see reference run"
                    run, directory = launch(name, backend, workers, options)
                    grids = read_outputs(directory)
                    comparisons = []
                    for filename, reference in reference_grids[shape].items():
                        comparison = compare_grids(reference, grids[filename], args.atol, args.rtol, args.l2_rtol)
                        comparisons.append(comparison)
                    save_json(directory / "comparisons.json", {"comparisons": comparisons})
                    assert all(item["passed"] for item in comparisons), f"numeric mismatch; inspect {directory / 'comparisons.json'}"
                    return {"run": run, "comparisons": comparisons}

                record(name, numerical_action)

    # A symmetric profile has zero camber/location: the NACA formula must avoid
    # dividing by zero in the camber branch and preserve cross-backend results.
    symmetric_reference = {}
    symmetric_options = ["-x", "48", "-y", "16", "-a", "0012", "-t", ".005", "-d", ".001", "-o", "field", "-c", "-f", "2"]

    def symmetric_reference_action():
        run, directory = launch("serial-symmetric-0012", "serial", 1, symmetric_options)
        symmetric_reference.update(read_outputs(directory))
        return {"run": run}

    record("serial-symmetric-0012", symmetric_reference_action)
    for backend in backends:
        if backend == "serial":
            continue
        name = f"{backend}-symmetric-0012"

        def symmetric_action(name=name, backend=backend):
            assert symmetric_reference, "serial symmetric-airfoil reference unavailable"
            workers = 2 if backend.startswith("openmp") or backend == "mpi" else 1
            run, directory = launch(name, backend, workers, symmetric_options)
            candidate = read_outputs(directory)
            comparisons = [compare_grids(reference, candidate[filename], args.atol, args.rtol, args.l2_rtol)
                           for filename, reference in symmetric_reference.items()]
            save_json(directory / "comparisons.json", {"comparisons": comparisons})
            assert all(item["passed"] for item in comparisons), f"symmetric-profile numeric mismatch; inspect {directory}"
            return {"run": run, "comparisons": comparisons}

        record(name, symmetric_action)

    # Output and argument handling are exercised for each compiled backend.
    for backend in backends:
        base_options = ["-x", "48", "-y", "16", "-t", ".005", "-d", ".001"]
        workers = 1
        name = f"{backend}-default-output"

        def default_action(name=name, backend=backend, base_options=base_options):
            directory = output / name
            (directory / "out").mkdir(parents=True, exist_ok=True)
            run, directory = launch(name, backend, 1, base_options)
            assert not list(directory.glob("*.vtr")), "unexpected file outside default out/"
            grids = read_outputs(directory / "out", base="airfoil", checkpoints=False)
            reference = reference_grids.get((48, 16), {}).get("field.vtr")
            assert reference is not None, "serial reference unavailable"
            comparison = compare_grids(reference, grids["airfoil.vtr"], args.atol, args.rtol, args.l2_rtol)
            assert comparison["passed"], "default output differs from explicit -o output"
            return {"run": run, "comparison": comparison}

        record(name, default_action)

        name = f"{backend}-adaptive-timestep"

        def adaptive_action(name=name, backend=backend):
            # Omit -d deliberately: this also exercises MPI timestep reductions.
            run, directory = launch(name, backend, 2 if backend == "mpi" else 1,
                                    ["-x", "48", "-y", "16", "-t", ".1", "-o", "adaptive"])
            assert {path.name for path in directory.glob("*.vtr")} == {"adaptive.vtr"}, "adaptive output missing"
            grid = read_grid(directory / "adaptive.vtr", strict=True)
            assert .1 <= grid.time < 1.0 and 1 <= grid.cycle < 1000, "adaptive time/cycle invalid"
            return {"run": run, "final_time": grid.time, "cycles": grid.cycle,
                    "scope": "Finite-output/control-flow smoke test; numerical comparisons use fixed dt"}

        record(name, adaptive_action)
        for noio in ("-n", "--noio"):
            name = f"{backend}-noio-{'short' if noio == '-n' else 'long'}"

            def noio_action(name=name, backend=backend, base_options=base_options, noio=noio):
                run, directory = launch(name, backend, 1, base_options + [noio, "-o", "forbidden", "-c", "-f", "1"])
                assert not list(directory.rglob("*.vtr")), "--noio still wrote VTK output"
                return {"run": run}

            record(name, noio_action)

        for suffix, bad_options in (("zero-dt", ["-d", "0"]), ("nan-dt", ["-d", "nan"]),
                                    ("bad-grid", ["-x", "-1"]), ("zero-frequency", ["-f", "0"]),
                                    ("malformed-number", ["-x", "48junk"]), ("unknown-option", ["--not-an-option"])):
            name = f"{backend}-reject-{suffix}"

            def invalid_action(name=name, backend=backend, base_options=base_options, bad_options=bad_options):
                run, directory = launch(name, backend, 1, base_options + ["-n"] + bad_options, expected_success=False)
                require_solver_failure(run, "Usage:" if bad_options == ["--not-an-option"] else "Error:")
                assert not list(directory.rglob("*.vtr")), "invalid input generated output"
                return {"run": run}

            record(name, invalid_action)

        name = f"{backend}-help"

        def help_action(name=name, backend=backend):
            run, directory = launch(name, backend, 1, ["--help"])
            assert not list(directory.rglob("*.vtr")), "--help generated output"
            return {"run": run}

        record(name, help_action)

    if "mpi" in backends:
        name = "mpi-one-column-slabs"

        def narrow_slabs_action():
            options = ["-x", "4", "-y", "8", "-t", ".005", "-d", ".001", "-o", "field", "-c", "-f", "2"]
            reference_run, reference_directory = launch("serial-one-column-reference", "serial", 1, options)
            candidate_run, candidate_directory = launch(name, "mpi", 4, options)
            reference = read_outputs(reference_directory)
            candidate = read_outputs(candidate_directory)
            comparisons = [compare_grids(reference[filename], candidate[filename], args.atol, args.rtol, args.l2_rtol)
                           for filename in reference]
            save_json(candidate_directory / "comparisons.json", {"comparisons": comparisons})
            assert all(item["passed"] for item in comparisons), f"one-column MPI slabs differ; inspect {candidate_directory}"
            return {"reference_run": reference_run, "run": candidate_run, "comparisons": comparisons}

        record(name, narrow_slabs_action)
        name = "mpi-reject-too-many-ranks"

        def too_many_action():
            run, directory = launch(name, "mpi", 5, ["-x", "4", "-y", "8", "-t", ".001", "-d", ".001", "-n"], expected_success=False)
            require_solver_failure(run, "Error: MPI ranks must not exceed X cells")
            return {"run": run}

        record(name, too_many_action)

    report["passed"] = all(case["passed"] for case in report["cases"])
    report["summary"] = {"passed": sum(case["passed"] for case in report["cases"]),
                         "failed": sum(not case["passed"] for case in report["cases"]),
                         "skipped_backends": len(skips)}
    save_json(output / "report.json", report)
    print(f"Evidence: {output / 'report.json'}", flush=True)
    print(f"{report['summary']['passed']} passed; {report['summary']['failed']} failed; {len(skips)} optional backends skipped.")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
