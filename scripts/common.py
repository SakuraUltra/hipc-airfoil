"""Process and metadata helpers shared by the benchmark and regression tools."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time


EXECUTABLES = {
    "serial": "airfoil_serial", "serial-opt": "airfoil_serial_opt",
    "openmp": "airfoil_openmp", "openmp-opt": "airfoil_openmp_opt",
    "mpi": "airfoil_mpi", "cuda": "airfoil_cuda",
}


def run_process(command: list[str], cwd: Path, env: dict[str, str], timeout: float,
                log_base: Path) -> dict:
    """Run with a timeout; terminate the whole process group, including MPI ranks."""
    log_base.parent.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = Path(str(log_base) + ".stdout.txt"), Path(str(log_base) + ".stderr.txt")
    start = time.perf_counter()
    result = {"command": command, "cwd": str(cwd), "started_utc": datetime.now(timezone.utc).isoformat(),
              "stdout": str(stdout_path), "stderr": str(stderr_path), "timeout_seconds": timeout}
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            # Pipes make communicate() wait on descriptor readiness instead of
            # wait(timeout)'s polling sleep, which distorts millisecond timings.
            # Solver logs are intentionally small (benchmark output freq=1e9).
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True)
            try:
                out, err = process.communicate(timeout=timeout)
                result["status"] = "completed"
            except subprocess.TimeoutExpired:
                result["status"] = "timeout"
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    out, err = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    out, err = process.communicate()
            stdout.write(out)
            stderr.write(err)
            result["returncode"] = process.returncode
        except OSError as error:
            result.update(status="launch-error", returncode=None, error=str(error))
    result["wall_seconds"] = time.perf_counter() - start
    return result


def metadata(root: Path, build_dir: Path | None = None) -> dict:
    """Portable host/build metadata; intentionally excludes usernames/hostnames."""
    result = {"platform": platform.platform(), "machine": platform.machine(),
              "processor": platform.processor(), "logical_cpus": os.cpu_count(),
              "python": platform.python_version(), "git_commit": None, "git_dirty": None}
    for key, command in (("git_commit", ["git", "rev-parse", "HEAD"]),
                         ("git_dirty", ["git", "status", "--porcelain"])):
        try:
            run = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=5)
            if run.returncode == 0:
                result[key] = bool(run.stdout.strip()) if key == "git_dirty" else run.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    selected_build = (Path(build_dir) if build_dir is not None else root / "build").resolve()
    build_file = selected_build / "build-info.json"
    result["build_directory"] = str(selected_build)
    if build_file.exists():
        try:
            result["build"] = json.loads(build_file.read_text())
        except (OSError, ValueError):
            result["build_metadata_error"] = f"{build_file} could not be read"
    return result


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
