#!/usr/bin/env python3
"""Record toolchain configuration, solver source hashes and built executable hashes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess


def version(command: str) -> str:
    try:
        result = subprocess.run(shlex.split(command) + ["--version"], text=True,
                                capture_output=True, timeout=10)
        return (result.stdout or result.stderr).strip()[:4000]
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    for key, default in (("cc", "cc"), ("mpicc", "mpicc"), ("nvcc", "nvcc"),
                         ("cflags", ""), ("omp-cflags", ""), ("omp-ldflags", ""), ("nvccflags", "")):
        parser.add_argument("--" + key, default=default)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_files = [root / "Makefile"]
    for folder in ("common", "serial", "openmp", "mpi", "cuda"):
        source_files.extend(path for path in (root / folder).iterdir() if path.suffix in {".c", ".cu", ".h"})
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(source_files)}
    executable_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(args.build_dir.glob("airfoil_*")) if path.is_file()}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "machine": platform.machine(),
              "declared_configuration": {key: value for key, value in vars(args).items() if key != "build_dir"},
              "toolchains": {key: version(getattr(args, key)) for key in ("cc", "mpicc", "nvcc")},
              "source_sha256": source_hashes, "executable_sha256": executable_hashes,
              "scope": "Configuration reported by make info; use a fresh BUILD_DIR when changing compiler flags."}
    args.build_dir.mkdir(parents=True, exist_ok=True)
    path = args.build_dir / "build-info.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
