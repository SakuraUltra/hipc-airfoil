#!/usr/bin/env python3
"""Verify archival integrity, raw-log summaries and presentation table structure.

This checker only reads evidence. It never writes a manifest, changes a digest,
regenerates CSV data, or merges independently archived scaling series.
Presentation values were transcribed manually: image hashes protect that evidence,
while this script checks structure rather than attempting OCR or proving timing
comparability between independently archived tables.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys


SCALING_ARCHIVES = {
    "strong_scaling.csv": {
        "openmp": "omp_scaling_27579401.out",
        "mpi": "mpi_scaling_27437194.out",
    },
    "additional_scaling.csv": {
        "openmp": "additional/strong_scaling_27404398.out",
        "mpi": "additional/mpi_test_27232003.out",
    },
}
KINDS = {"OpenMP threads": "openmp", "MPI processes": "mpi"}
HEADER = re.compile(r"^--- Running with ([1-9][0-9]*) (OpenMP threads|MPI processes) ---\s*$")
LEGACY_OMP_HEADER = re.compile(r"^\s*测试[:：]\s*([1-9][0-9]*)\s*线程\s*$")
LEGACY_OMP_DOMAIN = re.compile(r"^\s*OMP\s*强扩展性测试\s*\(\s*([1-9][0-9]*)\s*[xX×]\s*([1-9][0-9]*)\s*\)\s*$")
DOMAIN = re.compile(r"^Global domain:\s*([1-9][0-9]*)\s*x\s*([1-9][0-9]*)\s*$")
REAL = re.compile(r"^real[ \t]+([0-9]+)m([0-9]+)\.([0-9]{3})s[ \t]*$")
POSITIVE_INTEGER = re.compile(r"^[1-9][0-9]*$")
RATIO_TOLERANCE = Decimal("1e-12")
CUDA_EVENT = re.compile(r"^Total GPU time:\s*([^\s]+)\s+ms\s*$")
FINAL_STEP = re.compile(r"^Step\s+([0-9]+),\s*Time:\s*([^,\s]+),\s*Residual:\s*[^\s]+\s*$")
PRESENTATION_SUMMARY_IMAGE = "presentation/summary-table.png"
PRESENTATION_SIZE_IMAGE = "presentation/problem-size-table.png"
PROBLEM_SIZES = {
    "Small": (256, 64),
    "Medium": (1024, 256),
    "Large": (2048, 512),
    "ExtraLarge": (4096, 1024),
}
SUMMARY_UNITS = {"serial": "core", "serial_opt": "core", "openmp": "thread",
                 "mpi": "process", "cuda": "gpu"}


class ArchiveError(ValueError):
    """Evidence is missing, malformed, changed, or inconsistent."""


def archive_path(directory: Path, name: str) -> Path:
    """Resolve a canonical relative archive path without escaping the archive."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise ArchiveError("Archive paths must be nonempty relative POSIX paths")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or PureWindowsPath(name).drive or ".." in relative.parts
            or str(relative) != name or str(relative) == "."):
        raise ArchiveError(f"Unsafe or noncanonical archive path: {name!r}")
    resolved = (directory / name).resolve()
    if not resolved.is_relative_to(directory):
        raise ArchiveError(f"Archive path escapes the directory: {name!r}")
    if not resolved.is_file():
        raise ArchiveError(f"Archive file missing or not a regular file: {name!r}")
    return resolved


def verify_manifest(directory: Path) -> set[str]:
    manifest = json.loads(archive_path(directory, "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ArchiveError("manifest.json must have schema_version 1")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ArchiveError("manifest.json must contain a nonempty files list")
    seen_paths: set[Path] = set()
    names: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ArchiveError("Every manifest file entry must be an object")
        name = entry.get("path")
        path = archive_path(directory, name)
        if path in seen_paths:
            raise ArchiveError(f"Duplicate manifest file path: {name!r}")
        seen_paths.add(path)
        names.add(name)
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if type(size) is not int or size < 0:
            raise ArchiveError(f"Invalid manifest byte count: {name!r}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ArchiveError(f"Invalid manifest SHA-256: {name!r}")
        actual_hash = hashlib.sha256()
        actual_size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                actual_hash.update(block)
                actual_size += len(block)
        if actual_size != size:
            raise ArchiveError(f"Byte count mismatch: {name!r} ({actual_size} != {size})")
        if actual_hash.hexdigest() != digest:
            raise ArchiveError(f"SHA-256 mismatch: {name!r}")
    return names


def parse_log(path: Path, expected_backend: str) -> dict[int, tuple[Decimal, int, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    sections = []
    banner_domains = [tuple(map(int, match.groups())) for line in lines
                      if (match := LEGACY_OMP_DOMAIN.fullmatch(line))]
    for index, line in enumerate(lines):
        header = HEADER.fullmatch(line)
        legacy_header = LEGACY_OMP_HEADER.fullmatch(line)
        if header:
            sections.append((index, int(header[1]), KINDS[header[2]], False))
        elif legacy_header:
            if path.name != "strong_scaling_27404398.out" or expected_backend != "openmp" or len(banner_domains) != 1:
                raise ArchiveError(f"Unexpected legacy OpenMP run format in {path.name}")
            sections.append((index, int(legacy_header[1]), "openmp", True))
        elif line.startswith("--- Running with"):
            raise ArchiveError(f"Malformed run header in {path.name}, line {index + 1}")
    if not sections:
        raise ArchiveError(f"No scaling run sections in {path.name}")
    result = {}
    for position, (start, workers, backend, legacy) in enumerate(sections):
        if backend != expected_backend:
            raise ArchiveError(f"{path.name} contains {backend} data, expected {expected_backend}")
        if workers in result:
            raise ArchiveError(f"Duplicate {backend}/{workers} run section in {path.name}")
        stop = sections[position + 1][0] if position + 1 < len(sections) else len(lines)
        domains = []
        timings = []
        for line in lines[start + 1:stop]:
            domain, timing = DOMAIN.fullmatch(line), REAL.fullmatch(line)
            if domain:
                domains.append((int(domain[1]), int(domain[2])))
            if timing:
                minutes, seconds, milliseconds = map(int, timing.groups())
                if seconds >= 60:
                    raise ArchiveError(f"Invalid real timer seconds in {path.name}")
                timings.append(Decimal((minutes * 60 + seconds) * 1000 + milliseconds) / 1000)
            elif line.startswith("real"):
                raise ArchiveError(f"Malformed millisecond real timer in {path.name}")
        if legacy:
            if not domains:
                domains = banner_domains
            elif domains != banner_domains:
                raise ArchiveError(f"{path.name}: section domain differs from its OpenMP banner")
        if len(domains) != 1 or len(timings) != 1 or timings[0] <= 0:
            raise ArchiveError(f"{path.name}: {backend}/{workers} needs exactly one domain and positive real timer")
        result[workers] = (timings[0], *domains[0])
    if len({(nx, ny) for _, nx, ny in result.values()}) != 1:
        raise ArchiveError(f"{path.name}: grid changes within the strong-scaling series")
    return result


def integer(value: str, label: str) -> int:
    if not isinstance(value, str) or not POSITIVE_INTEGER.fullmatch(value):
        raise ArchiveError(f"{label} must be a positive integer")
    return int(value)


def positive_decimal(value: str, label: str) -> Decimal:
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ArchiveError(f"Invalid {label}") from error
    if not result.is_finite() or result <= 0:
        raise ArchiveError(f"{label} must be finite and positive")
    return result


def verify_scaling(directory: Path, manifest_names: set[str], csv_name: str,
                   source_logs: dict[str, str]) -> int:
    if csv_name not in manifest_names:
        raise ArchiveError(f"CSV summary is not covered by the manifest: {csv_name}")
    sources = {}
    for backend, name in source_logs.items():
        if name not in manifest_names:
            raise ArchiveError(f"Source log is not covered by the manifest: {name}")
        sources[backend] = parse_log(archive_path(directory, name), backend)
    required = {"backend", "workers", "wall_seconds", "speedup_vs_own_one_worker",
                "parallel_efficiency", "nx", "ny", "samples", "source_log"}
    records = {}
    with archive_path(directory, csv_name).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not required.issubset(columns):
            raise ArchiveError(f"{csv_name} has missing or duplicate required columns")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(key) is None for key in required):
                raise ArchiveError(f"Malformed CSV row at line {line_number}")
            backend = row["backend"]
            if backend not in source_logs:
                raise ArchiveError(f"Unexpected backend at CSV line {line_number}: {backend!r}")
            workers = integer(row["workers"], "workers")
            key = (backend, workers)
            if key in records:
                raise ArchiveError(f"Duplicate CSV backend/worker: {backend}/{workers}")
            if row["source_log"] != source_logs[backend]:
                raise ArchiveError(f"CSV {backend}/{workers}: source_log must name the declared {backend} archive for {csv_name}")
            if workers not in sources[backend]:
                raise ArchiveError(f"CSV {backend}/{workers}: worker count missing from source log")
            wall = positive_decimal(row["wall_seconds"], "wall_seconds")
            expected_wall, nx, ny = sources[backend][workers]
            if wall != expected_wall:
                raise ArchiveError(f"CSV {backend}/{workers}: wall_seconds differs from raw millisecond timer")
            if (integer(row["nx"], "nx"), integer(row["ny"], "ny")) != (nx, ny):
                raise ArchiveError(f"CSV {backend}/{workers}: grid differs from source Global domain")
            if integer(row["samples"], "samples") != 1:
                raise ArchiveError(f"CSV {backend}/{workers}: archived runs have samples=1")
            records[key] = {
                "wall": wall, "speedup": positive_decimal(row["speedup_vs_own_one_worker"], "speedup"),
                "efficiency": positive_decimal(row["parallel_efficiency"], "parallel_efficiency"),
            }
    for backend, sections in sources.items():
        csv_workers = {workers for name, workers in records if name == backend}
        if csv_workers != set(sections) or 1 not in csv_workers:
            raise ArchiveError(f"CSV {backend}: must cover every source log worker and include one-worker baseline")
        baseline = records[(backend, 1)]["wall"]
        with localcontext() as context:
            context.prec = 40
            for workers in csv_workers:
                record = records[(backend, workers)]
                expected_speedup = baseline / record["wall"]
                expected_efficiency = expected_speedup / workers
                if abs(record["speedup"] - expected_speedup) > RATIO_TOLERANCE:
                    raise ArchiveError(f"CSV {backend}/{workers}: speedup does not match its own one-worker baseline")
                if abs(record["efficiency"] - expected_efficiency) > RATIO_TOLERANCE:
                    raise ArchiveError(f"CSV {backend}/{workers}: parallel_efficiency does not match speedup/workers")
    return len(records)



def require_manifest_files(directory: Path, manifest_names: set[str], *names: str) -> None:
    for name in names:
        if name not in manifest_names:
            raise ArchiveError(f"Evidence file is not covered by the manifest: {name}")
        archive_path(directory, name)


def read_csv(directory: Path, name: str, required: set[str]) -> list[dict[str, str]]:
    with archive_path(directory, name).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not required.issubset(columns):
            raise ArchiveError(f"{name} has missing or duplicate required columns")
        rows = list(reader)
    for index, row in enumerate(rows, start=2):
        if None in row or any(row.get(key) is None for key in required):
            raise ArchiveError(f"{name}: malformed CSV row at line {index}")
    return rows


def verify_cuda_timing(directory: Path, manifest_names: set[str]) -> int:
    csv_name, source_log = "cuda_timing.csv", "cuda_steps.txt"
    require_manifest_files(directory, manifest_names, csv_name, source_log)
    rows = read_csv(directory, csv_name, {
        "backend", "event_milliseconds", "event_seconds", "steps", "final_time",
        "timing_scope", "source_log", "evidence",
    })
    if len(rows) != 1:
        raise ArchiveError(f"{csv_name}: expected exactly one raw CUDA observation")
    timings, finals = [], []
    for line in archive_path(directory, source_log).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        timing, final = CUDA_EVENT.fullmatch(line), FINAL_STEP.fullmatch(line)
        if timing:
            timings.append(positive_decimal(timing[1], "raw CUDA event milliseconds"))
        elif line.startswith("Total GPU time"):
            raise ArchiveError(f"{source_log}: malformed CUDA event timer")
        if final:
            finals.append((integer(final[1], "raw CUDA final steps"),
                           positive_decimal(final[2], "raw CUDA final time")))
    if len(timings) != 1 or len(finals) != 1:
        raise ArchiveError(f"{source_log}: expected one CUDA event timer and one final step/time")
    row = rows[0]
    if (row["backend"], row["source_log"], row["timing_scope"]) != (
            "cuda", source_log, "cuda_event_simulation_loop"):
        raise ArchiveError(f"{csv_name}: CUDA backend, source log or timing scope differs")
    milliseconds = positive_decimal(row["event_milliseconds"], "CUDA event milliseconds")
    seconds = positive_decimal(row["event_seconds"], "CUDA event seconds")
    if milliseconds != timings[0] or seconds != milliseconds / 1000:
        raise ArchiveError(f"{csv_name}: event milliseconds/seconds differ from the raw event timer")
    if (integer(row["steps"], "CUDA steps"),
            positive_decimal(row["final_time"], "CUDA final time")) != finals[0]:
        raise ArchiveError(f"{csv_name}: final steps/time differ from the raw log")
    if not row["evidence"].strip():
        raise ArchiveError(f"{csv_name}: evidence description must not be empty")
    return 1


def verify_presentation_summary(directory: Path, manifest_names: set[str]) -> int:
    csv_name = "presentation/summary.csv"
    require_manifest_files(directory, manifest_names, csv_name, PRESENTATION_SUMMARY_IMAGE)
    rows = read_csv(directory, csv_name, {
        "implementation", "workers", "worker_unit", "reported_real_seconds",
        "reported_user_seconds", "reported_speedup", "reported_efficiency_percent",
        "source_slide", "source_image",
    })
    expected = {("serial", 1), ("serial_opt", 1), ("cuda", 1)} | {
        (backend, workers) for backend in ("openmp", "mpi") for workers in (1, 2, 4, 8, 16)
    }
    seen = set()
    for row in rows:
        backend = row["implementation"]
        key = (backend, integer(row["workers"], "presentation summary workers"))
        if key not in expected or key in seen:
            raise ArchiveError(f"{csv_name}: unexpected or duplicate implementation/worker {key}")
        seen.add(key)
        if row["worker_unit"] != SUMMARY_UNITS[backend]:
            raise ArchiveError(f"{csv_name}: {key} has an invalid recorded worker unit")
        for name in ("reported_real_seconds", "reported_user_seconds", "reported_speedup"):
            positive_decimal(row[name], f"{csv_name} {key} {name}")
        efficiency = row["reported_efficiency_percent"]
        if backend == "cuda":
            if efficiency not in ("", "N/A"):
                raise ArchiveError(f"{csv_name}: CUDA efficiency must retain the reported N/A value")
        else:
            positive_decimal(efficiency, f"{csv_name} {key} efficiency")
        if backend == "serial" and (positive_decimal(row["reported_speedup"], "baseline speedup") != 1
                or positive_decimal(efficiency, "baseline efficiency") != 100):
            raise ArchiveError(f"{csv_name}: baseline must report speedup 1 and efficiency 100 percent")
        if integer(row["source_slide"], "summary source slide") != 19 or row["source_image"] != PRESENTATION_SUMMARY_IMAGE:
            raise ArchiveError(f"{csv_name}: source must be slide 19 and its archived summary image")
    if seen != expected:
        raise ArchiveError(f"{csv_name}: must contain all 13 recorded implementation/worker rows")
    # Rounded reported ratios are preserved, not recalculated from other campaigns.
    return len(rows)


def verify_presentation_sizes(directory: Path, manifest_names: set[str]) -> int:
    csv_name = "presentation/problem_sizes.csv"
    require_manifest_files(directory, manifest_names, csv_name, PRESENTATION_SIZE_IMAGE)
    rows = read_csv(directory, csv_name, {
        "size_category", "nx", "ny", "total_cells", "backend", "workers",
        "total_seconds", "reported_time_per_step", "source_slide", "source_image",
    })
    expected = {(category, backend) for category in PROBLEM_SIZES
                for backend in ("openmp", "mpi", "cuda")}
    seen = set()
    for row in rows:
        category, backend = row["size_category"], row["backend"]
        key = (category, backend)
        if key not in expected or key in seen:
            raise ArchiveError(f"{csv_name}: unexpected or duplicate size/backend {key}")
        seen.add(key)
        nx, ny = integer(row["nx"], "size nx"), integer(row["ny"], "size ny")
        if (nx, ny) != PROBLEM_SIZES[category]:
            raise ArchiveError(f"{csv_name}: {category} grid differs from its recorded dimensions")
        if integer(row["total_cells"], "total cells") != nx * ny:
            raise ArchiveError(f"{csv_name}: {key} total_cells does not equal nx * ny")
        if backend == "cuda":
            if row["workers"] != "":
                raise ArchiveError(f"{csv_name}: CUDA worker count was not recorded and must remain empty")
        elif integer(row["workers"], "size workers") != 16:
            raise ArchiveError(f"{csv_name}: CPU rows must record 16 workers")
        for name in ("total_seconds", "reported_time_per_step"):
            positive_decimal(row[name], f"{csv_name} {key} {name}")
        if integer(row["source_slide"], "sizes source slide") != 20 or row["source_image"] != PRESENTATION_SIZE_IMAGE:
            raise ArchiveError(f"{csv_name}: source must be slide 20 and its archived problem-size image")
    if seen != expected:
        raise ArchiveError(f"{csv_name}: must contain all 12 recorded size/backend rows")
    # The table does not establish exact step counts or a shared timing campaign.
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parents[1] / "benchmarks" / "historical",
                        help="Archive directory to check; evidence is never modified")
    args = parser.parse_args()
    try:
        directory = args.directory.resolve()
        names = verify_manifest(directory)
        rows = 0
        for csv_name, sources in SCALING_ARCHIVES.items():
            try:
                rows += verify_scaling(directory, names, csv_name, sources)
            except ArchiveError as error:
                raise ArchiveError(f"{csv_name}: {error}") from error
        cuda_rows = verify_cuda_timing(directory, names)
        summary_rows = verify_presentation_summary(directory, names)
        size_rows = verify_presentation_sizes(directory, names)
    except (ArchiveError, OSError, ValueError, csv.Error, InvalidOperation) as error:
        print(f"FAIL historical evidence: {error}", file=sys.stderr)
        return 1
    print(f"PASS historical evidence: {len(names)} file hashes and sizes; "
          f"{rows} raw CPU rows in 4 independent scaling series; {cuda_rows} raw CUDA row; "
          f"{summary_rows + size_rows} image-transcribed presentation rows "
          f"({summary_rows} summary + {size_rows} problem-size). "
          "Presentation image hashes and table structure checked; no OCR or cross-campaign ratio inference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
