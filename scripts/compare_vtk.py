#!/usr/bin/env python3
"""Validate and compare the solver's small ASCII RectilinearGrid VTK files.

Only the Python standard library is required. This deliberately does not claim to
be a general VTK reader: binary/appended data and multiple pieces are rejected.
Legacy archived implementation files put node-sized pressure in CellData; that one known
layout is accepted unless --strict is supplied and is always reported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


class VTKError(ValueError):
    """An invalid or unsupported VTK file, or incompatible comparison."""


@dataclass
class Field:
    location: str
    components: int
    values: list[float]


@dataclass
class Grid:
    path: Path
    extent: tuple[int, ...]
    coordinates: tuple[list[float], ...]
    time: float
    cycle: int
    fields: dict[str, Field]
    warnings: list[str]


def _extent(text: str | None, label: str) -> tuple[int, ...]:
    try:
        result = tuple(int(value) for value in (text or "").split())
    except ValueError as error:
        raise VTKError(f"{label}: extent must contain integers") from error
    if len(result) != 6 or any(result[i] > result[i + 1] for i in (0, 2, 4)):
        raise VTKError(f"{label}: invalid extent {text!r}")
    return result


def _numbers(array: ET.Element, label: str) -> list[float]:
    if array.get("format", "ascii").lower() != "ascii":
        raise VTKError(f"{label}: only ASCII DataArray format is supported")
    try:
        result = [float(value) for value in " ".join(array.itertext()).split()]
    except ValueError as error:
        raise VTKError(f"{label}: invalid numeric value") from error
    if not all(math.isfinite(value) for value in result):
        raise VTKError(f"{label}: NaN or infinity encountered")
    return result


def read_grid(path: str | Path, strict: bool = False) -> Grid:
    """Read one complete grid and reject malformed metadata/array dimensions."""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise VTKError(f"{path}: {error}") from error
    if root.tag != "VTKFile" or root.get("type") != "RectilinearGrid":
        raise VTKError(f"{path}: expected VTKFile type=RectilinearGrid")
    rect = root.find("RectilinearGrid")
    if rect is None:
        raise VTKError(f"{path}: RectilinearGrid missing")
    extent = _extent(rect.get("WholeExtent"), "WholeExtent")
    pieces = rect.findall("Piece")
    if len(pieces) != 1:
        raise VTKError(f"{path}: exactly one Piece is required")
    piece = pieces[0]
    if _extent(piece.get("Extent"), "Piece") != extent:
        raise VTKError(f"{path}: Piece extent differs from WholeExtent")
    dimensions = tuple(extent[i + 1] - extent[i] + 1 for i in (0, 2, 4))
    points = math.prod(dimensions)
    cells = math.prod(max(length - 1, 1) for length in dimensions)

    metadata: dict[str, list[float]] = {}
    for array in rect.findall("FieldData/DataArray"):
        name = array.get("Name", "")
        if name in metadata:
            raise VTKError(f"{path}: duplicate FieldData {name!r}")
        metadata[name] = _numbers(array, f"{path}: {name}")
    if any(len(metadata.get(name, [])) != 1 for name in ("TIME", "CYCLE")):
        raise VTKError(f"{path}: singleton TIME and CYCLE metadata required")
    time, cycle_value = metadata["TIME"][0], metadata["CYCLE"][0]
    if time < 0 or cycle_value < 0 or not cycle_value.is_integer():
        raise VTKError(f"{path}: TIME/CYCLE must be nonnegative; CYCLE integer")

    arrays = piece.findall("Coordinates/DataArray")
    if len(arrays) != 3:
        raise VTKError(f"{path}: exactly three coordinate arrays required")
    coordinates = []
    for axis, (array, length) in enumerate(zip(arrays, dimensions)):
        values = _numbers(array, f"{path}: coordinate axis {axis}")
        if len(values) != length:
            raise VTKError(f"{path}: coordinate axis {axis} length {len(values)} != {length}")
        if any(a >= b for a, b in zip(values, values[1:])):
            raise VTKError(f"{path}: coordinates must increase strictly")
        for attribute, expected in (("RangeMin", values[0]), ("RangeMax", values[-1])):
            if attribute in array.attrib:
                try:
                    declared = float(array.get(attribute, "nan"))
                except ValueError as error:
                    raise VTKError(f"{path}: invalid coordinate {attribute}") from error
                if not math.isfinite(declared) or not math.isclose(declared, expected, rel_tol=1e-10, abs_tol=1e-12):
                    raise VTKError(f"{path}: coordinate {attribute} disagrees with data")
        coordinates.append(values)

    warnings: list[str] = []
    fields: dict[str, Field] = {}
    for location in ("PointData", "CellData"):
        for array in piece.findall(f"{location}/DataArray"):
            name = array.get("Name", "")
            if not name or name in fields:
                raise VTKError(f"{path}: missing or duplicate field name {name!r}")
            try:
                components = int(array.get("NumberOfComponents", "1"))
            except ValueError as error:
                raise VTKError(f"{path}: invalid components for {name}") from error
            if components < 1:
                raise VTKError(f"{path}: nonpositive components for {name}")
            values = _numbers(array, f"{path}: {name}")
            expected = (points if location == "PointData" else cells) * components
            if location == "CellData" and name == "p" and len(values) == points * components:
                if strict:
                    raise VTKError(f"{path}: legacy node-sized p in CellData; use corrected PointData")
                warnings.append("Legacy p is node-sized but labelled CellData; treated as PointData for comparison.")
                location_name = "PointData"
            else:
                location_name = location
                if len(values) != expected:
                    raise VTKError(f"{path}: {name} has {len(values)} values, expected {expected}")
            fields[name] = Field(location_name, components, values)
    if "uv" not in fields or "p" not in fields:
        raise VTKError(f"{path}: required fields uv and p missing")
    if fields["uv"].components != 3 or fields["p"].components != 1:
        raise VTKError(f"{path}: expected 3-component uv and scalar p")
    if strict and any(fields[name].location != "PointData" for name in ("uv", "p")):
        raise VTKError(f"{path}: uv and p must both be PointData")
    return Grid(path, extent, tuple(coordinates), time, int(cycle_value), fields, warnings)


def compare_grids(reference: Grid, candidate: Grid, atol: float = 1e-9,
                  rtol: float = 1e-7, l2_rtol: float = 1e-7) -> dict:
    """Require identical metadata and compare each field, without resampling.

    A field passes if every component satisfies abs(diff)<=atol+rtol*abs(ref)
    and RMS(diff)<=atol+l2_rtol*RMS(ref). Relative L2 is reported unmodified;
    a zero reference with nonzero error is represented as null in JSON.
    """
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol, l2_rtol)):
        raise VTKError("Tolerances must be finite and nonnegative")
    for label in ("extent", "time", "cycle", "coordinates"):
        if getattr(reference, label) != getattr(candidate, label):
            raise VTKError(f"{label} mismatch between {reference.path} and {candidate.path}")
    if set(reference.fields) != set(candidate.fields):
        raise VTKError("Field names differ")
    fields = {}
    for name, ref in reference.fields.items():
        other = candidate.fields[name]
        if (ref.location, ref.components, len(ref.values)) != (other.location, other.components, len(other.values)):
            raise VTKError(f"{name}: location, components, or array length differs")
        differences = [abs(a - b) for a, b in zip(ref.values, other.values)]
        # math.hypot avoids overflow/underflow from naive sums of squares.
        ref_norm = math.hypot(*ref.values)
        error_norm = math.hypot(*differences)
        count = len(differences)
        relative_l2 = error_norm / ref_norm if ref_norm else (0.0 if not error_norm else None)
        components_ok = all(diff <= atol + rtol * abs(value) for diff, value in zip(differences, ref.values))
        l2_ok = error_norm <= atol * math.sqrt(count) + l2_rtol * ref_norm
        fields[name] = {
            "max_abs_error": max(differences, default=0.0),
            "relative_l2_error": relative_l2,
            "rms_abs_error": error_norm / math.sqrt(count),
            "max_abs_reference": max(map(abs, ref.values), default=0.0),
            "values": count,
            "componentwise_passed": components_ok,
            "l2_passed": l2_ok,
            "passed": components_ok and l2_ok,
        }
    return {
        "reference": str(reference.path), "candidate": str(candidate.path),
        "extent": list(reference.extent), "time": reference.time, "cycle": reference.cycle,
        "tolerances": {"absolute": atol, "relative": rtol, "relative_l2": l2_rtol},
        "fields": fields, "warnings": reference.warnings + candidate.warnings,
        "passed": all(value["passed"] for value in fields.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?", help="Omit to validate one file only")
    parser.add_argument("--strict", action="store_true", help="Reject legacy pressure in CellData")
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--l2-rtol", type=float, default=1e-7)
    parser.add_argument("--json", type=Path, help="Save a machine-readable report")
    args = parser.parse_args()
    try:
        reference = read_grid(args.reference, strict=args.strict)
        if args.candidate:
            report = compare_grids(reference, read_grid(args.candidate, strict=args.strict),
                                   args.atol, args.rtol, args.l2_rtol)
        else:
            report = {"path": str(reference.path), "extent": reference.extent,
                      "time": reference.time, "cycle": reference.cycle,
                      "fields": list(reference.fields), "warnings": reference.warnings, "passed": True}
    except VTKError as error:
        report = {"passed": False, "error": str(error)}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
