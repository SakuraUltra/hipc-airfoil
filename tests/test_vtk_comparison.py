"""Negative controls for the numerical comparator, independent of solver output."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_vtk import VTKError, compare_grids, read_grid


def fixture(path, *, pressure=None, velocity=None, location="PointData", time="0.005", cycle="5"):
    root = ET.Element("VTKFile", type="RectilinearGrid")
    grid = ET.SubElement(root, "RectilinearGrid", WholeExtent="0 2 0 1 0 0")
    metadata = ET.SubElement(grid, "FieldData")
    ET.SubElement(metadata, "DataArray", Name="TIME", format="ascii").text = time
    ET.SubElement(metadata, "DataArray", Name="CYCLE", format="ascii").text = cycle
    piece = ET.SubElement(grid, "Piece", Extent="0 2 0 1 0 0")
    coordinates = ET.SubElement(piece, "Coordinates")
    for name, values in (("X", "0 1 2"), ("Y", "0 1"), ("Z", "0")):
        ET.SubElement(coordinates, "DataArray", Name=name, format="ascii").text = values
    points = ET.SubElement(piece, "PointData")
    ET.SubElement(points, "DataArray", Name="uv", NumberOfComponents="3", format="ascii").text = velocity or " ".join(["1", "0", "0"] * 6)
    parent = points if location == "PointData" else ET.SubElement(piece, "CellData")
    ET.SubElement(parent, "DataArray", Name="p", format="ascii").text = pressure or "1 2 3 4 5 6"
    ET.ElementTree(root).write(path)


class CompareVTKTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hipc-vtk-test-")
        self.addCleanup(self.temporary.cleanup)
        self.reference = Path(self.temporary.name) / "reference.vtr"
        self.candidate = Path(self.temporary.name) / "candidate.vtr"
        fixture(self.reference)
        fixture(self.candidate)

    def compare(self, **options):
        return compare_grids(read_grid(self.reference), read_grid(self.candidate), **options)

    def test_identical_fields_have_zero_error(self):
        result = self.compare()
        self.assertTrue(result["passed"])
        self.assertEqual(result["fields"]["uv"]["max_abs_error"], 0)
        self.assertEqual(result["fields"]["p"]["relative_l2_error"], 0)

    def test_numeric_mismatch_fails_and_reports_error(self):
        fixture(self.candidate, pressure="1 2 3 4 5 7")
        result = self.compare()
        self.assertFalse(result["passed"])
        self.assertEqual(result["fields"]["p"]["max_abs_error"], 1)
        self.assertAlmostEqual(result["fields"]["p"]["relative_l2_error"], 1 / (91 ** .5))

    def test_absolute_tolerance_accepts_small_near_zero_error(self):
        fixture(self.reference, pressure="0 0 0 0 0 0")
        fixture(self.candidate, pressure="0 0 0 0 0 0.000000001")
        result = self.compare(atol=1e-8)
        self.assertTrue(result["passed"])
        self.assertIsNone(result["fields"]["p"]["relative_l2_error"])
        json.dumps(result, allow_nan=False)

    def test_relative_tolerance_is_componentwise(self):
        fixture(self.reference, pressure="1000000 0 0 0 0 0")
        fixture(self.candidate, pressure="1000000 0 0 0 0 0.001")
        self.assertFalse(self.compare(atol=1e-8, rtol=1e-3, l2_rtol=1e-3)["passed"])

    def test_relative_l2_tolerance_can_independently_reject(self):
        fixture(self.candidate, pressure="1 2 3 4 5 6.01")
        result = self.compare(atol=0, rtol=.1, l2_rtol=1e-8)
        self.assertTrue(result["fields"]["p"]["componentwise_passed"])
        self.assertFalse(result["fields"]["p"]["l2_passed"])
        self.assertFalse(result["passed"])

    def test_legacy_node_pressure_is_explicitly_warned(self):
        fixture(self.candidate, location="CellData")
        result = self.compare()
        self.assertTrue(result["passed"])
        self.assertTrue(result["warnings"])
        with self.assertRaisesRegex(VTKError, "legacy"):
            read_grid(self.candidate, strict=True)

    def test_nonfinite_field_is_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                fixture(self.candidate, pressure=f"1 2 3 4 5 {value}")
                with self.assertRaisesRegex(VTKError, "NaN or infinity"):
                    read_grid(self.candidate)

    def test_wrong_field_length_is_rejected(self):
        fixture(self.candidate, pressure="1 2 3 4 5")
        with self.assertRaisesRegex(VTKError, "expected 6"):
            read_grid(self.candidate)

    def test_wrong_velocity_components_are_rejected(self):
        tree = ET.parse(self.candidate)
        tree.find(".//PointData/DataArray").set("NumberOfComponents", "2")
        tree.write(self.candidate)
        with self.assertRaises(VTKError):
            read_grid(self.candidate)

    def test_time_and_cycle_match_exactly(self):
        for kwargs in ({"time": "0.005000000000001"}, {"cycle": "6"}):
            with self.subTest(kwargs=kwargs):
                fixture(self.candidate, **kwargs)
                with self.assertRaisesRegex(VTKError, "mismatch"):
                    self.compare()

    def test_invalid_time_or_cycle_metadata_is_rejected(self):
        for kwargs in ({"time": "nan"}, {"time": "-1"}, {"cycle": "1.5"}):
            with self.subTest(kwargs=kwargs):
                fixture(self.candidate, **kwargs)
                with self.assertRaises(VTKError):
                    read_grid(self.candidate)

    def test_piece_extent_mismatch_is_rejected(self):
        tree = ET.parse(self.candidate)
        tree.find(".//Piece").set("Extent", "0 1 0 1 0 0")
        tree.write(self.candidate)
        with self.assertRaisesRegex(VTKError, "extent differs"):
            read_grid(self.candidate)

    def test_coordinates_must_match(self):
        tree = ET.parse(self.candidate)
        tree.find(".//Coordinates/DataArray").text = "0 1 3"
        tree.write(self.candidate)
        with self.assertRaisesRegex(VTKError, "coordinates mismatch"):
            self.compare()

    def test_coordinate_metadata_and_shape_are_checked(self):
        for attribute, value in (("RangeMax", "100"), ("RangeMin", "nan")):
            fixture(self.candidate)
            tree = ET.parse(self.candidate)
            tree.find(".//Coordinates/DataArray").set(attribute, value)
            tree.write(self.candidate)
            with self.assertRaises(VTKError):
                read_grid(self.candidate)
        fixture(self.candidate)
        tree = ET.parse(self.candidate)
        tree.find(".//Coordinates/DataArray").text = "0 1"
        tree.write(self.candidate)
        with self.assertRaisesRegex(VTKError, "coordinate axis 0 length"):
            read_grid(self.candidate)

    def test_missing_duplicate_and_binary_fields_are_rejected(self):
        for modification in ("missing", "duplicate", "binary"):
            fixture(self.candidate)
            tree = ET.parse(self.candidate)
            parent = tree.find(".//PointData")
            if modification == "missing":
                parent.remove(parent.findall("DataArray")[1])
            elif modification == "duplicate":
                ET.SubElement(parent, "DataArray", Name="p").text = "1 2 3 4 5 6"
            else:
                parent.find("DataArray").set("format", "binary")
            tree.write(self.candidate)
            with self.assertRaises(VTKError):
                read_grid(self.candidate)

    def test_negative_tolerance_is_rejected(self):
        with self.assertRaisesRegex(VTKError, "Tolerances"):
            self.compare(atol=-1)

    def test_cli_exit_code_and_json_report(self):
        fixture(self.candidate, pressure="1 2 3 4 5 7")
        report = Path(self.temporary.name) / "report.json"
        run = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "compare_vtk.py"),
                              str(self.reference), str(self.candidate), "--strict", "--json", str(report)],
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertFalse(json.loads(report.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
