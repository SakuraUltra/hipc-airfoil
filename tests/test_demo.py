"""Failure-path evidence and run isolation for the offline demonstration."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import demo
from common import run_process


# A tiny independent fixture: 4x4 requested cells, 6x6 ghost-inclusive points,
# and exactly one timestep. No CFD executable or plotting dependency is needed.
FAKE_SOLVER = '''from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

mode = __MODE__
print("fixture solver started", flush=True)
print("fixture solver diagnostic", file=sys.stderr, flush=True)
if mode == "timeout":
    time.sleep(10)
base = sys.argv[sys.argv.index("-o") + 1]
path = Path(base + ".vtr")
if mode == "malformed":
    path.write_text("<VTKFile><broken>")
    sys.exit(0)
root = ET.Element("VTKFile", type="RectilinearGrid")
grid = ET.SubElement(root, "RectilinearGrid", WholeExtent="0 5 0 5 0 0")
metadata = ET.SubElement(grid, "FieldData")
ET.SubElement(metadata, "DataArray", Name="TIME", format="ascii").text = "0.001"
ET.SubElement(metadata, "DataArray", Name="CYCLE", format="ascii").text = "1"
piece = ET.SubElement(grid, "Piece", Extent="0 5 0 5 0 0")
coordinates = ET.SubElement(piece, "Coordinates")
for name, values in (("X", "0 1 2 3 4 5"), ("Y", "0 1 2 3 4 5"), ("Z", "0")):
    ET.SubElement(coordinates, "DataArray", Name=name, format="ascii").text = values
fields = ET.SubElement(piece, "PointData")
ET.SubElement(fields, "DataArray", Name="uv", NumberOfComponents="3", format="ascii").text = " ".join(["1", "0", "0"] * 36)
pressure = ["0"] * 36
if mode == "mismatch":
    pressure[-1] = "2"
ET.SubElement(fields, "DataArray", Name="p", format="ascii").text = " ".join(pressure)
ET.ElementTree(root).write(path)
'''


class DemoEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hipc-demo-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.build = self.directory / "build"
        self.build.mkdir()
        self.output = self.directory / "output"
        self.solver("serial", "good")
        self.solver("serial-opt", "good")

    def solver(self, backend, mode):
        path = self.build / demo.EXECUTABLES[backend]
        path.write_text(FAKE_SOLVER.replace("__MODE__", repr(mode)))

    @staticmethod
    def render_page(directory, *_args):
        page = directory / "index.html"
        page.write_text("<!doctype html><title>Verified fixture result</title>")
        return page

    def invoke(self, *, timeout=3, renderer=None):
        module = ModuleType("demo_view")
        module.write_demo_page = Mock(side_effect=renderer or self.render_page)
        arguments = ["demo.py", "--build-dir", str(self.build),
                     "--output-dir", str(self.output), "--nx", "4", "--ny", "4",
                     "--end-time", "0.001", "--dt", "0.001", "--timeout", str(timeout)]

        # Use the active Python interpreter, independent of shebang/PATH details.
        # The real process runner still launches, times out and retains all logs.
        def launch(command, *args):
            return run_process([sys.executable] + command, *args)

        with patch.object(sys, "argv", arguments), patch.object(demo, "run_process", side_effect=launch), \
                patch.dict(sys.modules, {"demo_view": module}), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = demo.main()
        return status, module.write_demo_page

    def retained(self, count):
        directories = sorted(self.output.glob("run-*"))
        self.assertEqual(len(directories), count)
        reports = [(path, json.loads((path / "report.json").read_text())) for path in directories]
        for path, report in reports:
            self.assertTrue(report["runs"])
            for run in report["runs"]:
                self.assertIn("command", run)
                self.assertIn("status", run)
                self.assertIn("returncode", run)
                for channel in ("stdout", "stderr"):
                    log = Path(run[channel])
                    self.assertEqual(log.parent, path)
                    self.assertTrue(log.is_file(), f"Missing retained {channel} log")
        return reports

    def assert_failed(self, stage, run_count):
        path, report = self.retained(1)[0]
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stage"], stage)
        self.assertTrue(report["error"])
        self.assertEqual(len(report["runs"]), run_count)
        self.assertNotIn("page", report)
        self.assertFalse((path / "index.html").exists())
        return path, report

    def test_process_timeout_keeps_failure_report_and_logs(self):
        self.solver("serial", "timeout")
        status, renderer = self.invoke(timeout=.5)
        self.assertNotEqual(status, 0)
        path, report = self.assert_failed("running serial", 1)
        self.assertEqual(report["runs"][0]["status"], "timeout")
        self.assertEqual(report["runs"][0]["timeout_seconds"], .5)
        self.assertIn("timeout", report["error"])
        self.assertIn("fixture solver started", (path / "serial.stdout.txt").read_text())
        self.assertIn("fixture solver diagnostic", (path / "serial.stderr.txt").read_text())
        renderer.assert_not_called()

    def test_full_field_mismatch_keeps_numeric_comparison(self):
        self.solver("serial-opt", "mismatch")
        status, renderer = self.invoke()
        self.assertNotEqual(status, 0)
        path, report = self.assert_failed("validating numerical output", 2)
        self.assertFalse(report["validation"]["passed"])
        self.assertEqual(report["validation"]["fields"]["p"]["max_abs_error"], 2)
        self.assertTrue((path / "serial.vtr").is_file())
        self.assertTrue((path / "serial-opt.vtr").is_file())
        renderer.assert_not_called()

    def test_malformed_solver_output_keeps_raw_evidence(self):
        self.solver("serial-opt", "malformed")
        status, renderer = self.invoke()
        self.assertNotEqual(status, 0)
        path, report = self.assert_failed("validating numerical output", 2)
        self.assertIn("VTKError", report["error"])
        self.assertEqual((path / "serial-opt.vtr").read_text(), "<VTKFile><broken>")
        self.assertTrue(all(run["returncode"] == 0 for run in report["runs"]))
        renderer.assert_not_called()

    def test_renderer_failure_removes_partial_success_page(self):
        def broken_renderer(directory, *_args):
            (directory / "index.html").write_text("incomplete renderer output")
            raise RuntimeError("renderer fixture failed")

        status, renderer = self.invoke(renderer=broken_renderer)
        self.assertNotEqual(status, 0)
        _, report = self.assert_failed("rendering offline page", 2)
        self.assertTrue(report["validation"]["passed"])
        self.assertIn("renderer fixture failed", report["error"])
        renderer.assert_called_once()

    def test_repeated_success_retains_both_runs_without_overwriting(self):
        first_status, first_renderer = self.invoke()
        self.assertEqual(first_status, 0)
        first_path, first_report = self.retained(1)[0]
        self.assertEqual(first_report["status"], "success")
        self.assertEqual(first_report["page"], "index.html")
        snapshot = {path.name: path.read_bytes() for path in first_path.iterdir() if path.is_file()}
        self.assertIn("index.html", snapshot)
        self.assertIn("report.json", snapshot)

        second_status, second_renderer = self.invoke()
        self.assertEqual(second_status, 0)
        reports = self.retained(2)
        self.assertTrue(all(report["status"] == "success" for _, report in reports))
        self.assertTrue(all((path / "index.html").is_file() for path, _ in reports))
        self.assertEqual(snapshot, {path.name: path.read_bytes() for path in first_path.iterdir() if path.is_file()})
        first_renderer.assert_called_once()
        second_renderer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
