"""Negative CLI checks must not count runtime crashes as correct rejections."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate import require_solver_failure


class FailureDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hipc-diagnostic-test-")
        self.addCleanup(self.temporary.cleanup)
        self.stderr = Path(self.temporary.name) / "stderr.txt"

    def test_normal_solver_rejection_is_accepted(self):
        self.stderr.write_text("Error: invalid timestep: 0\n")
        require_solver_failure({"returncode": 1, "stderr": str(self.stderr)}, "Error:")

    def test_signal_is_not_an_input_rejection(self):
        self.stderr.write_text("Error: runtime failed\n")
        with self.assertRaises(AssertionError):
            require_solver_failure({"returncode": -6, "stderr": str(self.stderr)}, "Error:")

    def test_aborted_runtime_or_launcher_is_not_an_input_rejection(self):
        self.stderr.write_text("AddressSanitizer: CHECK failed in runtime initialization\n")
        with self.assertRaises(AssertionError):
            require_solver_failure({"returncode": 134, "stderr": str(self.stderr)}, "Error:")


if __name__ == "__main__":
    unittest.main()
