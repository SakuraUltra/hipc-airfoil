"""Prevent a selected alternate build from inheriting unrelated provenance."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import metadata


class BuildMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hipc-metadata-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "build").mkdir()
        (self.root / "build" / "build-info.json").write_text(json.dumps({"configuration": "release"}))

    def test_default_build_metadata(self):
        self.assertEqual(metadata(self.root)["build"]["configuration"], "release")

    def test_alternate_build_metadata_is_selected(self):
        alternate = self.root / "build" / "sanitized"
        alternate.mkdir()
        (alternate / "build-info.json").write_text(json.dumps({"configuration": "sanitized"}))
        report = metadata(self.root, alternate)
        self.assertEqual(report["build"]["configuration"], "sanitized")
        self.assertEqual(report["build_directory"], str(alternate.resolve()))

    def test_missing_alternate_does_not_fall_back_to_release(self):
        report = metadata(self.root, self.root / "not-recorded")
        self.assertNotIn("build", report)


if __name__ == "__main__":
    unittest.main()
