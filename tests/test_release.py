# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_archive_contains_only_sources_and_verifiable_manifest(self):
        spec = importlib.util.spec_from_file_location("export_release", ROOT / "scripts/export_release.py")
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.tar.gz"
            second = Path(folder) / "second.tar.gz"
            exporter.export(first)
            exporter.export(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with tarfile.open(first) as archive:
                manifest = json.load(archive.extractfile("PixelPilot/RELEASE_MANIFEST.json"))
                names = archive.getnames()
                self.assertNotIn("PixelPilot/data", names)
                for row in manifest:
                    data = archive.extractfile("PixelPilot/" + row["path"]).read()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])
                    self.assertNotIn("__pycache__", row["path"])
                    self.assertNotIn("swanlog", row["path"])
                    self.assertNotIn("/build/", row["path"])
                    self.assertNotIn(".egg-info/", row["path"])


if __name__ == "__main__":
    unittest.main()
