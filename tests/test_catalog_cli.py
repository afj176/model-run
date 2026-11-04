import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CatalogCLITests(unittest.TestCase):
    def test_cli_fixture(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            directory = Path(tmp) / "vae"
            directory.mkdir()
            (directory / "unknown.bin").write_bytes(b"fixture")
            result = subprocess.run([sys.executable, "-m", "model_run.catalog_cli", "--models-dir", tmp],
                                    capture_output=True, text=True, check=True)
            report = json.loads(result.stdout)
            self.assertEqual(report["artifacts"][0]["status"], "uninspected")
            self.assertEqual(report["artifacts"][0]["origins"][0]["category"], "vae")
