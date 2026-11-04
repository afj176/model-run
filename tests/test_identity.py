import hashlib
import tempfile
import unittest
from pathlib import Path
from model_run.identity import fingerprint_files


class IdentityTests(unittest.TestCase):
    def test_hash_and_boundaries(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            (root / "config.json").write_bytes(b"abc")
            result = fingerprint_files(root, ["config.json"])
            self.assertEqual(result["files"][0]["sha256"], hashlib.sha256(b"abc").hexdigest())
            with self.assertRaises(ValueError): fingerprint_files(root, ["config.json", "config.json"])
            with self.assertRaises(ValueError): fingerprint_files(root, [str(root / "config.json")])
