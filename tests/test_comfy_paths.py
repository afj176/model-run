import tempfile
import unittest
from pathlib import Path
from model_run.comfy_paths import read_extra_paths, standard_model_paths


class ComfyPathsTests(unittest.TestCase):
    def test_relative_multiline_and_custom_nodes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            (root / "models" / "vae").mkdir(parents=True)
            config = root / "paths.yaml"
            config.write_text("studio:\n  base_path: models\n  is_default: true\n  vae: |\n    vae\n    other\n  custom_nodes: nodes\n")
            result = read_extra_paths(config)
            self.assertEqual(len(result["entries"]), 2)
            self.assertTrue(result["entries"][0]["exists"])
            self.assertFalse(result["entries"][1]["exists"])
            self.assertTrue(result["entries"][0]["default"])

    def test_bad_section(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("x: 1\n")
            with self.assertRaises(ValueError): read_extra_paths(path)

    def test_standard_paths_and_aliases(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            (root / "clip").mkdir()
            result = standard_model_paths(root)
            clip = next(e for e in result["entries"] if e["path"] == str(root / "clip"))
            self.assertEqual(clip["category"], "text_encoders")
            self.assertTrue(clip["exists"])
            self.assertIn("static", result["limitations"][0])
