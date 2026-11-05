import tempfile
import unittest
from pathlib import Path
from model_run.comfy_paths import comfy_folder_paths, read_extra_paths, standard_model_paths


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

    def test_comfy_folders_put_shared_ones_before_its_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "install" / "ComfyUI"
            (root / "models" / "model_patches").mkdir(parents=True)
            shared = Path(tmp) / "shared" / "model_patches"
            shared.mkdir(parents=True)
            logs = root.parent / "logs"
            logs.mkdir()
            gone = Path(tmp) / "gone"
            (logs / "comfyui.log").write_text(f"[INFO] Adding extra search path model_patches {shared}\nother line\n[INFO] Adding extra search path vae {gone}\n", encoding="utf-8")
            paths = [e["path"] for e in comfy_folder_paths(root)["entries"] if e["category"] == "model_patches"]
            self.assertEqual(paths, [str(shared), str((root / "models" / "model_patches").resolve())])
            self.assertFalse(any(e["path"] == str(gone) for e in comfy_folder_paths(root)["entries"]))

    def test_comfy_folders_read_its_own_config_and_survive_a_bad_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "extra_model_paths.yaml").write_text("mine:\n  base_path: elsewhere\n  vae: vae\n", encoding="utf-8")
            self.assertEqual([e["category"] for e in comfy_folder_paths(root)["entries"]], ["vae"])
            (root / "extra_model_paths.yaml").write_text("- not a mapping\n", encoding="utf-8")
            self.assertEqual(comfy_folder_paths(root)["entries"], [])
