from pathlib import Path
import tempfile
import unittest

from model_run.launch import comfy_python, extra_paths_config, link_node_pack
from model_run.model_sources import recipe_for, text_recipes


class Launcher(unittest.TestCase):
    def test_comfy_runs_on_its_own_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(comfy_python(root))
            python = root / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("")
            self.assertEqual(comfy_python(root), python)

    def test_model_run_is_registered_and_desktop_folders_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            comfy = Path(tmp) / "Install" / "ComfyUI"
            comfy.mkdir(parents=True)
            shared = Path(tmp) / "Shared" / "checkpoints"
            shared.mkdir(parents=True)
            logs = comfy.parent / "logs"
            logs.mkdir()
            (logs / "comfyui.log").write_text(f"[INFO] Adding extra search path checkpoints {shared}\n", encoding="utf-8")
            text = extra_paths_config(comfy, runtime=Path(tmp) / ".runtime")
        import yaml
        config = yaml.safe_load(text)
        self.assertEqual(config["model_run"]["custom_nodes"], "custom_nodes")
        self.assertEqual(Path(config["model_run"]["base_path"]), Path(tmp) / ".runtime")
        self.assertEqual(config["comfy_desktop_shared"]["checkpoints"].strip(), str(shared))

    def test_the_node_pack_is_a_link_to_this_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "checkout"
            checkout.mkdir()
            (checkout / "__init__.py").write_text("")
            try:
                link = link_node_pack(runtime=Path(tmp) / ".runtime", root=checkout)
            except OSError as exc:
                self.skipTest(f"links are not allowed here: {exc}")
            self.assertTrue((link / "__init__.py").is_file())
            self.assertEqual(link_node_pack(runtime=Path(tmp) / ".runtime", root=checkout), link)  # a second start reuses it
            other = Path(tmp) / "other"
            other.mkdir()
            with self.assertRaises(RuntimeError):
                link_node_pack(runtime=Path(tmp) / ".runtime", root=other)


class TextRecipes(unittest.TestCase):
    templates = {
        "chat": {"title": "Gemma Chat", "text_loaders": [("gemma4_e4b.safetensors", "stable_diffusion")], "generates_text": True},
        "image": {"title": "Image", "text_loaders": [("qwen_vl.safetensors", "qwen_image")], "generates_text": False},
        "both": {"title": "Enhanced", "text_loaders": [("qwen_vl.safetensors", "qwen_image"), ("gemma4_e4b.safetensors", "stable_diffusion")],
                 "generates_text": True},
    }

    def test_a_file_runs_the_way_comfys_own_templates_load_it(self):
        recipes = text_recipes(self.templates)
        self.assertEqual(recipes["gemma4_e4b.safetensors"]["mode"], "generate")
        self.assertEqual(recipes["qwen_vl.safetensors"], {"type": "qwen_image", "mode": "encode", "workflow": "Image"})

    def test_a_file_no_template_uses_borrows_from_its_closest_relative(self):
        recipes = text_recipes(self.templates)
        borrowed = recipe_for(recipes, "gemma4_12b_int8.safetensors")
        self.assertEqual((borrowed["type"], borrowed["mode"], borrowed["like"]), ("stable_diffusion", "generate", "gemma4_e4b.safetensors"))
        self.assertIsNone(recipe_for(recipes, "flux2-vae.safetensors"))


if __name__ == "__main__":
    unittest.main()
