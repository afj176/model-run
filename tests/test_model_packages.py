from hashlib import sha1, sha256
import json
from pathlib import Path
import tempfile
import time
import unittest

from model_run.application import Application
from model_run.model_install import ModelInstaller
from model_run.model_packages import check_file, classify, package_files

from tests.test_model_install import Response, safetensors

CAUSAL = frozenset({"qwen3", "llama", "gemma3"})


def info(names, *, config=None, tags=(), sizes=None):
    return {"id": "org/model", "tags": list(tags), "config": config or {},
            "siblings": [{"rfilename": n, "size": (sizes or {}).get(n, 10)} for n in names]}


class Verdicts(unittest.TestCase):
    def verdict(self, *args, **kwargs):
        return classify(info(*args, **kwargs), CAUSAL)["verdict"]

    def test_a_transformers_language_model_runs_here(self):
        result = classify(info(["config.json", "model.safetensors", "tokenizer.json"],
                               config={"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}), CAUSAL)
        self.assertEqual((result["verdict"], result["kind"]), ("runs_here", "text"))

    def test_what_does_not_run_here_says_why(self):
        self.assertEqual(self.verdict(["config.json", "model.safetensors"], config={"architectures": ["XForCausalLM"], "model_type": "x"}, tags=["custom_code"]), "inspect")
        self.assertEqual(self.verdict(["config.json", "model.safetensors"], config={"architectures": ["NewForCausalLM"], "model_type": "new"}), "inspect")
        self.assertEqual(self.verdict(["config.json", "pytorch_model.bin"], config={"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}), "inspect")
        self.assertEqual(self.verdict(["config.json", "model.safetensors"], config={"architectures": ["Qwen3TTSForConditionalGeneration"], "model_type": "qwen3_tts"}), "inspect")
        self.assertEqual(self.verdict(["model-Q4_K_M.gguf"]), "comfyui")
        self.assertEqual(self.verdict(["flux1-dev.safetensors"]), "comfyui")

    def test_other_kinds_that_run_here(self):
        self.assertEqual(self.verdict(["config.json", "model.safetensors", "modules.json"], config={"architectures": ["Gemma3TextModel"]}), "runs_here")
        self.assertEqual(self.verdict(["config.json", "model.safetensors"], config={"architectures": ["ViTForImageClassification"]}), "runs_here")
        self.assertEqual(self.verdict(["config.json", "model.safetensors"], config={"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}), "runs_here")
        pipeline = ["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"]
        self.assertEqual(self.verdict(pipeline), "inspect")  # the VAE has only a variant the loader does not read
        self.assertEqual(self.verdict(pipeline + ["vae/diffusion_pytorch_model.safetensors"]), "runs_here")


class Files(unittest.TestCase):
    def test_a_package_takes_its_weights_and_settings_and_nothing_else(self):
        names = ["config.json", "model.safetensors", "model.fp16.safetensors", "consolidated.safetensors", "tokenizer.json", "tokenizer.model",
                 "chat_template.jinja", "merges.txt", "README.md", "pytorch_model.bin", "onnx/model.onnx", "onnx/config.json", "original/x.safetensors",
                 ".gitattributes", "handler.py", "1_Pooling/config.json", "2_Dense/model.safetensors", "2_Dense/config.json", "extra/config.json"]
        found = [f["path"] for f in package_files(info(names))]
        self.assertEqual(sorted(found), sorted(["config.json", "model.safetensors", "tokenizer.json", "tokenizer.model", "chat_template.jinja", "merges.txt",
                                                "1_Pooling/config.json", "2_Dense/model.safetensors", "2_Dense/config.json"]))
        self.assertTrue(found[0].endswith(".safetensors"))
        self.assertEqual(found[-1], "config.json")  # last, so the folder is listed only once complete

    def test_a_pipeline_skips_the_single_file_copy_at_its_top(self):
        names = ["model_index.json", "flux1-dev.safetensors", "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors",
                 "scheduler/scheduler_config.json"]
        found = [f["path"] for f in package_files(info(names))]
        self.assertNotIn("flux1-dev.safetensors", found)
        self.assertEqual(found[-1], "model_index.json")

    def test_hashes_come_from_lfs_or_git(self):
        entry = {"id": "o/m", "siblings": [{"rfilename": "model.safetensors", "size": 1, "lfs": {"sha256": "a" * 64, "size": 99}},
                                           {"rfilename": "config.json", "size": 5, "blobId": "b" * 40}]}
        files = {f["path"]: f for f in package_files(entry)}
        self.assertEqual((files["model.safetensors"]["size_bytes"], files["model.safetensors"]["sha256"]), (99, "a" * 64))
        self.assertEqual((files["config.json"]["sha256"], files["config.json"]["git_oid"]), (None, "b" * 40))

    def test_oddly_named_shards_are_still_weights(self):
        names = ["config.json", "model.safetensors.index.json", "model.safetensors-00001-of-00002.safetensors",
                 "model.safetensors-00002-of-00002.safetensors", "model.fp16-00001-of-00002.safetensors"]
        found = [f["path"] for f in package_files(info(names))]
        self.assertIn("model.safetensors-00002-of-00002.safetensors", found)
        self.assertNotIn("model.fp16-00001-of-00002.safetensors", found)

    def test_no_safetensors_is_no_package(self):
        with self.assertRaises(ValueError):
            package_files(info(["config.json", "pytorch_model.bin"]))

    def test_downloaded_files_are_what_their_names_say(self):
        cases = [("config.json", b'{"a": 1}', True), ("config.json", b"{nope", False), ("merges.txt", b"a b", True),
                 ("merges.txt", b"\xff\xfe\xfa", False), ("tokenizer.model", b"\x0a\x05hello", True),
                 ("tokenizer.model", b"\x80\x04pickle", False), ("model.safetensors", safetensors(), True),
                 ("model.safetensors", b"PK\x03\x04zip", False), ("handler.py", b"print()", False)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x"
            for name, data, good in cases:
                path.write_bytes(data)
                if good:
                    check_file(name, path)
                else:
                    with self.assertRaises(Exception, msg=name):
                        check_file(name, path)


class Hub:
    """Hugging Face with one repository: its model information and its files."""
    def __init__(self, files, *, gated=False):
        self.files, self.gated, self.requests = files, gated, []
        self.listed = dict(files)  # what Hugging Face says; files is what it serves

    def info(self):
        siblings = []
        for name, data in self.listed.items():
            entry = {"rfilename": name, "size": len(data)}
            if name.endswith(".safetensors"):
                entry["lfs"] = {"sha256": sha256(data).hexdigest(), "size": len(data)}
            else:
                entry["blobId"] = sha1(f"blob {len(data)}".encode() + b"\x00" + data).hexdigest()
            siblings.append(entry)
        return {"id": "org/tiny", "sha": "c" * 40, "gated": self.gated, "tags": [],
                "config": {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}, "siblings": siblings}

    def open(self, url, *, method="GET", headers=None, timeout=60, redirects=True):
        self.requests.append((method, url))
        if url.startswith("https://huggingface.co/api/models/"):
            return Response(200, json.dumps(self.info()).encode())
        name = url.split("/resolve/" + "c" * 40 + "/", 1)[1]
        if method == "HEAD":
            return Response(401 if self.gated else 200)
        return Response(200, self.files[name])


class Packages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "packages"
        self.files = {"model.safetensors": safetensors(), "config.json": b'{"model_type": "llama"}',
                      "tokenizer.json": b'{"version": "1.0"}'}

    def installer(self, hub):
        installer = ModelInstaller(lambda: [], transport=hub, packages_dir=self.root,
                                   index={"models": {}, "by_id": {}, "workflows": [], "notes": []})
        self.addCleanup(installer.close)
        return installer

    def wait(self, installer):
        for _ in range(500):
            jobs = installer.status()["jobs"]
            if all(j["status"] not in ("queued", "downloading", "verifying") for j in jobs):
                return jobs
            time.sleep(0.01)
        self.fail("downloads did not finish")

    def test_a_package_plans_then_arrives_whole_and_checked(self):
        installer = self.installer(Hub(self.files))
        plan = installer.package_plan("org/tiny")
        self.assertTrue(plan["ok"], plan["reason"])
        self.assertEqual(plan["download_bytes"], sum(len(d) for d in self.files.values()))
        self.assertFalse(self.root.exists())  # planning downloads nothing
        with self.assertRaises(ValueError):
            installer.package_start("org/tiny", plan["download_bytes"] - 1)
        installer.package_start("org/tiny", plan["download_bytes"])
        jobs = self.wait(installer)
        self.assertEqual([j["status"] for j in jobs], ["installed"] * 3)
        self.assertTrue(all(j["hash_checked"] for j in jobs))
        self.assertEqual(jobs[-1]["filename"], "config.json")
        self.assertEqual({p.name for p in (self.root / "org" / "tiny").iterdir()}, set(self.files))
        self.assertEqual(installer.package_plan("org/tiny")["reason"], "This package is already here")

    def test_a_changed_settings_file_is_refused(self):
        hub = Hub(self.files)
        installer = self.installer(hub)
        plan = installer.package_plan("org/tiny")
        hub.files["tokenizer.json"] = b'{"version": "2.0"}'  # same size, different contents from the listed file
        installer.package_start("org/tiny", plan["download_bytes"])
        jobs = {j["filename"]: j for j in self.wait(installer)}
        self.assertEqual(jobs["tokenizer.json"]["status"], "failed")
        self.assertFalse((self.root / "org" / "tiny" / "tokenizer.json").exists())

    def test_a_gated_package_says_what_to_do(self):
        plan = self.installer(Hub(self.files, gated=True)).package_plan("org/tiny")
        self.assertFalse(plan["ok"])
        self.assertIn("licence", plan["reason"])

    def test_names_cannot_leave_the_packages_folder(self):
        installer = self.installer(Hub(self.files))
        for repo in ("../x", "org/..", "a/b/c", "", None):
            with self.assertRaises(ValueError, msg=repo):
                installer.package_plan(repo)

    def test_the_app_scans_its_packages_but_never_installs_comfy_files_there(self):
        app = Application([], packages_dir=self.root)
        self.addCleanup(app.close)
        self.assertIn({"category": "packages", "path": str(self.root.resolve())}, app.entries)
        self.assertEqual(app.installer.categories(), {})
        self.assertTrue(app.snapshot()["packages"])
        inside = Application([], host_kind="comfyui", packages_dir=self.root)
        self.addCleanup(inside.close)
        self.assertFalse(inside.snapshot()["packages"])


if __name__ == "__main__":
    unittest.main()
