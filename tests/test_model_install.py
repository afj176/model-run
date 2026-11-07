import email.message
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest import mock

from model_run.model_install import ModelInstaller, Transport, _safe_part
from model_run.model_sources import blocked_reason, guess_folder, huggingface_parts, install_note, load_index, merge_mirrors, parse_size, template_thumbnail

URL = "https://huggingface.co/org/repo/resolve/main/a.safetensors"


def safetensors(size=64):
    header = json.dumps({"t": {"dtype": "U8", "shape": [size], "data_offsets": [0, size]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"x" * size


def gguf(version=3, tensors=1, entries=0, size=64):
    return b"GGUF" + struct.pack("<IQQ", version, tensors, entries) + b"x" * size


class Response(io.BytesIO):
    def __init__(self, code, data=b"", headers=None):
        super().__init__(data)
        self.code = code
        self.headers = email.message.Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value


class FakeHub:
    """Serves one file; records every request so tests can see what was asked for."""
    def __init__(self, data, *, sha=None, head=None, get=None, cut=None):
        self.data, self.sha, self.head, self.get, self.cut = data, sha, head, get, cut
        self.requests = []

    def open(self, url, *, method="GET", headers=None, timeout=60, redirects=True):
        self.requests.append((method, url, dict(headers or {})))
        if method == "HEAD":
            if self.head:
                return self.head
            return Response(302, headers={"X-Linked-Size": str(len(self.data)), "X-Linked-ETag": f'"{self.sha or sha256(self.data).hexdigest()}"', "Location": "https://cdn-lfs.huggingface.co/x"})
        if self.get:
            return self.get
        start = int(headers["Range"][6:-1]) if headers and "Range" in headers else 0
        body = self.data[start:self.cut] if self.cut else self.data[start:]
        return Response(206 if start else 200, body, {"Content-Range": f"bytes {start}-{len(self.data) - 1}/{len(self.data)}"} if start else None)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.vae = self.root / "models" / "vae"
        self.folders = [{"category": "vae", "path": str(self.vae)}]

    def installer(self, hub, **kwargs):
        installer = ModelInstaller(lambda: self.folders, transport=hub, index=kwargs.pop("index", {"models": {}, "by_id": {}, "workflows": [], "notes": []}), **kwargs)
        self.addCleanup(installer.close)
        return installer

    def request(self, size, **extra):
        return {"url": URL, "category": "vae", "filename": "a.safetensors", "confirmed_bytes": size, **extra}

    def wait(self, installer, job_id):
        for _ in range(500):
            job = next(j for j in installer.status()["jobs"] if j["id"] == job_id)
            if job["status"] not in ("queued", "downloading", "verifying"):
                return job
            time.sleep(0.01)
        self.fail("install did not finish")


class Rules(unittest.TestCase):
    def test_only_https_hugging_face_files_count(self):
        self.assertEqual(huggingface_parts(URL), ("org/repo", "main", "a.safetensors"))
        for url in ("http://huggingface.co/o/r/resolve/main/a.safetensors", "https://huggingface.co.evil.com/o/r/resolve/main/a.safetensors",
                    "https://user@huggingface.co/o/r/resolve/main/a.safetensors", "https://huggingface.co:8443/o/r/resolve/main/a.safetensors",
                    "https://huggingface.co/o/r/blob/main/a.safetensors", "https://huggingface.co/o/r/resolve/main/../../a.safetensors", "file:///etc/passwd", None):
            self.assertIsNone(huggingface_parts(url), url)

    def test_formats_that_can_run_code_are_refused(self):
        self.assertIsNone(blocked_reason(URL, "a.safetensors"))
        self.assertIsNone(blocked_reason(URL, "a.sft"))
        for name in ("a.ckpt", "a.pt", "a.pth", "a.bin", "a.pkl", "a.safetensors.exe"):
            self.assertIn("run code", blocked_reason(URL, name))

    def test_gguf_installs_where_comfyui_gguf_reads_it(self):
        self.assertIsNone(blocked_reason(URL, "flux1-dev-Q4_K_S.gguf"))
        self.assertEqual(guess_folder("flux1-dev-Q4_K_S.gguf"), "diffusion_models")
        self.assertEqual(guess_folder("t5-v1_1-xxl-encoder-Q5_K_M.gguf"), "text_encoders")
        self.assertIn("ComfyUI-GGUF", install_note("flux1-dev-Q4_K_S.gguf"))
        self.assertIsNone(install_note("flux1-dev.safetensors"))

    def test_unsafe_names_are_refused(self):
        for name in ("..", ".", "a/b", "a\\b", "c:evil", "CON", "nul.safetensors", "a ", "a.", "x" * 201, "a\x00b", ""):
            with self.assertRaises(ValueError, msg=repr(name)):
                _safe_part(name)
        self.assertEqual(_safe_part("flux1-dev.safetensors"), "flux1-dev.safetensors")

    def test_redirects_may_not_leave_hugging_face(self):
        with self.assertRaisesRegex(ValueError, "left Hugging Face"):
            Transport().open("https://evil.example/x")
        with self.assertRaisesRegex(ValueError, "left Hugging Face"):
            Transport().open("http://huggingface.co/x")

    def test_guessing_and_sizes(self):
        self.assertEqual(guess_folder("split_files/vae/ae.safetensors"), "vae")
        self.assertEqual(guess_folder("my_lora_v2.safetensors"), "loras")
        self.assertEqual(guess_folder("text_encoders/t5xxl_fp16.safetensors"), "text_encoders")
        self.assertEqual(guess_folder("sd_xl_base_1.0.safetensors"), "checkpoints")
        self.assertEqual(parse_size("6.94GB"), int(6.94 * 1024 ** 3))
        self.assertIsNone(parse_size("unknown"))


class Index(unittest.TestCase):
    def setUp(self):
        # Inside ComfyUI's own Python the real template package is importable; these tests read only their folder.
        patcher = mock.patch("model_run.model_sources.importlib.util.find_spec", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_lists_merge_by_url_and_keep_their_workflows(self):
        with tempfile.TemporaryDirectory() as tmp:
            templates = Path(tmp) / ".venv" / "Lib" / "site-packages" / "comfyui_workflow_templates_json" / "templates"
            templates.mkdir(parents=True)
            model = {"name": "a.safetensors", "url": URL, "directory": "vae", "hash": "A" * 64, "hash_type": "SHA256"}
            (templates / "flux.json").write_text(json.dumps({"nodes": [{"properties": {"models": [model]}}]}), encoding="utf-8")
            (templates / "index.json").write_text(json.dumps([{"templates": [{"name": "flux", "title": "Flux", "description": "d"}]}]), encoding="utf-8")
            cache = Path(tmp) / "user" / "__manager" / "cache"
            cache.mkdir(parents=True)
            manager = [{"name": "A", "type": "VAE", "save_path": "default", "filename": "a.safetensors", "url": URL + "?download=true", "size": "1MB", "base": "flux"},
                       {"name": "B", "type": "checkpoint", "save_path": "checkpoints/SD", "filename": "b.ckpt", "url": "https://huggingface.co/o/r/resolve/main/b.ckpt"},
                       {"name": "C", "type": "x", "save_path": "default", "filename": "c.safetensors", "url": "o/r"}]
            (cache / "1_model-list.json").write_text(json.dumps({"models": manager}), encoding="utf-8")
            index = load_index(tmp)
        self.assertEqual(len(index["models"]), 2)
        merged = index["models"][URL]
        self.assertEqual(sorted(merged["sources"]), ["manager", "templates"])
        self.assertEqual(merged["sha256"], "a" * 64)
        self.assertEqual(merged["workflows"], [{"name": "flux", "title": "Flux"}])
        self.assertEqual(merged["size_bytes"], 1024 ** 2)
        ckpt = index["models"]["https://huggingface.co/o/r/resolve/main/b.ckpt"]
        self.assertEqual((ckpt["category"], ckpt["subfolder"]), ("checkpoints", "SD"))
        self.assertIn("run code", ckpt["blocked"])
        self.assertEqual(index["workflows"][0]["models"], [merged["id"]])

    def test_templates_keep_their_group_rank_size_and_thumbnail(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / ".venv" / "Lib" / "site-packages"
            templates = site / "comfyui_workflow_templates_json" / "templates"
            templates.mkdir(parents=True)
            model = {"name": "a.safetensors", "url": URL, "directory": "vae"}
            (templates / "flux.json").write_text(json.dumps({"models": [model]}), encoding="utf-8")
            index = [{"title": "Image", "type": "image", "templates": [{"name": "flux", "title": "Flux", "searchRank": 7, "size": 2048}]},
                     {"title": "Product & Ads", "type": "image", "templates": [{"name": "flux", "title": "Flux again"}]}]
            (templates / "index.json").write_text(json.dumps(index), encoding="utf-8")
            media = site / "comfyui_workflow_templates_media_image" / "templates"
            media.mkdir(parents=True)
            (media / "flux-1.webp").write_bytes(b"RIFF")
            workflow = load_index(tmp)["workflows"][0]
            self.assertEqual((workflow["title"], workflow["kind"], workflow["media"], workflow["rank"], workflow["size_bytes"]),
                             ("Flux", "Image", "image", 7, 2048))
            self.assertEqual(Path(template_thumbnail("flux", tmp)).name, "flux-1.webp")
            for name in ("../flux", "flux/x", "", "-flux", None):
                self.assertIsNone(template_thumbnail(name, tmp), name)

    def test_one_file_hosted_in_several_repositories_is_one_entry(self):
        def item(url, workflows, sha=None, size=None, sources=("templates",), category="vae", filename="flux2-vae.safetensors"):
            return {"id": url.rsplit("?", 1)[1], "name": filename, "filename": filename, "url": url, "category": category, "subfolder": "",
                    "size_bytes": size, "sha256": sha, "description": "", "base": "", "sources": list(sources),
                    "workflows": [{"name": w, "title": w.title()} for w in workflows], "blocked": None, "note": None}
        hf = "https://huggingface.co/"
        items = [item(hf + "a/r/flux2-vae.safetensors?a1", ["one"]),
                 item(hf + "b/r/flux2-vae.safetensors?b22", ["two", "three"]),
                 item(hf + "c/r/flux2-vae.safetensors?c333", ["four"], size=336, sources=("manager",)),
                 item(hf + "d/r/flux2-vae.safetensors?d4444", ["five"], sha="1" * 64),
                 item(hf + "e/r/flux2-vae.safetensors?e55555", ["six"], sha="2" * 64),
                 item(hf + "f/r/flux2-vae.safetensors?f666666", ["seven"], category="text_encoders")]
        merged, by_id = merge_mirrors({m["url"]: m for m in items})
        same = by_id["b22"]
        self.assertEqual(same["url"], hf + "b/r/flux2-vae.safetensors?b22")  # the copy most workflows name
        self.assertIs(by_id["a1"], same)
        self.assertIs(by_id["c333"], same)
        self.assertIs(by_id["d4444"], same)  # a published hash joins the copies that list none
        self.assertEqual(sorted(w["name"] for w in same["workflows"]), ["five", "four", "one", "three", "two"])
        self.assertEqual((same["size_bytes"], same["sha256"], same["mirrors"]), (336, "1" * 64, 3))
        self.assertEqual(sorted(same["sources"]), ["manager", "templates"])
        self.assertIsNot(by_id["e55555"], same)  # a different hash is a different file
        self.assertIsNot(by_id["f666666"], same)  # so is the same name in another folder
        self.assertEqual(len(merged), 3)

    def test_workflows_name_the_merged_entry_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            templates = Path(tmp) / ".venv" / "Lib" / "site-packages" / "comfyui_workflow_templates_json" / "templates"
            templates.mkdir(parents=True)
            models = [{"name": "a.safetensors", "url": URL, "directory": "vae"},
                      {"name": "a.safetensors", "url": URL.replace("org/repo", "other/repo"), "directory": "vae"}]
            (templates / "flux.json").write_text(json.dumps({"models": models}), encoding="utf-8")
            (templates / "index.json").write_text(json.dumps([{"templates": [{"name": "flux", "title": "Flux"}]}]), encoding="utf-8")
            index = load_index(tmp)
        self.assertEqual(len(index["models"]), 1)
        self.assertEqual(len(index["workflows"][0]["models"]), 1)
        self.assertIn(index["workflows"][0]["models"][0], index["by_id"])

    def test_no_lists_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("Hugging Face search still works", load_index(tmp)["notes"][-1])


class Install(Base):
    def test_plan_downloads_nothing_and_reports_everything(self):
        data = safetensors()
        hub = FakeHub(data)
        plan = self.installer(hub).plan([self.request(len(data))])
        item = plan["items"][0]
        self.assertTrue(item["ok"])
        self.assertEqual(item["size_bytes"], len(data))
        self.assertEqual(item["sha256"], sha256(data).hexdigest())
        self.assertEqual(Path(item["destination"]), (self.vae / "a.safetensors").resolve())
        self.assertGreater(item["free_bytes"], 0)
        self.assertEqual([r[0] for r in hub.requests], ["HEAD"])
        self.assertFalse(self.vae.exists())

    def test_a_renamed_repository_still_gives_the_published_hash(self):
        # Comfy-Org renames repositories (hunyuan3D_2.0_repackaged became Hunyuan-3D-2.0); old links answer 307.
        data = safetensors()
        class Renamed(FakeHub):
            def open(self, url, **kwargs):
                if kwargs.get("method") == "HEAD" and url == URL:
                    self.requests.append(("HEAD", url, {}))
                    return Response(307, headers={"Location": "/org/repo-renamed/resolve/main/a.safetensors"})
                return super().open(url, **kwargs)
        hub = Renamed(data)
        item = self.installer(hub).plan([self.request(len(data))])["items"][0]
        self.assertTrue(item["ok"])
        self.assertEqual(item["sha256"], sha256(data).hexdigest())
        self.assertEqual([r[1] for r in hub.requests], [URL, "https://huggingface.co/org/repo-renamed/resolve/main/a.safetensors"])

    def test_a_copy_in_another_folder_of_the_type_counts_as_installed(self):
        data = safetensors()
        other = self.root / "shared" / "vae"
        other.mkdir(parents=True)
        (other / "a.safetensors").write_bytes(data)
        self.folders.append({"category": "vae", "path": str(other)})
        installer = self.installer(FakeHub(data))
        item = installer.plan([self.request(len(data))])["items"][0]
        self.assertFalse(item["ok"])
        self.assertIn("already installed", item["reason"])
        chosen = installer.plan([{**self.request(len(data)), "folder": str(self.vae)}])["items"][0]
        self.assertTrue(chosen["ok"])

    def test_lists_are_found_in_any_comfy_folder(self):
        with tempfile.TemporaryDirectory() as data, tempfile.TemporaryDirectory() as code:
            cache = Path(code) / "user" / "__manager" / "cache"
            cache.mkdir(parents=True)
            url = "https://huggingface.co/o/r/resolve/main/only-in-cache.safetensors"
            (cache / "1_model-list.json").write_text(json.dumps({"models": [{"name": "x", "filename": "only-in-cache.safetensors", "url": url, "save_path": "vae", "type": "vae"}]}), encoding="utf-8")
            self.assertTrue(any(m["filename"] == "only-in-cache.safetensors" for m in load_index([data, code])["models"].values()))

    def test_install_checks_the_hash_and_arrives_whole(self):
        data = safetensors()
        installer = self.installer(FakeHub(data))
        job = self.wait(installer, installer.start([self.request(len(data))])["jobs"][0])
        self.assertEqual(job["status"], "installed")
        self.assertTrue(job["hash_checked"])
        self.assertEqual((self.vae / "a.safetensors").read_bytes(), data)
        self.assertEqual([p.name for p in self.vae.iterdir()], ["a.safetensors"])
        self.assertTrue(installer.take_rescan())

    def test_a_wrong_hash_leaves_nothing_behind(self):
        data = safetensors()
        installer = self.installer(FakeHub(data, sha="0" * 64))
        job = self.wait(installer, installer.start([self.request(len(data))])["jobs"][0])
        self.assertEqual(job["status"], "failed")
        self.assertIn("SHA-256", job["error"])
        self.assertEqual(list(self.vae.iterdir()), [])

    def test_a_file_that_is_not_safetensors_leaves_nothing_behind(self):
        data = b"PK\x03\x04 this is a pickle zip" * 8
        installer = self.installer(FakeHub(data))
        job = self.wait(installer, installer.start([self.request(len(data))])["jobs"][0])
        self.assertEqual(job["status"], "failed")
        self.assertIn("not a valid safetensors", job["error"])
        self.assertEqual(list(self.vae.iterdir()), [])

    def test_workflows_nearest_to_ready_come_first_with_their_groups(self):
        def model(i, size=None):
            return {"id": f"m{i}", "name": f"m{i}.safetensors", "filename": f"m{i}.safetensors", "url": URL.replace("a.safetensors", f"m{i}.safetensors"),
                    "category": "vae", "subfolder": "", "size_bytes": size, "sha256": None, "description": "", "base": "", "sources": ["templates"],
                    "workflows": [], "blocked": None, "note": None}
        models = {f"m{i}": model(i, 1000 * i) for i in range(1, 6)}
        workflows = [{"name": "far", "title": "Far", "description": "", "kind": "Video", "rank": 900, "models": ["m1", "m2", "m3"]},
                     {"name": "near", "title": "Near", "description": "", "kind": "Image", "rank": 5, "models": ["m4", "m1"]},
                     {"name": "ready", "title": "Ready", "description": "", "kind": "Image", "rank": 1, "models": ["m4"]},
                     {"name": "popular", "title": "Popular", "description": "", "kind": "Image", "rank": 50, "models": ["m4", "m2"]}]
        index = {"models": {m["url"]: m for m in models.values()}, "by_id": models, "workflows": workflows, "notes": []}
        self.vae.mkdir(parents=True)
        (self.vae / "m4.safetensors").write_bytes(b"x")
        result = self.installer(FakeHub(b""), index=index).workflows("")
        self.assertEqual([w["name"] for w in result["workflows"]], ["ready", "popular", "near", "far"])
        self.assertEqual(result["ready"], 1)
        self.assertEqual(result["workflows"][2]["missing_bytes"], 1000)
        self.assertEqual({k["kind"]: k["count"] for k in result["kinds"]}, {"Video": 1, "Image": 3})
        only = self.installer(FakeHub(b""), index=index).workflows("", kind="Video")
        self.assertEqual([w["name"] for w in only["workflows"]], ["far"])
        self.assertEqual({k["kind"] for k in only["kinds"]}, {"Video", "Image"})  # every group stays offered

    def test_a_name_match_comes_before_a_description_match(self):
        def model(i, filename, description=""):
            return {"id": f"m{i}", "name": filename, "filename": filename, "url": URL.replace("a.safetensors", filename),
                    "category": "vae", "subfolder": "", "size_bytes": None, "sha256": None, "description": description, "base": "",
                    "sources": ["templates"], "workflows": [], "blocked": None, "note": None}
        models = {"m1": model(1, "qwen_vae.safetensors", "for users who want speed"), "m2": model(2, "wan_2.1_vae.safetensors")}
        workflows = [{"name": "edit", "title": "Image Edit", "description": "Edit what you want", "kind": "Image", "rank": 900, "models": ["m1"]},
                     {"name": "wan", "title": "Wan 2.2 Text to Video", "description": "", "kind": "Video", "rank": 1, "models": ["m2"]}]
        index = {"models": {m["url"]: m for m in models.values()}, "by_id": models, "workflows": workflows, "notes": []}
        installer = self.installer(FakeHub(b""), index=index)
        self.assertEqual([w["name"] for w in installer.workflows("wan")["workflows"]], ["wan", "edit"])
        self.assertEqual([m["id"] for m in installer.search("wan")["models"]], ["m2", "m1"])
        self.assertEqual(installer.workflows("nothing like it")["total"], 0)

    def test_unsized_files_are_sized_once_on_request(self):
        data = safetensors()
        item = {"id": "m1", "name": "a.safetensors", "filename": "a.safetensors", "url": URL, "category": "vae", "subfolder": "",
                "size_bytes": None, "sha256": None, "description": "", "base": "", "sources": ["templates"], "workflows": [], "blocked": None, "note": None}
        hub = FakeHub(data)
        installer = self.installer(hub, index={"models": {URL: item}, "by_id": {"m1": item}, "workflows": [], "notes": []})
        self.assertEqual(installer.sizes(["m1", "unknown"]), {"sizes": {"m1": len(data)}})
        installer.sizes(["m1"])
        self.assertEqual([r[0] for r in hub.requests], ["HEAD"])
        with self.assertRaises(ValueError):
            installer.sizes([])

    def test_finished_downloads_can_be_cleared(self):
        data = safetensors()
        installer = self.installer(FakeHub(data))
        self.wait(installer, installer.start([self.request(len(data))])["jobs"][0])
        self.assertEqual(len(installer.status()["jobs"]), 1)
        self.assertEqual(installer.clear_finished()["jobs"], [])
        self.assertTrue((self.vae / "a.safetensors").is_file())

    def test_search_finds_gguf_repos_too_and_merges_them(self):
        answers = {"safetensors": [{"id": "org/both", "downloads": 50}, {"id": "org/plain", "downloads": 900}],
                   "gguf": [{"id": "org/both", "downloads": 50}, {"id": "org/quantised", "downloads": 300}]}
        class Hub:
            asked = []
            def open(self, url, **_):
                tag = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))["filter"]
                Hub.asked.append(tag)
                return Response(200, json.dumps(answers[tag]).encode())
        repos = self.installer(Hub()).hub_search("flux")["repos"]
        self.assertEqual(Hub.asked, ["safetensors", "gguf"])
        self.assertEqual([(r["repo"], r["formats"]) for r in repos],
                         [("org/plain", ["safetensors"]), ("org/quantised", ["gguf"]), ("org/both", ["safetensors", "gguf"])])

    def test_a_gguf_download_is_checked_as_gguf_and_kept(self):
        data = gguf()
        installer = self.installer(FakeHub(data))
        request = {**self.request(len(data)), "url": URL.replace(".safetensors", ".gguf"), "filename": "a.gguf"}
        job = self.wait(installer, installer.start([request])["jobs"][0])
        self.assertEqual(job["status"], "installed", job.get("error"))
        self.assertEqual((self.vae / "a.gguf").read_bytes(), data)

    def test_a_file_named_gguf_that_is_not_gguf_leaves_nothing_behind(self):
        data = safetensors()
        installer = self.installer(FakeHub(data))
        request = {**self.request(len(data)), "url": URL.replace(".safetensors", ".gguf"), "filename": "a.gguf"}
        job = self.wait(installer, installer.start([request])["jobs"][0])
        self.assertEqual(job["status"], "failed")
        self.assertIn("not a valid GGUF", job["error"])
        self.assertEqual(list(self.vae.iterdir()), [])

    def test_a_cut_download_resumes_and_the_whole_file_is_checked(self):
        data = safetensors(4096)
        hub = FakeHub(data, cut=1000)
        installer = self.installer(hub)
        job = self.wait(installer, installer.start([self.request(len(data))])["jobs"][0])
        self.assertEqual(job["status"], "failed")
        self.assertIn("retry to resume", job["error"])
        self.assertEqual((self.vae / "a.safetensors.modelrun-part").stat().st_size, 1000)
        hub.cut = None
        self.assertEqual(self.installer(hub).plan([self.request(len(data))])["items"][0]["resume_bytes"], 1000)
        job = self.wait(installer, installer.start([self.request(len(data))])["jobs"][0])
        self.assertEqual(job["status"], "installed")
        self.assertTrue(job["hash_checked"])
        self.assertEqual(hub.requests[-1][2]["Range"], "bytes=1000-")
        self.assertEqual((self.vae / "a.safetensors").read_bytes(), data)

    def test_an_existing_file_is_never_replaced(self):
        data = safetensors()
        self.vae.mkdir(parents=True)
        (self.vae / "a.safetensors").write_bytes(b"mine")
        installer = self.installer(FakeHub(data))
        plan = installer.plan([self.request(len(data))])["items"][0]
        self.assertFalse(plan["ok"])
        self.assertIn("already installed", plan["reason"])
        with self.assertRaisesRegex(ValueError, "already installed"):
            installer.start([self.request(len(data))])
        self.assertEqual((self.vae / "a.safetensors").read_bytes(), b"mine")

    def test_a_size_changed_since_confirming_is_refused(self):
        data = safetensors()
        installer = self.installer(FakeHub(data))
        with self.assertRaisesRegex(ValueError, "changed size"):
            installer.start([self.request(len(data) + 1)])
        self.assertEqual(installer.status()["jobs"], [])

    def test_paths_stay_inside_comfyuis_folder(self):
        installer = self.installer(FakeHub(safetensors()))
        for extra in ({"subfolder": "../.."}, {"subfolder": "a/../../b"}, {"filename": "../x.safetensors"}, {"category": "custom_nodes"}, {"folder": str(self.root)}):
            plan = installer.plan([self.request(1, **extra)])["items"][0]
            self.assertFalse(plan["ok"], extra)
        self.assertFalse(any(self.root.rglob("*.safetensors")))

    def test_gated_and_missing_files_explain_themselves(self):
        gated = self.installer(FakeHub(b"", head=Response(401))).plan([self.request(1)])["items"][0]
        self.assertFalse(gated["ok"])
        self.assertEqual(gated["access"], "gated")
        self.assertIn("licence", gated["reason"])
        missing = self.installer(FakeHub(b"", head=Response(404))).plan([self.request(1)])["items"][0]
        self.assertEqual(missing["access"], "missing")

    def test_too_little_disk_is_refused(self):
        huge = FakeHub(b"", head=Response(302, headers={"X-Linked-Size": str(10 ** 18), "Location": "https://cdn-lfs.huggingface.co/x"}))
        plan = self.installer(huge).plan([self.request(10 ** 18)])["items"][0]
        self.assertFalse(plan["ok"])
        self.assertIn("disk space", plan["reason"])

    def test_cancel_can_keep_or_discard_the_partial_file(self):
        data = safetensors(1024)
        class Slow(Response):
            installer, job = None, None
            def read(self, size=-1):
                time.sleep(0.02)
                return super().read(16)
        slow = Slow(200, data)
        installer = self.installer(FakeHub(data, get=slow))
        job_id = installer.start([self.request(len(data))])["jobs"][0]
        installer.cancel(job_id, discard=True)
        self.assertEqual(self.wait(installer, job_id)["status"], "cancelled")
        self.assertEqual(list(self.vae.iterdir()) if self.vae.exists() else [], [])

    def test_status_never_shows_a_token_and_a_token_only_goes_to_hugging_face(self):
        previous = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "secret-token"
        self.addCleanup(lambda: os.environ.pop("HF_TOKEN", None) if previous is None else os.environ.__setitem__("HF_TOKEN", previous))
        installer = self.installer(FakeHub(safetensors()))
        self.assertTrue(installer.search()["logged_in"])
        self.assertNotIn("secret-token", json.dumps(installer.search()) + json.dumps(installer.status()))
        sent = []
        transport = Transport()
        class Opener:
            def open(self, request, timeout=None):
                sent.append((request.full_url, request.get_header("Authorization")))
                if len(sent) == 1:
                    return Response(302, headers={"Location": "https://cdn-lfs.huggingface.co/file"})
                return Response(200)
        transport._opener = Opener()
        transport.open(URL)
        self.assertEqual(sent, [(URL, "Bearer secret-token"), ("https://cdn-lfs.huggingface.co/file", None)])


if __name__ == "__main__":
    unittest.main()
