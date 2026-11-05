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

from model_run.model_install import ModelInstaller, Transport, _safe_part
from model_run.model_sources import blocked_reason, guess_folder, huggingface_parts, load_index, parse_size

URL = "https://huggingface.co/org/repo/resolve/main/a.safetensors"


def safetensors(size=64):
    header = json.dumps({"t": {"dtype": "U8", "shape": [size], "data_offsets": [0, size]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"x" * size


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
