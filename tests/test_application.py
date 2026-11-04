import json
from pathlib import Path
import tempfile
from threading import Thread
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from model_run.application import Application
from model_run.server import make_server


def trace():
    return {"schema_version": "1.0", "kind": "pytorch_module_capture", "model_class": "fixture",
            "torch_version": "fixture", "capture": {}, "events": [], "limitations": []}


class ApplicationTests(unittest.TestCase):
    def test_atomic_trace_replacement_preserves_history(self):
        app = Application()
        try:
            ids = [app.publish(trace())["id"] for _ in range(8)]
            updated = trace()
            updated["model_class"] = "updated"
            result = app.publish(updated, replace_id=ids[-1])
            self.assertEqual(result["id"], ids[-1])
            retained = app.snapshot()["traces"]
            self.assertEqual([t["id"] for t in retained], ids)
            self.assertEqual(retained[-1]["revision"], 2)
            self.assertEqual(app.trace(ids[-1]), updated)
            self.assertEqual(app._trace_bytes, sum(t["size_bytes"] for t in retained))
            with self.assertRaises(ValueError):
                app.publish({"schema_version": "invalid"}, replace_id=ids[-1])
            self.assertEqual(app.trace(ids[-1]), updated)
        finally:
            app.close()

    def test_discovery_and_retention(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type":"fixture","architectures":["FixtureForCausalLM"]}')
            (root / "model.safetensors").write_bytes(b"bad")
            app = Application([{"path": str(root), "category": "diffusers"}])
            try:
                app.scan()
                app._future.result(5)
                catalog = app.catalog()
                package = next(a for a in catalog["artifacts"] if a["task"] == "text")
                self.assertNotIn("path", package)
                self.assertEqual(app.inspect(package["id"])["config"]["model_type"], "fixture")
                with self.assertRaises(ValueError): app.inspect("../config.json")
                ids = [app.publish(trace())["id"] for _ in range(10)]
                self.assertEqual(len(app.snapshot()["traces"]), 8)
                with self.assertRaises(ValueError): app.trace(ids[0])
                self.assertEqual(app.trace(ids[-1]), trace())
                app.delete_trace(ids[-1])
                self.assertEqual(len(app.snapshot()["traces"]), 7)
            finally:
                app.close()

    def test_http_origin_token_and_trace_roundtrip(self):
        app = Application()
        server = make_server(app, 0)
        with self.assertRaises(OSError): make_server(app, server.server_port)
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        def request(path, data=None, headers=None):
            req = Request(base + path, data=None if data is None else json.dumps(data).encode(),
                          headers={"Content-Type": "application/json", **(headers or {})})
            with urlopen(req, timeout=3) as response: return json.load(response)
        try:
            with self.assertRaises(HTTPError) as error: request("/api/state")
            self.assertEqual(error.exception.code, 403)
            with self.assertRaises(HTTPError): request("/api/session", headers={"Origin": "https://example.org"})
            with self.assertRaises(HTTPError): request("/api/session", headers={"Host": "evil.test"})
            token = request("/api/session")["token"]
            headers = {"X-Model-Run-Token": token}
            result = request("/api/import", trace(), headers)
            self.assertEqual(request("/api/traces/" + result["id"], headers=headers), trace())
            with self.assertRaises(HTTPError): request("/api/import", {"schema_version": "9"}, headers)
            self.assertEqual(request("/api/state", headers=headers)["host"], "standalone")
        finally:
            server.shutdown()
            server.server_close()
            app.close()


if __name__ == "__main__": unittest.main()


class AblationAttachTests(unittest.TestCase):
    def test_attach_merges_heads_and_checks_prompt(self):
        app = Application()
        try:
            attention = {"schema_version": "1.0", "kind": "attention_capture", "model_class": "fixture", "torch_version": "fixture",
                         "layer": 1, "head": 0, "head_count": 2, "axes": ["query_position", "key_position"], "shape": [2, 2],
                         "values": [[1.0, 0.0], [0.5, 0.5]], "input": {"token_count": 2, "token_ids_included": False}, "limitations": []}
            identifier = app.publish(attention, "fixture")["id"]
            distribution = {"top_tokens": [{"token_id": 3, "probability": 0.5}], "entropy_bits": 1.0}
            def result(head, kl):
                return {"schema_version": "1.0", "kind": "head_ablation", "model_class": "fixture", "torch_version": "fixture", "method": "zero_ablation", "layer": 1, "head_count": 2, "head_dim": 4, "position": 1,
                        "baseline": distribution, "limitations": [],
                        "heads": [{"head": head, **distribution, "kl_bits": kl, "baseline_top_probability": 0.4, "top_token_changed": False}]}
            app._attach_ablation(identifier, result(1, 0.5))
            app._attach_ablation(identifier, result(0, 0.25))
            app._attach_ablation(identifier, result(1, 0.75))
            merged = app.trace(identifier)
            self.assertEqual([(e["head"], e["kl_bits"]) for e in merged["ablation"]["heads"]], [(0, 0.25), (1, 0.75)])
            self.assertEqual(merged["values"], attention["values"])
            self.assertEqual(app.snapshot()["traces"][-1]["revision"], 4)
            self.assertEqual(app.snapshot()["traces"][-1]["name"], "fixture")
            wrong_prompt = result(0, 0.1)
            wrong_prompt["position"] = 5
            with self.assertRaises(ValueError): app._attach_ablation(identifier, wrong_prompt)
            wrong_layer = result(0, 0.1)
            wrong_layer["layer"] = 0
            with self.assertRaises(ValueError): app._attach_ablation(identifier, wrong_layer)
            with self.assertRaises(ValueError): app._attach_ablation("missing", result(0, 0.1))
            with self.assertRaises(ValueError): app.run({"prompt": "hi", "ablate": "head"})
            with self.assertRaises(ValueError): app.run({"prompt": "hi", "ablate": "nope", "attach_to": identifier})
        finally:
            app.close()
