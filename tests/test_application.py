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
                self.assertEqual(package["size_bytes"], sum(p.stat().st_size for p in root.iterdir()))
                self.assertEqual(app.inspect(package["id"])["config"]["model_type"], "fixture")
                with self.assertRaises(ValueError): app.inspect("../config.json")
                ids = [app.publish(trace())["id"] for _ in range(18)]
                self.assertEqual(len(app.snapshot()["traces"]), 16)
                with self.assertRaises(ValueError): app.trace(ids[0])
                self.assertEqual(app.trace(ids[-1]), trace())
                app.delete_trace(ids[-1])
                self.assertEqual(len(app.snapshot()["traces"]), 15)
            finally:
                app.close()

    def test_comfyui_folders_that_hold_no_models_are_not_listed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            for name in ("configs", "custom_nodes", "vae"):
                (root / name).mkdir()
            (root / "configs" / "v1-inference.yaml").write_text("model: {}")
            (root / "custom_nodes" / "pack.py").write_text("")
            (root / "vae" / "a.bin").write_bytes(b"x")
            app = Application([{"path": str(root / name), "category": name} for name in ("configs", "custom_nodes", "vae")])
            try:
                app.scan()
                app._future.result(5)
                self.assertEqual([a["relative_path"] for a in app.catalog()["artifacts"]], ["vae/a.bin"])
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

    def test_runs_are_retained_and_evicted_whole(self):
        import model_run.application as module
        app = Application()
        try:
            first = [app.publish(trace(), run="a", step=step)["id"] for step in ("module", "generation", "attention")]
            for index in range(module.MAX_RUNS - 1):
                app.publish(trace(), run=f"r{index}")
            self.assertEqual(app.snapshot()["retention"]["runs"], module.MAX_RUNS)
            self.assertEqual(app.trace(first[0]), trace())
            app.publish(trace(), run="next")
            kept = {t["id"] for t in app.snapshot()["traces"]}
            self.assertFalse(kept & set(first))  # the whole oldest run left together
            self.assertEqual(app._trace_bytes, sum(t["size_bytes"] for t in app.snapshot()["traces"]))
        finally:
            app.close()

    def test_pinned_runs_survive_and_pins_are_capped(self):
        import model_run.application as module
        app = Application()
        try:
            keep = app.publish(trace(), run="keep", step="module")["id"]
            app.pin_trace(keep, True)
            sibling = app.publish(trace(), run="keep", step="attention")["id"]
            self.assertTrue(all(t["pinned"] for t in app.snapshot()["traces"] if t["run"] == "keep"))
            for index in range(module.MAX_RUNS + 4):
                app.publish(trace(), run=f"r{index}")
            kept = {t["id"] for t in app.snapshot()["traces"]}
            self.assertLessEqual({keep, sibling}, kept)
            self.assertEqual(app.snapshot()["retention"]["runs"], module.MAX_RUNS)
            for index in range(module.MAX_PINNED_RUNS - 1):
                app.pin_trace(app.publish(trace(), run=f"p{index}")["id"], True)
            extra = app.publish(trace(), run="extra")["id"]
            with self.assertRaises(ValueError):
                app.pin_trace(extra, True)
            app.pin_trace(keep, False)
            app.pin_trace(extra, True)
            with self.assertRaises(ValueError):
                app.pin_trace(extra, "yes")
        finally:
            app.close()

    def test_replacing_a_trace_keeps_its_run_and_pin(self):
        app = Application()
        try:
            identifier = app.publish(trace(), run="a", step="attention", note="p" * 99)["id"]
            app.pin_trace(identifier, True)
            app.publish(trace(), replace_id=identifier)
            self.assertEqual(app.snapshot()["traces"][0]["note"], "p" * 48)
            record = app.snapshot()["traces"][0]
            self.assertEqual((record["run"], record["step"], record["pinned"], record["revision"]), ("a", "attention", True, 2))
        finally:
            app.close()

    def test_delete_removes_the_whole_run(self):
        app = Application()
        try:
            one = app.publish(trace(), run="a", step="module")["id"]
            app.publish(trace(), run="a", step="attention")
            other = app.publish(trace())["id"]
            app.delete_trace(one)
            self.assertEqual([t["id"] for t in app.snapshot()["traces"]], [other])
            self.assertEqual(app._trace_bytes, app.snapshot()["traces"][0]["size_bytes"])
        finally:
            app.close()

    def test_byte_limit_never_evicts_the_newest_trace(self):
        import model_run.application as module
        app = Application()
        original = module.MAX_RETAINED_BYTES
        try:
            module.MAX_RETAINED_BYTES = 1
            older = app.publish(trace(), run="a")["id"]
            newest = app.publish(trace(), run="b")["id"]
            self.assertEqual([t["id"] for t in app.snapshot()["traces"]], [newest])
            self.assertNotEqual(older, newest)
        finally:
            module.MAX_RETAINED_BYTES = original
            app.close()

    def test_invalid_run_group_is_rejected(self):
        app = Application()
        try:
            with self.assertRaises(ValueError):
                app.run({"prompt": "x", "run_group": "../bad"})
            with self.assertRaises(ValueError):
                app.run({"prompt": "x", "run_step": "other"})
        finally:
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
            # Another captured layer gets its own block; an uncaptured layer is refused.
            attention["layer_stacks"] = [{"index": 0, "mean": [[1.0, 0.0], [0.5, 0.5]]}, {"index": 1, "heads": [[[1.0, 0.0], [0.5, 0.5]], [[1.0, 0.0], [0.5, 0.5]]]}]
            app.publish(attention, "fixture", replace_id=identifier)
            other = result(1, 0.3)
            other["layer"] = 0
            app._attach_ablation(identifier, other)
            self.assertEqual([b["layer"] for b in app.trace(identifier)["ablations"]], [0])
            self.assertNotIn("ablation", app.trace(identifier))  # the republished fixture carried no probed-layer block
            missing = result(0, 0.1)
            missing["layer"] = 5
            with self.assertRaises(ValueError): app._attach_ablation(identifier, missing)
            wrong_prompt = result(0, 0.1)
            wrong_prompt["position"] = 5
            with self.assertRaises(ValueError): app._attach_ablation(identifier, wrong_prompt)
            wrong_layer = result(0, 0.1)
            wrong_layer["layer"] = 7  # not the probed layer and not in layer_stacks
            with self.assertRaises(ValueError): app._attach_ablation(identifier, wrong_layer)
            with self.assertRaises(ValueError): app._attach_ablation("missing", result(0, 0.1))
            with self.assertRaises(ValueError): app.run({"prompt": "hi", "ablate": "head"})
            with self.assertRaises(ValueError): app.run({"prompt": "hi", "ablate": "nope", "attach_to": identifier})
        finally:
            app.close()



class NativeWorkflowTests(unittest.TestCase):
    def test_workflow_stays_out_of_the_trace_and_follows_retention(self):
        app = Application()
        try:
            identifier = app.publish(trace(), "fixture")["id"]
            app.remember_workflow(identifier, {"6": {"class_type": "ModelRunObserve", "inputs": {"ablate_tokens": ""}, "is_changed": [float("nan")]}}, 6)
            self.assertEqual(app.native_workflow(identifier)["node"], "6")
            json.dumps(app.native_workflow(identifier), allow_nan=False)
            self.assertNotIn("prompt", app.trace(identifier))
            copy = app.native_workflow(identifier)
            copy["prompt"]["6"]["inputs"]["ablate_tokens"] = "1"
            self.assertEqual(app.native_workflow(identifier)["prompt"]["6"]["inputs"]["ablate_tokens"], "")
            app.delete_trace(identifier)
            with self.assertRaises(ValueError): app.native_workflow(identifier)
            with self.assertRaises(ValueError): app.native_workflow("missing")
        finally:
            app.close()
