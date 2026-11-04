import tempfile
import unittest
from pathlib import Path
from model_run.trace_store import save_trace, load_trace


class TraceStoreTests(unittest.TestCase):
    def test_reject_malformed_viewer_metadata(self):
        from model_run.trace_store import encode_trace
        base = {"schema_version": "1.0", "kind": "pytorch_module_capture", "model_class": "fixture",
                "torch_version": "fixture", "events": [], "capture": {}, "limitations": []}
        for extra in ({"model_name": {}}, {"tokens": [{"id": 1, "text": []}]},
                      {"structure": {"nodes": [{"path": 5, "type": "fixture"}]}},
                      {"output": {"top_tokens": [{"token_id": 1, "probability": 2}]}},
                      {"preview": {"mime": "text/html", "base64": ""}}):
            with self.subTest(extra=extra), self.assertRaises(ValueError): encode_trace({**base, **extra})

    def test_roundtrip_and_reject_invalid(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / "trace.json"
            value = {"schema_version": "1.0", "kind": "attention_capture", "model_class": "test.Model", "torch_version": "test",
                     "limitations": [], "shape": [1, 1], "axes": ["query_position", "key_position"], "layer": 0, "head": 0, "values": [[0.5]]}
            save_trace(path, value)
            self.assertEqual(load_trace(path), value)
            with self.assertRaises(ValueError): save_trace(path, {"schema_version": "1.0", "kind": "attention_capture", "x": float("nan")})
            self.assertEqual(load_trace(path), value)
            path.write_text('{"schema_version":"1.0","kind":"unknown"}')
            with self.assertRaises(ValueError): load_trace(path)

    def test_matrix_shape_mismatch(self):
        value = {"schema_version": "1.0", "kind": "attention_capture", "model_class": "test", "torch_version": "test",
                 "limitations": [], "shape": [1, 2], "axes": ["query_position", "key_position"], "layer": 0, "head": 0, "values": [[0.5]]}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            with self.assertRaises(ValueError): save_trace(Path(tmp) / "bad.json", value)

    def test_duplicate_fields_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"schema_version":"1.0","schema_version":"1.0","kind":"attention_capture"}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"): load_trace(path)
