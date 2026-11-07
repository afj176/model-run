import tempfile
import unittest
from pathlib import Path
from model_run.trace_store import decode_trace, encode_trace, load_trace, pack_values, save_trace, unpack_values


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


class PackedValueTests(unittest.TestCase):
    def test_measured_values_pack_and_come_back(self):
        values = [round(0.1 * i - 3.3, 4) for i in range(200)] + [None, -4.3177, 12345.6]
        value = {"events": [{"values": values, "indices": list(range(100)), "short": [0.5, 1.5]}]}
        packed = pack_values(value)
        self.assertEqual(packed["events"][0]["values"]["__packed__"], "float32")
        self.assertEqual(packed["events"][0]["indices"], list(range(100)))  # whole numbers stay exact
        self.assertEqual(packed["events"][0]["short"], [0.5, 1.5])
        self.assertEqual(unpack_values(packed), value)

    def test_a_stored_trace_is_smaller_and_reads_back(self):
        # An attention matrix: each 128-value row is packed on its own.
        import math
        rows = [[float(f"{abs(math.sin(r * 128 + c)) * 0.987654:.5g}") for c in range(128)] for r in range(128)]
        trace = {"schema_version": "1.0", "kind": "attention_capture", "model_class": "test.Model", "torch_version": "test",
                 "limitations": [], "shape": [128, 128], "axes": ["query_position", "key_position"], "layer": 0, "head": 0,
                 "values": rows}
        import json
        data = encode_trace(trace)
        self.assertLess(len(data), len(json.dumps(trace, separators=(",", ":"))) * 0.75)
        self.assertEqual(decode_trace(data), trace)

    def test_a_damaged_pack_is_refused(self):
        for bad in ({"__packed__": "float32", "n": 3, "data": "AAAA"}, {"__packed__": "float32", "n": 1, "data": "!!"},
                    {"__packed__": "float32", "n": 1, "data": "AAAAAA==", "extra": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                unpack_values(bad)


if __name__ == "__main__":
    unittest.main()
