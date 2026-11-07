import json
import struct
import tempfile
import unittest
from pathlib import Path

from model_run.inventory import InventoryError, check_gguf, inspect, inspect_file, inspect_index


def write_tensor(path, header, payload=b"1234"):
    data = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(data)) + data + payload)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_inventory_is_metadata_only(self):
        path = self.root / "model.safetensors"
        write_tensor(path, {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
        report = inspect(path)
        self.assertEqual(report["artifact"]["tensors"][0]["shape"], [1])
        self.assertEqual(report["capabilities"]["execution"]["status"], "unsupported")

    def test_gguf_header_is_read_without_its_tensors(self):
        path = self.root / "model.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 291, 24) + b"x" * 32)
        self.assertEqual((check_gguf(path)["version"], check_gguf(path)["tensor_count"]), (3, 291))
        for bad in (b"GGUG" + struct.pack("<IQQ", 3, 1, 0), b"GGUF" + struct.pack("<IQQ", 1, 1, 0), b"GGUF" + struct.pack("<IQQ", 3, 0, 0),
                    b"GGUF" + struct.pack("<IQQ", 3, 2**40, 0), b"GGUF"):
            path.write_bytes(bad)
            with self.assertRaises(InventoryError, msg=bad[:12]): check_gguf(path)

    def test_summary_counts_parameters_bytes_and_block_stacks(self):
        from model_run.inventory import summarize
        report = {"tensors": [{"name": "double_blocks.0.attn.w", "dtype": "BF16", "shape": [4, 4]}, {"name": "double_blocks.18.x", "dtype": "BF16", "shape": [2]},
                              {"name": "model.single_blocks.37.y", "dtype": "F8_E4M3", "shape": [10]}, {"name": "img_in.weight", "dtype": "F32", "shape": [3]}]}
        summary = summarize(report)
        self.assertEqual(summary["parameters"], 31)
        self.assertEqual(summary["bytes_by_dtype"], {"BF16": 36, "F32": 12, "F8_E4M3": 10})
        self.assertEqual(summary["block_stacks"], {"single_blocks": 38, "double_blocks": 19})

    def test_bad_header_and_overlap(self):
        path = self.root / "bad.safetensors"
        path.write_bytes(struct.pack("<Q", 2**40))
        with self.assertRaises(InventoryError): inspect_file(path)
        write_tensor(path, {"a": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]}, "b": {"dtype": "U8", "shape": [2], "data_offsets": [2, 4]}})
        with self.assertRaises(InventoryError): inspect_file(path)

    def test_duplicate_json_key(self):
        path = self.root / "bad.safetensors"
        data = b'{"x":{},"x":{}}'
        path.write_bytes(struct.pack("<Q", len(data)) + data)
        with self.assertRaises(InventoryError): inspect_file(path)

    def test_shard_index_validation(self):
        shard = self.root / "one.safetensors"
        write_tensor(shard, {"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}})
        index = self.root / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"a": shard.name}}))
        self.assertEqual(inspect_index(index)["tensor_count"], 1)
        index.write_text(json.dumps({"weight_map": {"b": shard.name}}))
        with self.assertRaises(InventoryError): inspect_index(index)
        index.write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}))
        with self.assertRaises(InventoryError): inspect_index(index)

    def test_empty_tensor(self):
        path = self.root / "empty.safetensors"
        write_tensor(path, {"a": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}}, b"")
        self.assertEqual(len(inspect_file(path)["tensors"]), 1)

    def test_bad_lengths_and_unindexed_data(self):
        path = self.root / "bad.safetensors"
        write_tensor(path, {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}})
        with self.assertRaises(InventoryError): inspect_file(path)
        write_tensor(path, {"a": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]}})
        with self.assertRaises(InventoryError): inspect_file(path)

    def test_unknown_dtype_reported_without_guessing(self):
        path = self.root / "future.safetensors"
        write_tensor(path, {"a": {"dtype": "FUTURE", "shape": [1], "data_offsets": [0, 4]}})
        self.assertIn("FUTURE", inspect_file(path)["limitations"][0])


if __name__ == "__main__": unittest.main()
