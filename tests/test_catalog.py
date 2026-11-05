import json
import struct
import tempfile
import unittest
from pathlib import Path
from model_run.catalog import discover_models


class CatalogTests(unittest.TestCase):
    def test_inventory_dedup_and_unknown(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            header = json.dumps({"w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
            (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"x")
            (root / "unknown.bin").write_bytes(b"not parsed")
            result = discover_models([{"path": str(root), "category": "vae"}, {"path": str(root), "category": "checkpoints"}])
            self.assertEqual(len(result["artifacts"]), 2)
            item = next(x for x in result["artifacts"] if x["format"] == "safetensors")
            self.assertEqual(item["tensor_count"], 1)
            self.assertEqual(len(item["origins"]), 2)
            self.assertEqual(item["configured_categories"], ["checkpoints", "vae"])
            self.assertEqual(item["tensor_dtypes"], {"U8": 1})
            self.assertEqual(item["capabilities"]["execution"]["status"], "unverified")
            self.assertEqual(next(x for x in result["artifacts"] if x["format"] == "unknown")["status"], "uninspected")

    def test_comfy_placeholders_are_not_models(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            (root / "put_checkpoints_here").write_bytes(b"")
            (root / "put_this.bin").write_bytes(b"x")
            found = discover_models([{"category": "checkpoints", "path": str(root)}])
            self.assertEqual([Path(a["path"]).name for a in found["artifacts"]], ["put_this.bin"])

    def test_limit_invalid_and_missing(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            (root / "a.safetensors").write_bytes(b"bad")
            (root / "b.bin").write_bytes(b"x")
            result = discover_models([{"path": str(root / "missing"), "category": "vae"}, {"path": str(root), "category": "vae"}], max_files=1)
            self.assertTrue(result["truncated"])
            self.assertEqual(result["artifacts"][0]["status"], "invalid_or_unreadable")
            self.assertEqual(len(result["errors"]), 1)

    def test_depth_limit_reported(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            (Path(tmp) / "nested").mkdir()
            result = discover_models([{"path": tmp, "category": "vae"}], max_depth=0)
            self.assertTrue(result["depth_limited"])

    def test_sharded_index_is_validated_and_links_members(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            header = json.dumps({"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
            shard = root / "one.safetensors"
            shard.write_bytes(struct.pack("<Q", len(header)) + header + b"x")
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"a": shard.name}}))
            report = discover_models([{"path": str(root), "category": "text_encoders"}])
            item = next(x for x in report["artifacts"] if x["format"] == "safetensors_index")
            self.assertEqual(item["status"], "metadata_inspected")
            self.assertEqual(item["shard_paths"], [str(shard.resolve())])
            index.write_text(json.dumps({"weight_map": {"b": shard.name}}))
            report = discover_models([{"path": str(root), "category": "text_encoders"}])
            self.assertEqual(next(x for x in report["artifacts"] if x["format"] == "safetensors_index")["status"], "invalid_or_unreadable")
