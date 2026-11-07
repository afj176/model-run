import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

from model_run.trace_store import encode_trace


@unittest.skipIf(Image is None, "optional Pillow dependency unavailable")
class SourceImageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        Image.new("RGB", (900, 600), (200, 40, 90)).save(Path(self.tmp.name) / "cool-cat.jpg")
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_annotated_filepath = lambda name: str(Path(self.tmp.name) / name)
        previous = sys.modules.get("folder_paths")
        sys.modules["folder_paths"] = folder_paths
        self.addCleanup(lambda: sys.modules.__setitem__("folder_paths", previous) if previous else sys.modules.pop("folder_paths", None))

    def test_a_runs_load_image_becomes_a_small_source_picture(self):
        from model_run.comfy_bridge import source_images
        graph = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "cool-cat.jpg"}, "_meta": {"title": "Load Image"}},
            "2": {"class_type": "LoadImage", "inputs": {"image": "gone.png"}},
            "3": {"class_type": "KSampler", "inputs": {}},
        }
        found = source_images(graph)
        self.assertEqual([(s["title"], s["file"]) for s in found], [("Load Image", "cool-cat.jpg")])
        import base64, io
        with Image.open(io.BytesIO(base64.b64decode(found[0]["base64"]))) as thumb:
            self.assertEqual(max(thumb.size), 384)
        trace = {"schema_version": "1.0", "kind": "pytorch_module_capture", "model_class": "Unet", "torch_version": "2", "events": [], "capture": {}, "limitations": [], "source_images": found}
        encode_trace(trace)
        trace["source_images"] = [{**found[0], "base64": "bm90IGEgcG5n"}]
        with self.assertRaises(ValueError):
            encode_trace(trace)


if __name__ == "__main__":
    unittest.main()
