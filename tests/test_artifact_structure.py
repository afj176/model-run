import unittest
from model_run.artifact_structure import tensor_namespace


class ArtifactStructureTests(unittest.TestCase):
    def test_namespace_is_not_claimed_as_execution(self):
        result = tensor_namespace({"tensors": [
            {"name": "model.layers.0.weight", "shape": [3, 4], "dtype": "F32"},
            {"name": "model.layers.0.bias", "shape": [3], "dtype": "F32"}]}, name="fixture")
        self.assertEqual(result["kind"], "artifact_tensor_namespace")
        self.assertEqual(len(result["nodes"][-1]["parameters"]), 2)
        self.assertNotIn("edges", result)
        ordered = tensor_namespace({"tensors": [{"name": f"layers.{i}.weight", "shape": [1], "dtype": "F32"} for i in [10, 2, 1]]}, name="fixture")
        self.assertEqual([n["path"] for n in ordered["nodes"] if n["parameters"]], ["layers.1", "layers.2", "layers.10"])
        bounded = tensor_namespace({"tensors": [{"name": "a.b.c.weight", "shape": [], "dtype": "F32"}]}, name="fixture", max_nodes=2)
        self.assertTrue(bounded["truncated"])
        self.assertEqual(len(bounded["nodes"]), 2)
