import tempfile
import unittest
from pathlib import Path

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    onnx = None

from model_run.onnx_graph import inspect_onnx


@unittest.skipIf(onnx is None, "optional onnx dependency unavailable")
class OnnxTests(unittest.TestCase):
    def test_real_serialized_graph_and_external_weights(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / "model.onnx"
            x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 2])
            y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 2])
            weight = helper.make_tensor("w", TensorProto.FLOAT, [2], [1.0, 2.0])
            weight.ClearField("float_data")
            weight.data_location = TensorProto.EXTERNAL
            entry = weight.external_data.add()
            entry.key, entry.value = "location", "missing-weights.bin"
            graph = helper.make_graph([helper.make_node("Add", ["x", "w"], ["y"], name="add")], "test", [x], [y], [weight])
            model = helper.make_model(graph)
            path.write_bytes(model.SerializeToString())
            result = inspect_onnx(path)
            self.assertEqual(result["nodes"][0]["inputs"], ["x", "w"])
            self.assertEqual(result["outputs"][0]["shape"], ["batch", 2])
            self.assertEqual(result["kind"], "onnx_graph")
            self.assertTrue(result["initializers"][0]["external"])

    def test_corrupt_file(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / "bad.onnx"
            path.write_bytes(b"not a graph")
            with self.assertRaises(ValueError): inspect_onnx(path)

    def test_nested_graphs_have_distinct_scopes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / "branch.onnx"
            inner_output = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
            branch = helper.make_graph([helper.make_node("Identity", ["outer"], ["z"])], "branch", [], [inner_output])
            graph = helper.make_graph([helper.make_node("If", ["cond"], ["y"], then_branch=branch, else_branch=branch)], "outer", [], [])
            path.write_bytes(helper.make_model(graph).SerializeToString())
            result = inspect_onnx(path)
            children = result["nodes"][0]["subgraphs"]
            self.assertEqual(len(children), 2)
            self.assertNotEqual(children[0]["graph"]["scope"], children[1]["graph"]["scope"])
            self.assertEqual(children[0]["graph"]["nodes"][0]["inputs"], ["outer"])
            self.assertEqual(result["capabilities"]["execution"]["status"], "unsupported")
