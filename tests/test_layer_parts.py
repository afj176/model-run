import unittest

try:
    import torch
except ImportError:
    torch = None

from model_run.capture import capture_forward
from model_run.layer_parts import PART_FEATURES, layer_parts, order_parts
from model_run.tensor_slices import SliceRecorder
from model_run.trace_store import encode_trace


@unittest.skipIf(torch is None, "optional PyTorch dependency unavailable")
class LayerPartTests(unittest.TestCase):
    def model(self, depth=600):
        class Attention(torch.nn.Module):
            def forward(self, x):
                return x * 0.5, None  # like a real attention: (output, cache)
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = torch.nn.Identity()
                self.self_attn = Attention()
                self.mlp = torch.nn.Linear(depth, depth)
            def forward(self, x):
                x = x + self.self_attn(self.input_layernorm(x))[0]
                return x + self.mlp(x)
        class Stack(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer(), Layer()])
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        return Stack().eval()

    def test_parts_are_found_tagged_and_put_after_the_layers(self):
        model = self.model()
        layers = ["layers.0", "layers.1"]
        parts = layer_parts(model, layers)
        self.assertEqual([(p, k) for p, _l, k in parts],
                         [("layers.0.self_attn", "attention"), ("layers.0.mlp", "mlp"), ("layers.1.self_attn", "attention"), ("layers.1.mlp", "mlp")])
        recorder = SliceRecorder(max_values=10 ** 6, per_event=10 ** 6 // 2, token_features=1 << 20,
                                 parts=[p for p, _l, _k in parts], part_features=PART_FEATURES)
        _, trace = capture_forward(model, {"x": torch.randn(1, 5, 600)}, [*layers, *(p for p, _l, _k in parts)], slices=recorder)
        order_parts(trace, parts)
        self.assertEqual([e["module"] for e in trace["events"]][:2], layers)
        tail = trace["events"][2:]
        self.assertEqual([(e["part_of"], e["part"]) for e in tail],
                         [("layers.0", "attention"), ("layers.0", "mlp"), ("layers.1", "attention"), ("layers.1", "mlp")])
        # Layers keep every feature; parts keep fewer, with each token's RMS over all of them.
        self.assertEqual(trace["events"][0]["slices"]["cols"], 600)
        self.assertEqual(tail[0]["slices"]["cols"], PART_FEATURES)
        self.assertEqual(len(tail[0]["slices"]["token_rms"]), 5)
        encode_trace(trace)

    def test_a_part_that_does_not_belong_to_its_layer_is_refused(self):
        model = self.model()
        _, trace = capture_forward(model, {"x": torch.randn(1, 3, 600)}, ["layers.0", "layers.0.mlp"])
        order_parts(trace, layer_parts(model, ["layers.0"]))
        trace["events"][1]["part_of"] = "layers.1"
        with self.assertRaises(ValueError):
            encode_trace(trace)


if __name__ == "__main__":
    unittest.main()
