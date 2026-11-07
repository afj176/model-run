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


    def test_image_blocks_are_measured_even_when_attention_is_called_in_pieces(self):
        from model_run.layer_parts import block_part_meter
        class SelfAttention(torch.nn.Module):  # like Flux: the block calls qkv and proj, never forward
            def __init__(self):
                super().__init__()
                self.qkv = torch.nn.Linear(4, 12)
                self.proj = torch.nn.Linear(4, 4)
        class DoubleBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.img_attn = SelfAttention()
                self.img_mlp = torch.nn.Sequential(torch.nn.Linear(4, 4))
            def forward(self, img):
                q, k, v = self.img_attn.qkv(img).chunk(3, dim=-1)
                img = img + self.img_attn.proj(torch.softmax(q @ k.transpose(-1, -2), -1) @ v)
                # Text first, as Qwen-Image does: the larger image stream is the block's output.
                return torch.full((1, 2, 4), 1e4), img + self.img_mlp(img)
        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.double_blocks = torch.nn.ModuleList([DoubleBlock(), DoubleBlock()])
                self.final = torch.nn.Linear(4, 4)
            def forward(self, img):
                for block in self.double_blocks:
                    img = block(img)[1]
                return self.final(img)
        core = Core().eval()
        results, remove = block_part_meter(core, ["double_blocks.0", "double_blocks.1", "final", ""])
        with torch.no_grad():
            core(torch.randn(1, 6, 4))
            core(torch.randn(1, 6, 4) * 100)  # a later call does not change what was measured
        remove()
        found = results()
        self.assertEqual([b["block"] for b in found], ["double_blocks.0", "double_blocks.1"])
        self.assertEqual([p["kind"] for p in found[0]["parts"]], ["attention", "mlp"])
        self.assertTrue(all(0 < p["rms"] < 50 for p in found[0]["parts"]))
        self.assertLess(found[0]["output_rms"], 1000)  # the image stream, not the text
        # What the block changed: its attention and MLP added together, matched to its input by shape.
        self.assertGreater(found[0]["update_rms"], 0)
        self.assertLess(found[0]["update_rms"], found[0]["output_rms"] * 10)
        trace = {"schema_version": "1.0", "kind": "pytorch_module_capture", "model_class": "Core", "torch_version": "2",
                 "events": [], "capture": {}, "limitations": [], "block_parts": found}
        encode_trace(trace)
        found[0]["parts"][0]["path"] = "elsewhere.attn"
        with self.assertRaises(ValueError):
            encode_trace(trace)

    def test_a_vit_layer_keeps_every_patch_and_its_maps(self):
        recorder = SliceRecorder(token_grid=(1, 4, 4), patch_features=1 << 20, parts=["layers.0.attention"], part_features=PART_FEATURES)
        tensor = torch.randn(1, 17, 48)
        record = recorder.record(tensor, None, path="layers.0")
        self.assertEqual(record["layout"], "patch_tokens")  # the patch maps are still there
        patches = record["patches"]
        self.assertEqual((patches["rows"], patches["cols"], patches["patch_grid"]), (17, 48, [1, 4, 4]))
        self.assertEqual(patches["row_labels"][:3], ["CLS", "r0 c0", "r0 c1"])
        self.assertEqual(patches["row_labels"][-1], "r3 c3")
        part = recorder.record(tensor, None, path="layers.0.attention")
        self.assertEqual((part["layout"], part["rows"], part["patch_grid"]), ("tokens", 17, [1, 4, 4]))
        from model_run.tensor_slices import validate_slices
        validate_slices(record)
        validate_slices(part)
        record["patches"]["patch_grid"] = [1, 4, 5]
        with self.assertRaises(ValueError):
            validate_slices(record)


if __name__ == "__main__":
    unittest.main()
