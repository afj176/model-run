import sys
import types
import unittest

try:
    import torch
except ImportError:
    torch = None

from model_run.trace_store import encode_trace


@unittest.skipIf(torch is None, "optional PyTorch dependency unavailable")
class AttentionWatchTests(unittest.TestCase):
    def setUp(self):
        # A stand-in for a ComfyUI text model module: it looks up the attention factory by name when it runs.
        fake = types.ModuleType("comfy.text_encoders.fake_watch")
        def optimized_attention_for_device(device, mask=False, small_input=True):
            def kernel(q, k, v, heads, mask=None, skip_reshape=False, scale=None, **_):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale).transpose(1, 2).flatten(2)
            return kernel
        fake.optimized_attention_for_device = optimized_attention_for_device
        sys.modules[fake.__name__] = fake
        self.addCleanup(sys.modules.pop, fake.__name__)
        self.fake = fake

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Linear(8, 8)
                self.k = torch.nn.Linear(8, 4)  # two query heads share each key head
                self.v = torch.nn.Linear(8, 4)
            def forward(self, x, mask):
                b, t, _ = x.shape
                q = self.q(x).view(b, t, 4, 2).transpose(1, 2)
                k = self.k(x).view(b, t, 2, 2).transpose(1, 2).repeat_interleave(2, dim=1)
                v = self.v(x).view(b, t, 2, 2).transpose(1, 2).repeat_interleave(2, dim=1)
                kernel = fake.optimized_attention_for_device(x.device, mask=mask is not None)
                return kernel(q, k, v, 4, mask=mask, skip_reshape=True)
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()
            def forward(self, x, mask):
                return x + self.self_attn(x, mask)
        class Stack(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer(), Layer()])
            def forward(self, x, mask):
                for layer in self.layers:
                    x = layer(x, mask)
                return x
        torch.manual_seed(0)
        self.stack = Stack().eval()

    def causal(self, t):
        return torch.full((t, t), -1e9).triu(1)[None, None]

    def test_weights_match_the_kernel_and_the_output_is_unchanged(self):
        from model_run.comfy_attention import AttentionWatch, attention_trace
        x = torch.randn(1, 5, 8)
        with torch.no_grad():
            before = self.stack(x, self.causal(5))
            watch = AttentionWatch(self.stack, ["layers.0.self_attn", "layers.1.self_attn"])
            with watch.watching():
                during = self.stack(x, self.causal(5))
                self.stack(x[:, :1], None)  # a generated token: one query, not recorded again
        self.assertTrue(torch.equal(before, during))
        self.assertIs(self.fake.optimized_attention_for_device.__name__, "optimized_attention_for_device")
        self.assertEqual(sorted(watch.found), ["layers.0.self_attn", "layers.1.self_attn"])
        # Reference: the first layer's weights from its own query and key.
        attention = self.stack.layers[0].self_attn
        q = attention.q(x).view(1, 5, 4, 2).transpose(1, 2)
        k = attention.k(x).view(1, 5, 2, 2).transpose(1, 2).repeat_interleave(2, dim=1)
        reference = torch.softmax(q @ k.transpose(-1, -2) / 2 ** 0.5 + self.causal(5), dim=-1)[0]
        self.assertTrue(torch.allclose(watch.found["layers.0.self_attn"], reference, atol=1e-5))
        self.assertEqual(float(watch.found["layers.0.self_attn"][0, 0, 1]), 0.0)  # causal
        trace = attention_trace(watch, [{"id": 7 + i, "text": w} for i, w in enumerate(["<bos>", "Walk", "the", "park", "."])], "Stack")
        self.assertEqual((trace["layer_count"], trace["head_count"], trace["shape"]), (2, 4, [5, 5]))
        self.assertEqual(trace["tokens"][1], {"id": 8, "text": "Walk"})
        self.assertEqual(len(trace["layer_stacks"]), 2)
        self.assertEqual(trace.pop("stack"), "layers")
        encode_trace(trace)

    def test_nothing_is_recorded_outside_the_watch(self):
        from model_run.comfy_attention import AttentionWatch
        watch = AttentionWatch(self.stack, ["layers.0.self_attn"])
        with torch.no_grad():
            self.stack(torch.randn(1, 3, 8), self.causal(3))
        self.assertEqual(watch.found, {})


if __name__ == "__main__":
    unittest.main()
