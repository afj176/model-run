import unittest

try:
    import torch
except ImportError:
    torch = None

from model_run.cross_attention import parse_token_list, text_key_range


class ParsingTests(unittest.TestCase):
    def test_parse_and_ranges(self):
        self.assertEqual(parse_token_list("5, 2 5"), [2, 5])
        self.assertEqual(parse_token_list([3, 1]), [1, 3])
        self.assertEqual(parse_token_list(None), [])
        with self.assertRaises(ValueError): parse_token_list("a")
        with self.assertRaises(ValueError): parse_token_list([2000])
        self.assertEqual(text_key_range(256, 77, 77), (0, 77))
        self.assertEqual(text_key_range(512 + 256, 512 + 256, 512, allow_joint=True), (0, 512))
        self.assertIsNone(text_key_range(256, 256, 77))


@unittest.skipIf(torch is None, "optional torch dependency unavailable")
class KeyMaskTests(unittest.TestCase):
    def test_mask_removes_keys_and_folds_existing(self):
        from model_run.cross_attention import key_mask, MASK_BIAS
        bias = key_mask([1, 3], 5, device="cpu", dtype=torch.float32)
        self.assertEqual(list(bias.shape), [1, 1, 1, 5])
        self.assertEqual(bias[0, 0, 0].tolist(), [0.0, MASK_BIAS, 0.0, MASK_BIAS, 0.0])
        existing = torch.tensor([[True, True, True, True, False]])
        folded = key_mask([1], 5, existing=existing, device="cpu", dtype=torch.float32)
        self.assertEqual(folded[0, 0, 0].tolist(), [0.0, MASK_BIAS, 0.0, 0.0, MASK_BIAS])
        joint = key_mask([0], 10, start=3, device="cpu", dtype=torch.float32)
        self.assertEqual(joint[0, 0, 0, 3].item(), MASK_BIAS)
        self.assertIsNone(key_mask([], 5))
        # A masked key gets no attention weight.
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 5, 8)
        scores = (q @ k.transpose(-1, -2)) / 8 ** 0.5 + bias
        probabilities = torch.softmax(scores, dim=-1)
        self.assertLess(probabilities[..., [1, 3]].max().item(), 1e-4)

    def test_comfy_override_masks_text_keys_every_call(self):
        from model_run.comfy_bridge import DenoiserObserver
        from tests.test_comfy_bridge import Patcher
        patcher = Patcher()
        observer = DenoiserObserver(patcher, ["layer"], max_calls=1, elements=3, ablate_tokens="1", token_labels=["a", "b", "c"])
        seen = []
        def kernel(q, k, v, heads, mask=None, **kwargs):
            seen.append(None if mask is None else mask[0, 0, 0].tolist())
            return v
        def denoiser(x, timestep, c_crossattn=None, transformer_options=None):
            q = torch.randn(1, 2, 3 + 16, 4)
            observer.attention_override(kernel, q, q, q, 2, transformer_options={"block_type": "double", "block_index": 0})
            return patcher.model.diffusion_model(x)
        for _ in range(3):  # beyond max_calls the intervention still applies
            observer(denoiser, {"input": torch.ones(1, 4, 4, 4), "timestep": None, "c": {"c_crossattn": torch.zeros(1, 3, 8)}})
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(m is not None and m[1] < -1000 and m[0] == 0 and m[3] == 0 for m in seen))
        block = observer.trace["token_ablation"]
        self.assertEqual((block["tokens"], block["labels"]), ([1], ["b"]))
        self.assertEqual(block["applied_attention_calls"], 1)  # counted within the recorded call; the mask list above proves later calls
        from model_run.trace_store import encode_trace
        encode_trace(observer.trace)

    def test_diffusers_processor_mask(self):
        try:
            from diffusers.models.attention_processor import Attention
        except ImportError:
            self.skipTest("diffusers unavailable")
        from model_run.diffusion_capture import TokenAblation
        attn = Attention(query_dim=8, cross_attention_dim=6, heads=2, dim_head=4).eval()
        class Pipe:
            unet = torch.nn.Sequential(attn)
        hidden = torch.randn(2, 5, 8)
        context = torch.randn(2, 3, 6)
        with torch.inference_mode():
            baseline = attn(hidden, encoder_hidden_states=context)
            # Removing key 1 by hand: run with only keys 0 and 2 present.
            reduced = attn(hidden, encoder_hidden_states=context[:, [0, 2]])
        ablation = TokenAblation(Pipe(), [1], token_labels=["x", "y", "z"])
        ablation.install()
        try:
            with torch.inference_mode():
                masked = attn(hidden, encoder_hidden_states=context)
        finally:
            ablation.remove()
        torch.testing.assert_close(masked, reduced, atol=1e-4, rtol=1e-3)
        with torch.inference_mode():
            torch.testing.assert_close(attn(hidden, encoder_hidden_states=context), baseline)
        section = ablation.section()
        self.assertEqual((section["tokens"], section["labels"], section["applied_attention_calls"]), ([1], ["y"], 1))
