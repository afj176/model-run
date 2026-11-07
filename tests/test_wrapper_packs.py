import copy
import sys
import unittest

try:
    import torch
except ImportError:  # these tests build PyTorch modules at import, so the whole module skips
    raise unittest.SkipTest("optional PyTorch dependency unavailable")

from model_run.comfy_bridge import DenoiserObserver
from model_run.wrapper_packs import OBSERVER_KEY, call_arguments, route_transformer


class PackTransformer(torch.nn.Module):
    """Called the way ComfyUI-WanVideoWrapper's sampler calls its WanModel: lists of latents and text, options as a keyword."""
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])

    def forward(self, x, t, context, seq_len, is_uncond=False, transformer_options={}):
        tokens = x[0].flatten(1).transpose(0, 1)  # [F*H*W, C]
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.transpose(0, 1).reshape(x[0].shape).unsqueeze(0), None, None


class Patcher:
    def __init__(self, core):
        self.model = type("Base", (), {})()
        self.model.diffusion_model = core
        self.model_options = {"transformer_options": {"merge_loras": False}}


class WanVideoWrapperTests(unittest.TestCase):
    def test_an_observed_call_is_recorded_and_returns_what_the_plain_call_does(self):
        torch.manual_seed(0)
        core = PackTransformer().eval()
        published = []
        observer = DenoiserObserver(Patcher(core), ["blocks.0", "blocks.1"], max_calls=2, elements=8, publish=published.append)
        route_transformer(core)
        route_transformer(core)  # routing twice must not stack
        x, t, context = [torch.randn(4, 1, 2, 2)], torch.tensor([500.0]), [torch.randn(6, 8)]
        plain = core(x, t, context, 4, transformer_options={"merge_loras": False})[0]
        # The sampler deep-copies transformer_options; the observer must come through as itself.
        options = copy.deepcopy({"merge_loras": False, OBSERVER_KEY: observer})
        self.assertIs(options[OBSERVER_KEY], observer)
        observed = core(x, t, context, 4, transformer_options=options)[0]
        self.assertTrue(torch.equal(plain, observed))
        self.assertEqual(len(published), 1)
        self.assertEqual([e["module"] for e in published[0]["events"]], ["blocks.0", "blocks.1"])
        self.assertEqual(observer.calls, 1)
        core(x, t, context, 4, transformer_options={"merge_loras": False})
        self.assertEqual(observer.calls, 1)  # a call without the observer is not recorded

    def test_arguments_are_put_in_comfyui_form(self):
        x, context = [torch.zeros(16, 3, 4, 4)], [torch.zeros(7, 32)]
        arguments = call_arguments((x, torch.tensor([1.0])), {"context": context, "is_uncond": True})
        self.assertEqual(list(arguments["input"].shape), [1, 16, 3, 4, 4])
        self.assertEqual(list(arguments["c"]["c_crossattn"].shape), [1, 7, 32])
        self.assertEqual(arguments["cond_or_uncond"], [1])


def attention(q, k, v, attention_mode="sdpa", heads=1, attn_mask=None):
    """Stands in for the pack's module-level attention: [batch, tokens, heads, dim] in and out."""
    out = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask)
    return out.transpose(1, 2).contiguous()


class WanBlock(torch.nn.Module):
    def __init__(self, dim=4, heads=2):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v = (torch.nn.Linear(dim, dim) for _ in range(3))

    def forward(self, tokens, text):
        split = lambda t: t.view(1, t.shape[1], self.heads, -1)
        x = split(tokens)
        tokens = tokens + attention(x, x, x, attention_mode="sdpa", heads=self.heads).flatten(2)  # self-attention
        q, k, v = split(self.q(tokens)), split(self.k(text)), split(self.v(text))
        return tokens + attention(q, k, v, attention_mode="sdpa", heads=self.heads).flatten(2)  # cross-attention to the padded text


class WanModel(torch.nn.Module):
    """Called like the pack's WanModel; pads its text to 8 keys as the pack pads to 512."""
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([WanBlock() for _ in range(2)])

    def forward(self, x, t, context, seq_len, is_uncond=False, transformer_options={}):
        tokens = x[0].flatten(1).transpose(0, 1).unsqueeze(0)  # [1, F*H*W, C]
        text = torch.cat([context[0], context[0].new_zeros(8 - context[0].shape[0], context[0].shape[1])]).unsqueeze(0)
        for block in self.blocks:
            tokens = block(tokens, text)
        return tokens[0].transpose(0, 1).reshape(x[0].shape).unsqueeze(0), None, None


class WanVideoAttentionTests(unittest.TestCase):
    def call(self, core, observer=None):
        torch.manual_seed(1)
        x, context = [torch.randn(4, 1, 4, 4)], [torch.randn(3, 4)]
        options = {} if observer is None else {OBSERVER_KEY: observer}
        with torch.no_grad():
            return core(x, torch.tensor([500.0]), context, 16, transformer_options=options)[0]

    def test_cross_attention_is_mapped_by_block_and_output_kept(self):
        torch.manual_seed(0)
        core = WanModel().eval()
        route_transformer(core)
        plain = self.call(core)
        published = []
        observer = DenoiserObserver(Patcher(core), ["blocks.1"], max_calls=1, elements=8, publish=published.append,
                                    attention_block="double:1", token_labels=["a", "cube", "</s>"])
        self.assertTrue(torch.equal(plain, self.call(core, observer)))
        section = published[0]["cross_attention"]
        self.assertEqual(len(section["events"]), 1, section.get("skipped"))
        event = section["events"][0]
        self.assertEqual((event["block"], event["attention_kind"], event["text_tokens"]), ("double:1", "cross", 3))
        self.assertEqual(section["text_labels"], ["a", "cube", "</s>"])
        self.assertIs(sys.modules[__name__].attention, attention)  # put back after the call

    def test_masking_a_word_reaches_the_kernel(self):
        torch.manual_seed(0)
        core = WanModel().eval()
        route_transformer(core)
        plain = self.call(core)
        observer = DenoiserObserver(Patcher(core), ["blocks.1"], max_calls=1, elements=8, ablate_tokens="1")
        masked = self.call(core, observer)
        self.assertGreater(float((plain - masked).abs().max()), 0)
