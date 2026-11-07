import unittest

try:
    import torch
except ImportError:  # these tests build PyTorch modules at import, so the whole module skips
    raise unittest.SkipTest("optional PyTorch dependency unavailable")

from model_run.comfy_bridge import DenoiserObserver
from model_run.wrapper_packs import ARMED, arm_transformer, route_transformer


def attention(q, k, v, heads=1, mode="sdpa", attn_mask=None, **_unused):
    """Stands in for HunyuanVideoWrapper's module-level attention: [batch, tokens, heads, dim] in, [batch, tokens, heads*dim] out."""
    out = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask)
    return out.transpose(1, 2).flatten(2)


KERNEL = attention


class HyBlock(torch.nn.Module):
    """Joint attention over the image, then the text, as the pack's double and single blocks order them."""
    def __init__(self, dim=4, heads=2):
        super().__init__()
        self.heads = heads
        self.qkv = torch.nn.Linear(dim, 3 * dim)

    def forward(self, tokens):
        q, k, v = (t.reshape(1, tokens.shape[1], self.heads, -1) for t in self.qkv(tokens).chunk(3, -1))
        return tokens + attention(q, k, v, heads=self.heads, mode="sdpa")


class HYVideoDiffusionTransformer(torch.nn.Module):
    """Called like the pack's transformer: a latent tensor, text_states by keyword, a dict back."""
    def __init__(self):
        super().__init__()
        self.double_blocks = torch.nn.ModuleList([HyBlock() for _ in range(2)])
        self.single_blocks = torch.nn.ModuleList([HyBlock() for _ in range(2)])

    def forward(self, x, t, text_states=None, is_uncond=False, return_dict=True):
        image = x[0].flatten(1).transpose(0, 1).unsqueeze(0)
        tokens = torch.cat([image, text_states], dim=1)
        for block in [*self.double_blocks, *self.single_blocks]:
            tokens = block(tokens)
        return {"x": tokens[0, :image.shape[1]].transpose(0, 1).reshape(x[0].shape).unsqueeze(0)}


class Patcher:
    def __init__(self, core):
        self.model = type("Base", (), {})()
        self.model.diffusion_model = core
        self.model_options = {}


class HunyuanVideoWrapperTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.core = HYVideoDiffusionTransformer().eval()

    def call(self):
        torch.manual_seed(1)
        with torch.no_grad():
            return self.core(torch.randn(1, 4, 1, 4, 4), torch.tensor([500.0]), text_states=torch.randn(1, 3, 4), is_uncond=False)["x"]

    def test_an_armed_call_is_recorded_by_block_and_output_kept(self):
        plain = self.call()
        published = []
        observer = DenoiserObserver(Patcher(self.core), ["single_blocks.1"], max_calls=1, elements=8, publish=published.append,
                                    attention_block="single:1", token_labels=["a", "cube", "eot"])
        arm_transformer(self.core, observer)  # outside ComfyUI no prompt runs, so the armed prompt is None too
        self.assertTrue(torch.equal(plain, self.call()))
        section = published[0]["cross_attention"]
        self.assertEqual(len(section["events"]), 1, section.get("skipped"))
        event = section["events"][0]
        self.assertEqual((event["block"], event["attention_kind"], event["text_tokens"]), ("single:1", "joint", 3))
        self.assertEqual(section["text_labels"], ["a", "cube", "eot"])
        self.assertIs(globals()["attention"], KERNEL)  # put back after the call

    def test_a_call_from_another_prompt_is_not_recorded(self):
        observer = DenoiserObserver(Patcher(self.core), ["double_blocks.0"], max_calls=1, elements=8)
        route_transformer(self.core)
        setattr(self.core, ARMED, (observer, "an earlier prompt"))
        self.call()
        self.assertEqual(observer.calls, 0)

    def test_masking_a_word_reaches_the_kernel(self):
        plain = self.call()
        arm_transformer(self.core, DenoiserObserver(Patcher(self.core), ["double_blocks.0"], max_calls=1, elements=8, ablate_tokens="1"))
        self.assertGreater(float((plain - self.call()).abs().max()), 0)
