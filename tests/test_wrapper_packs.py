import copy
import unittest

import torch

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
