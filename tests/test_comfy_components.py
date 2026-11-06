import unittest

import torch

from model_run.comfy_components import encoder_section, readable_piece, real_tokens, stage_paths
from model_run.denoising import DenoisingRecorder
from model_run.trace_store import encode_trace


class Decoder(torch.nn.Module):
    """Shaped like ComfyUI's VAE decoder: plain containers whose own forward never runs."""
    def __init__(self):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(1, 1, 1)
        self.mid = torch.nn.Module()
        self.mid.block_1 = torch.nn.Conv2d(1, 1, 1)
        self.up = torch.nn.ModuleList([torch.nn.Module()])
        self.up[0].block = torch.nn.ModuleList([torch.nn.Conv2d(1, 1, 1)])
        self.head = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv2d(1, 1, 1))
        self.act = torch.nn.ReLU()
    def forward(self, x):
        return x


def base_trace(**extra):
    return {"schema_version": "1.0", "kind": "pytorch_module_capture", "model_class": "m", "torch_version": "t", "limitations": [], "status": "completed",
            "capture": {"modules": ["m"], "output_selectors": {}, "max_tensor_elements": 1, "max_events": 1}, "elapsed_host_seconds": 0, "dropped_events": 0, "events": [], **extra}


class EncoderSection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tokens = {"ids": [1, 2, 3, 4, 4], "weights": [1.0] * 5, "labels": ["<|startoftext|>", "red", "cube", "<|endoftext|>", "<|endoftext|>"]}

    def section(self):
        states = {f"layers.{i}": torch.randn(5, 16) for i in range(3)}
        for state in states.values():
            state[0] *= 1000  # a start token far stronger than any word
        return encoder_section("clip_l", states, self.tokens)

    def test_padding_is_left_out_and_tokens_keep_their_ids(self):
        section = self.section()
        self.assertEqual([t["text"] for t in section["tokens"]], ["<|startoftext|>", "red", "cube", "<|endoftext|>"])
        self.assertEqual([t["id"] for t in section["tokens"]], [1, 2, 3, 4])
        self.assertEqual([t["special"] for t in section["tokens"]], [True, False, False, True])

    def test_padding_can_be_on_either_side(self):
        self.assertEqual(real_tokens(["a", "<pad>", "<pad>"]), [0])
        self.assertEqual(real_tokens(["<pad>", "<pad>", "a", "b"]), [2, 3])  # Gemma pads on the left
        self.assertEqual(real_tokens(["a", "<|endoftext|>", "<|endoftext|>"]), [0, 1])
        self.assertEqual(real_tokens(["<pad>", "<pad>"]), [0, 1])
        self.assertEqual(readable_piece("\u0120red"), "red")
        self.assertEqual(readable_piece("red</w>"), "red")
        self.assertEqual(readable_piece("\u010a"), "\u23ce")

    def test_a_strong_start_token_does_not_choose_the_map_or_units(self):
        section = self.section()
        explained = section["map"]["explained"]
        self.assertLess(explained[0], 0.95)
        layer = section["layers"][0]
        self.assertGreater(len(layer["units"]), 0)
        self.assertEqual(len(layer["unit_values"]), 4)
        self.assertEqual(len(section["map"]["points"]), 3)
        self.assertEqual(len(section["map"]["points"][0]), 4)

    def test_tokens_that_do_not_line_up_are_numbered_not_guessed(self):
        section = encoder_section("clip_l", {"layers.0": torch.randn(3, 8), "layers.1": torch.randn(3, 8)}, self.tokens)
        self.assertTrue(section["numbered"])
        self.assertEqual([t["text"] for t in section["tokens"]], ["#0", "#1", "#2"])

    def test_section_passes_the_trace_store(self):
        encode_trace(base_trace(text_encoder={"encoders": [self.section()], "limitations": []}))

    def test_trace_store_refuses_a_map_of_the_wrong_size(self):
        section = self.section()
        section["map"]["points"].pop()
        with self.assertRaises(ValueError):
            encode_trace(base_trace(text_encoder={"encoders": [section], "limitations": []}))


class Stages(unittest.TestCase):
    def test_finds_modules_that_really_run(self):
        self.assertEqual(stage_paths(Decoder()), ["conv_in", "mid.block_1", "up.0.block.0", "head.1"])


class Format:
    latent_rgb_factors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    latent_rgb_factors_bias = None
    latent_rgb_factors_reshape = None


def step_args(sigma, *, scale=7.0, schedule=None):
    x = torch.randn(1, 4, 8, 8)
    cond = torch.randn(1, 4, 8, 8)
    uncond = torch.randn(1, 4, 8, 8)
    denoised = uncond + (cond - uncond) * scale
    return {"denoised": denoised, "input": x, "sigma": torch.tensor([sigma]), "cond_scale": scale, "cond_denoised": cond, "uncond_denoised": uncond,
            "model_options": {"transformer_options": {"sample_sigmas": schedule}} if schedule is not None else {}}


class Recorder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_returns_the_exact_tensor_it_was_given(self):
        recorder = DenoisingRecorder(Format())
        args = step_args(5.0)
        self.assertIs(recorder(args), args["denoised"])

    def test_records_what_each_step_did(self):
        updates = []
        recorder = DenoisingRecorder(Format(), on_update=updates.append)
        schedule = torch.tensor([9.0, 5.0, 1.0, 0.0])
        for sigma in (9.0, 5.0, 1.0):
            recorder(step_args(sigma, schedule=schedule))
        section = recorder.section
        self.assertEqual(section["projection"], "latent_rgb_factors")
        self.assertEqual(section["total_steps"], 3)
        self.assertEqual([s["sigma"] for s in section["steps"]], [9.0, 5.0, 1.0])
        self.assertIsNone(section["steps"][0]["estimate_change"])
        self.assertIsNotNone(section["steps"][1]["estimate_change"])
        self.assertEqual(sorted(section["steps"][0]["images"]), ["estimate", "guidance", "latent", "negative", "noise", "positive"])
        self.assertTrue(updates)
        encode_trace(base_trace(denoising=section))

    def test_no_guidance_means_no_negative_panels(self):
        recorder = DenoisingRecorder(Format())
        recorder(step_args(5.0, scale=1.0))
        step = recorder.section["steps"][0]
        self.assertNotIn("guidance_rms", step)
        self.assertEqual(sorted(step["images"]), ["estimate", "latent", "noise"])

    def test_no_factors_falls_back_and_says_so(self):
        class NoFactors:
            latent_rgb_factors = None
        recorder = DenoisingRecorder(NoFactors())
        recorder(step_args(5.0))
        self.assertEqual(recorder.section["projection"], "first_channels")
        self.assertEqual(len(recorder.section["steps"]), 1)

    def test_a_long_run_is_thinned_but_keeps_its_last_step(self):
        recorder = DenoisingRecorder(Format(), max_steps=4)
        schedule = torch.linspace(10, 0, 21)
        for i in range(20):
            recorder(step_args(float(schedule[i]) + 0.001, schedule=schedule))
        indices = [s["index"] for s in recorder.section["steps"]]
        self.assertLessEqual(len(indices), 5)
        self.assertEqual(indices[-1], 19)

    def test_a_failure_stops_recording_and_never_breaks_sampling(self):
        recorder = DenoisingRecorder(Format())
        args = {"denoised": torch.ones(1), "input": None, "sigma": None}
        self.assertIs(recorder(args), args["denoised"])
        self.assertIn("skipped", recorder.section)
        good = step_args(5.0)
        self.assertIs(recorder(good), good["denoised"])
        self.assertEqual(recorder.section["steps"], [])

    def test_trace_store_refuses_a_panel_it_does_not_know(self):
        recorder = DenoisingRecorder(Format())
        recorder(step_args(5.0))
        recorder.section["steps"][0]["images"]["script"] = "x"
        with self.assertRaises(ValueError):
            encode_trace(base_trace(denoising=recorder.section))


class LogitsAndCache(unittest.TestCase):
    def test_logits_view_can_recover_softmax_exactly(self):
        from model_run.transformers_capture import _logits_view
        logits = torch.randn(1000)
        view = _logits_view(logits)
        unit = next(g for g in view["temperatures"] if g["temperature"] == 1.0)
        top = view["top"][0]
        self.assertAlmostEqual(float(torch.softmax(logits.double(), -1).max()), float(torch.exp(torch.tensor(top["logit"], dtype=torch.double) - unit["log_total"])), places=3)
        self.assertEqual(sum(view["histogram"]["counts"]), 1000)
        encode_trace(base_trace(logits_view=view))

    def test_cache_size_counts_each_tensor_once(self):
        from model_run.transformers_capture import _cache_size
        key = torch.zeros(1, 2, 5, 8)
        size = _cache_size(((key, key.clone()), (key.clone(), key.clone())))
        self.assertEqual(size["tensors"], 4)
        self.assertEqual(size["tokens"], 5)
        self.assertEqual(size["bytes"], 4 * key.numel() * key.element_size())
        self.assertEqual(_cache_size((key, key))["tensors"], 1)
        self.assertIsNone(_cache_size(None))


if __name__ == "__main__":
    unittest.main()


class GenerationTrace(unittest.TestCase):
    class Clip:
        cond_stage_model = torch.nn.Linear(1, 1)
        def decode(self, ids, skip_special_tokens=True):
            return "".join(f"<{i}>" for i in ids)

    def test_each_written_token_gets_its_step_distribution(self):
        from model_run.comfy_components import generation_trace
        from model_run.trace_store import encode_trace
        logits = [torch.tensor([0.0, 3.0, 1.0, 0.0]), torch.tensor([2.0, 0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 0.0, 9.0])]
        seen = {"tokens": {"qwen3": [[(7, 1.0), (8, 1.0)]]}, "ids": [1, 0, 3]}
        trace = generation_trace(self.Clip(), seen, logits, max_length=3)
        self.assertEqual([s["token_id"] for s in trace["steps"]], [1, 0, 3])
        self.assertAlmostEqual(trace["steps"][0]["probability"], float(torch.softmax(logits[0], -1)[1]), places=6)
        self.assertEqual(trace["steps"][0]["top_tokens"][0]["token_id"], 1)
        self.assertEqual((trace["decoding"], trace["stopped"], trace["input"]["token_count"], trace["response"]), ("greedy", "max_new_tokens", 2, "<1><0><3>"))
        self.assertEqual([t["id"] for t in trace["tokens"]], [7, 8])
        encode_trace({**trace, "model_name": "test"})
        sampled = generation_trace(self.Clip(), seen, logits + [torch.zeros(4)], max_length=8, sampling={"temperature": 0.7, "seed": 3})
        self.assertEqual((sampled["decoding"], sampled["stopped"], sampled["sampling"]["seed"]), ("sampled", "eos", 3))
        encode_trace(sampled)

    def test_steps_that_cannot_be_paired_give_no_trace(self):
        from model_run.comfy_components import generation_trace
        self.assertIsNone(generation_trace(self.Clip(), {"ids": [1, 2]}, [torch.zeros(4)], max_length=8))
        self.assertIsNone(generation_trace(self.Clip(), {"ids": []}, [], max_length=8))
