from copy import copy
from copy import deepcopy
import unittest
import torch
from model_run.comfy_bridge import DenoiserObserver
from model_run.trace_store import encode_trace


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)
    def forward(self, value):
        return self.layer(value)


class Patcher:
    def __init__(self):
        self.model = type("Base", (), {})()
        self.model.diffusion_model = Core().eval()
        self.model_options = {}


class ComfyBridgeTests(unittest.TestCase):
    def test_reused_module_respects_aggregate_value_budget(self):
        patcher = Patcher()
        observer = DenoiserObserver(patcher, ["layer"], max_calls=2, elements=3)
        def repeated(x, timestep):
            for _ in range(5):
                x = patcher.model.diffusion_model(x)
            return x
        args = {"input": torch.ones(1, 4), "timestep": None, "c": {}}
        expected = repeated(args["input"], None)
        for _ in range(2):
            torch.testing.assert_close(observer(repeated, args), expected, rtol=0, atol=0)
        self.assertEqual(sum(len(e["values"]) for e in observer.trace["events"]), 6)
        self.assertEqual(observer.trace["dropped_events"], 8)

    def test_native_wrapper_composition_and_budget(self):
        patcher = Patcher()
        called, published = [], []
        def previous(fn, args):
            called.append(True)
            return fn(args["input"], args["timestep"], **args["c"]) + 2
        patcher.model_options["model_function_wrapper"] = previous
        observer = DenoiserObserver(patcher, ["layer"], max_calls=2, elements=3, publish=lambda t: published.append(encode_trace(t)))
        args = {"input": torch.ones(1, 4), "timestep": torch.ones(1), "c": {"scale": 3}}
        def forward(x, timestep, scale): return patcher.model.diffusion_model(x) * scale
        expected = previous(forward, args)
        for _ in range(4):
            torch.testing.assert_close(observer(forward, args), expected, rtol=0, atol=0)
        self.assertEqual(len(published), 2)
        self.assertEqual(len(observer.trace["events"]), 2)
        self.assertEqual(observer.trace["events"][1]["denoiser_call"], 1)
        self.assertEqual(len(patcher.model.diffusion_model.layer._forward_hooks), 0)
        self.assertIs(patcher.model_options["model_function_wrapper"], previous)
        self.assertIs(deepcopy({"wrapper": observer})["wrapper"], observer)

    def test_multimodal_output_selection(self):
        class Pair(torch.nn.Module):
            def forward(self, value): return value + 1, value + 2
        patcher = Patcher()
        patcher.model.diffusion_model.layer = Pair().eval()
        observer = DenoiserObserver(patcher, ["layer"], output_selectors={"layer": [1]})
        value = torch.zeros(1, 4)
        output = observer(lambda x, t: patcher.model.diffusion_model(x), {"input": value, "timestep": None, "c": {}})
        self.assertEqual(observer.trace["events"][0]["values"], [2., 2., 2., 2.])
        self.assertEqual(observer.trace["events"][0]["output_path"], [1])
        torch.testing.assert_close(output[0], value + 1)

    def test_failure_cleanup_and_invalid_selection(self):
        patcher = Patcher()
        with self.assertRaises(ValueError): DenoiserObserver(patcher, ["missing"])
        with self.assertRaises(ValueError): DenoiserObserver(patcher, ["", "layer", "layer"], max_calls=32, elements=4096)
        observer = DenoiserObserver(patcher, ["layer"])
        def fail(*args, **kwargs): raise ValueError("fixture failure")
        with self.assertRaises(ValueError): observer(fail, {"input": None, "timestep": None, "c": {}})
        self.assertEqual(len(patcher.model.diffusion_model.layer._forward_hooks), 0)


if __name__ == "__main__": unittest.main()


@unittest.skipIf(torch is None, "optional torch dependency unavailable")
class CrossAttentionOverrideTests(unittest.TestCase):
    def test_override_measures_selected_block_and_passes_through(self):
        patcher = Patcher()
        published = []
        observer = DenoiserObserver(patcher, ["layer"], max_calls=2, elements=3, publish=lambda t: published.append(encode_trace(t)),
                                    attention_block="double:1", attention_grid=4, token_labels=["a", "b", "c"])
        calls = []
        def kernel(q, k, v, heads, *args, **kwargs):
            calls.append(kwargs.get("transformer_options", {}).get("block_index"))
            return v
        def denoiser(x, timestep, c_crossattn=None, transformer_options=None):
            q = torch.randn(2, 2, 3 + 16, 4)
            for index in range(2):
                options = {"block_type": "double", "block_index": index}
                observer.attention_override(kernel, q, q, q, 2, transformer_options=options, skip_reshape=True)
            return patcher.model.diffusion_model(x)
        context = torch.zeros(2, 3, 8)
        for call in range(3):
            observer(denoiser, {"input": torch.ones(2, 4, 4, 4), "timestep": None,
                                "c": {"c_crossattn": context, "transformer_options": {}}, "cond_or_uncond": [1, 0]})
        self.assertEqual(calls, [0, 1] * 3)
        section = observer.trace["cross_attention"]
        self.assertEqual(section["text_labels"], ["a", "b", "c"])
        self.assertEqual([e["denoiser_call"] for e in section["events"]], [0, 1])
        event = section["events"][0]
        self.assertEqual((event["block"], event["attention_kind"], event["grid"], event["batch"]), ("double:1", "joint", [4, 4], "conditional (cond_or_uncond)"))
        self.assertEqual(len(event["maps"]), 3)
        self.assertNotIn("skipped", section)
        self.assertIsNone(observer._current)
        self.assertEqual(len(published), 2)

    def test_override_reports_unsupported_geometry(self):
        patcher = Patcher()
        observer = DenoiserObserver(patcher, ["layer"], max_calls=1, elements=3, attention_block="output:1")
        def denoiser(x, timestep, c_crossattn=None, transformer_options=None):
            q = torch.randn(1, 7, 8)
            observer.attention_override(lambda q, k, v, h, **kw: v, q, q, q, 2, transformer_options={"block": ("output", 1), "block_index": 0})
            return patcher.model.diffusion_model(x)
        observer(denoiser, {"input": torch.ones(1, 4, 4, 4), "timestep": None, "c": {"c_crossattn": torch.zeros(1, 3, 8)}})
        section = observer.trace["cross_attention"]
        self.assertEqual(section["events"], [])
        self.assertEqual(len(section["skipped"]), 1)
        encode_trace(observer.trace)
        with self.assertRaises(ValueError): DenoiserObserver(patcher, ["layer"], attention_block="")


@unittest.skipIf(torch is None, "optional torch dependency unavailable")
class PublishSafetyTests(unittest.TestCase):
    def test_publish_failure_never_breaks_sampling(self):
        patcher = Patcher()
        attempts = []
        def failing_publish(trace):
            attempts.append(len(trace.get("cross_attention", {}).get("events", [])))
            raise ValueError("Trace exceeds the size bound")
        observer = DenoiserObserver(patcher, ["layer"], max_calls=1, elements=3, publish=failing_publish, attention_block="double:0")
        def denoiser(x, timestep, c_crossattn=None, transformer_options=None):
            q = torch.randn(1, 2, 3 + 16, 4)
            observer.attention_override(lambda q, k, v, h, **kw: v, q, q, q, 2, transformer_options={"block_type": "double", "block_index": 0})
            return patcher.model.diffusion_model(x)
        output = observer(denoiser, {"input": torch.ones(1, 4, 4, 4), "timestep": None, "c": {"c_crossattn": torch.zeros(1, 3, 8)}})
        self.assertIsNotNone(output)
        self.assertEqual(attempts, [1, 0])  # retried once without the map, then gave up quietly
        self.assertTrue(any("fits" in s["reason"] for s in observer.trace["cross_attention"]["skipped"]))

    def test_section_value_budget(self):
        from model_run import comfy_bridge
        patcher = Patcher()
        original = comfy_bridge.MAX_SECTION_VALUES
        comfy_bridge.MAX_SECTION_VALUES = 3 * 16
        try:
            observer = DenoiserObserver(patcher, ["layer"], max_calls=3, elements=3, attention_block="double:0")
            def denoiser(x, timestep, c_crossattn=None, transformer_options=None):
                q = torch.randn(1, 2, 3 + 16, 4)
                observer.attention_override(lambda q, k, v, h, **kw: v, q, q, q, 2, transformer_options={"block_type": "double", "block_index": 0})
                return patcher.model.diffusion_model(x)
            for _ in range(2):
                observer(denoiser, {"input": torch.ones(1, 4, 4, 4), "timestep": None, "c": {"c_crossattn": torch.zeros(1, 3, 8)}})
            section = observer.trace["cross_attention"]
            self.assertEqual(len(section["events"]), 1)
            self.assertIn("budget", section["skipped"][0]["reason"])
            encode_trace(observer.trace)
        finally:
            comfy_bridge.MAX_SECTION_VALUES = original



@unittest.skipIf(torch is None, "optional torch dependency unavailable")
class AttachOutputTests(unittest.TestCase):
    def observed(self, published):
        patcher = Patcher()
        observer = DenoiserObserver(patcher, ["layer"], max_calls=1, elements=3, publish=lambda t: published.append(encode_trace(t)))
        observer(lambda x, t: patcher.model.diffusion_model(x), {"input": torch.ones(1, 4), "timestep": None, "c": {}})
        return patcher, observer

    def test_single_image_and_video_frames(self):
        from model_run.comfy_bridge import ModelRunAttachOutput
        published = []
        patcher, observer = self.observed(published)
        patcher.model_options["model_function_wrapper"] = observer
        image = torch.rand(1, 32, 32, 3)
        out = ModelRunAttachOutput().attach(patcher, image)
        self.assertIs(out[0], image)
        self.assertEqual(observer.trace["preview"]["mime"], "image/png")
        self.assertNotIn("preview_frames", observer.trace)
        ModelRunAttachOutput().attach(patcher, torch.rand(40, 32, 32, 3))
        frames = observer.trace["preview_frames"]
        self.assertEqual((len(frames["frames"]), frames["frame_indices"][0], frames["frame_indices"][-1], frames["source_frames"]), (16, 0, 39, 40))
        encode_trace(observer.trace)

    def test_oversized_output_degrades_without_failing(self):
        patcher = Patcher()
        attempts = []
        def publish(trace):
            attempts.append(("preview_frames" in trace, "preview" in trace))
            if "preview_frames" in trace or len(attempts) == 1:
                raise ValueError("too big")
        observer = DenoiserObserver(patcher, ["layer"], max_calls=1, elements=3, publish=publish)
        observer.publish = None
        observer(lambda x, t: patcher.model.diffusion_model(x), {"input": torch.ones(1, 4), "timestep": None, "c": {}})
        observer.publish = publish
        self.assertFalse(observer.attach_images(torch.rand(3, 8, 8, 3)))
        self.assertEqual(attempts, [(True, True), (False, True)])

    def test_unobserved_model_passes_images_through(self):
        from model_run.comfy_bridge import ModelRunAttachOutput
        image = torch.rand(1, 8, 8, 3)
        self.assertIs(ModelRunAttachOutput().attach(Patcher(), image)[0], image)
