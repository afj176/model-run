from copy import copy
from copy import deepcopy
import unittest
try:
    import torch
except ImportError:  # these tests build PyTorch modules at import, so the whole module skips
    raise unittest.SkipTest("optional PyTorch dependency unavailable")
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


class AllLayerPlanTests(unittest.TestCase):
    def test_every_top_level_block_then_the_output(self):
        from model_run.comfy_bridge import all_layer_paths
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        class Dit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.double_blocks = torch.nn.ModuleList([Inner(), Inner()])
                self.single_blocks = torch.nn.ModuleList([torch.nn.Identity()])
                self.proj = torch.nn.ModuleList([torch.nn.Identity()])
        self.assertEqual(all_layer_paths(Dit()), ["double_blocks.0", "double_blocks.1", "single_blocks.0", ""])
        self.assertEqual(all_layer_paths(torch.nn.Linear(1, 1)), [""])

    def test_sequential_blocks_hook_a_child_that_actually_runs(self):
        from model_run.comfy_bridge import all_layer_paths
        class Unet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Identity()), torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())])
                self.middle_block = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
                self.output_blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Identity())])
        self.assertEqual(all_layer_paths(Unet()), ["input_blocks.0.0", "input_blocks.1.1", "middle_block.1", "output_blocks.0.0", ""])

    def test_a_refused_trace_is_kept_without_its_slices(self):
        patcher = Patcher()
        seen = []
        def publish(trace):
            if any("slices" in e for e in trace["events"]):
                raise ValueError("Too many slices")
            seen.append(trace)
        observer = DenoiserObserver(patcher, ["layer"], max_calls=1, elements=3, publish=publish)
        observer.trace = {"events": [{"slices": {}}], "limitations": []}
        observer._publish_safely(0)
        self.assertEqual(len(seen), 1)
        self.assertIn("dropped", seen[0]["limitations"][0])

    def test_large_models_are_sampled_and_stay_inside_the_store_limits(self):
        from model_run.comfy_bridge import all_layer_paths, all_layer_plan, MAX_LAYER_PATHS
        class Big(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(200)])
        paths = all_layer_paths(Big())
        self.assertEqual(len(paths), MAX_LAYER_PATHS)
        self.assertEqual((paths[0], paths[-2], paths[-1]), ("blocks.0", "blocks.199", ""))
        plan = all_layer_plan(paths, 32, 1024)
        self.assertLessEqual(len(paths) * plan["max_calls"], 1024)
        self.assertLessEqual(len(paths) * plan["max_calls"] * plan["elements"], 262144)

    def test_per_event_share_keeps_every_layer_recorded(self):
        from model_run.tensor_slices import SliceRecorder, validate_slices
        recorder = SliceRecorder(max_values=524288, per_event=524288 // 200)
        latent = [1, 16, 1, 16, 16]
        for _ in range(200):
            record = recorder.record(torch.ones(1, 256, 3072), latent)
            self.assertNotIn("unavailable", record)
            validate_slices(record)
        self.assertGreaterEqual(len(record["axes"][0]["indices"]), 2)
        video = SliceRecorder(per_event=1337).record(torch.ones(1, 128, 40, 64, 64), [1, 128, 40, 64, 64])
        validate_slices(video)
        self.assertGreaterEqual(len(video["axes"][1]["indices"]), 8)
        self.assertLessEqual((len(video["values"]) + len(video["aggregate"])), 1337 + 64)

    def test_text_streams_get_a_map_unless_the_count_also_fits_the_image(self):
        patcher = Patcher()
        observer = DenoiserObserver(patcher, ["layer"], max_calls=1, elements=3)
        plan = observer._token_sequence([1, 16, 16, 16], 15)
        self.assertEqual(plan(15)[1], "text token")
        self.assertIsNone(plan(77))
        self.assertIsNone(observer._token_sequence([1, 16, 16, 16], 16)(16))  # a 4 x 4 grid at factor 4

    def test_a_small_model_never_gets_more_slices_than_the_store_accepts(self):
        from model_run.tensor_slices import SliceRecorder, MAX_SLICES, validate_slices
        record = SliceRecorder(per_event=524288 // 6).record(torch.ones(1, 192, 32), [1, 16, 3, 16, 16])
        validate_slices(record)
        self.assertLessEqual(len(record["values"]) // (record["rows"] * record["cols"]), MAX_SLICES)

    def test_mixed_blocks_drop_text_tokens_only_when_the_count_fits(self):
        from model_run.tensor_slices import SliceRecorder
        recorder = SliceRecorder()
        recorder.text_tokens = 2
        tokens = torch.zeros(1, 2 + 16, 3)
        tokens[0, :2] = 99  # text must not reach the map
        record = recorder.record(tokens, [1, 16, 4, 4])
        self.assertEqual(record["layout"], "packed_tokens")
        self.assertIn("Text is assumed to come first", record["assumption"])
        self.assertEqual(max(record["values"]), 0.0)
        self.assertIsNone(recorder.record(torch.zeros(1, 2 + 15, 3), [1, 16, 4, 4]))


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

    def test_auto_measures_the_first_block_that_reads_the_prompt_and_keeps_it(self):
        patcher = Patcher()
        observer = DenoiserObserver(patcher, ["layer"], max_calls=2, elements=3, attention_block="auto", attention_grid=4, token_labels=["a", "b", "c"])
        def kernel(q, k, v, heads, *args, **kwargs):
            return v
        def denoiser(x, timestep, c_crossattn=None, transformer_options=None):
            q = torch.randn(2, 2, 3 + 16, 4)
            for index in range(2):
                observer.attention_override(kernel, q, q, q, 2, transformer_options={"block_type": "double", "block_index": index}, skip_reshape=True)
            return patcher.model.diffusion_model(x)
        context = torch.zeros(2, 3, 8)
        for call in range(3):  # as above, a call's record is kept when the next one starts
            observer(denoiser, {"input": torch.ones(2, 4, 4, 4), "timestep": None,
                                "c": {"c_crossattn": context, "transformer_options": {}}, "cond_or_uncond": [1, 0]})
        section = observer.trace["cross_attention"]
        self.assertEqual(section["selection"], "double:0")
        self.assertEqual([e["block"] for e in section["events"]], ["double:0", "double:0"])
        self.assertTrue(any("chosen automatically" in item for item in section["limitations"]))

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
    def test_unlabelled_joint_attention_is_found_by_its_shape(self):
        patcher = Patcher()
        observer = DenoiserObserver(patcher, ["layer"], max_calls=1, elements=3, attention_block="call:0")
        def denoiser(x, timestep, c_crossattn=None, transformer_options=None):
            q = torch.randn(1, 2, 3 + 16, 4)
            observer.attention_override(lambda q, k, v, h, **kw: v, q, q, q, 2, transformer_options={})
            return patcher.model.diffusion_model(x)
        observer(denoiser, {"input": torch.ones(1, 4, 4, 4), "timestep": None, "c": {"c_crossattn": torch.zeros(1, 3, 8)}})
        self.assertEqual(observer.trace["cross_attention"]["events"][0]["attention_kind"], "joint")

    def test_models_whose_text_comes_last_are_measured_there(self):
        class MOSSImage(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(4, 4)
            def forward(self, value):
                return self.layer(value)
        patcher = Patcher()
        patcher.model.diffusion_model = MOSSImage().eval()
        observer = DenoiserObserver(patcher, ["layer"], max_calls=1, elements=3)
        self.assertEqual(observer._text_start, -1)

    def test_mochi_counts_only_the_prompt_tokens_it_keeps(self):
        class AsymmDiTJoint(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(4, 4)
            def forward(self, value):
                return self.layer(value)
        patcher = Patcher()
        patcher.model.diffusion_model = AsymmDiTJoint().eval()
        observer = DenoiserObserver(patcher, ["layer"], max_calls=1, elements=3, attention_block="call:0")
        arguments = {"input": torch.zeros(1, 12, 2, 4, 4), "c": {"c_crossattn": torch.zeros(1, 256, 8), "num_tokens": 11}}
        self.assertEqual(observer._attention_context(0, arguments)["text_tokens"], 11)
        del arguments["c"]["num_tokens"]
        self.assertEqual(observer._attention_context(0, arguments)["text_tokens"], 256)

    def test_unet_labels_count_as_checked_by_their_text_width(self):
        class Attention(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.to_k = torch.nn.Linear(width, 4)
        class Block(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.attn2 = Attention(width)
                self.layer = torch.nn.Linear(4, 4)
        class UNetModel(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.block = Block(width)
        for width, checked in ((1024, True), (1280, True), (4096, False)):  # SD 2.x, SDXL Refiner, Kolors
            patcher = Patcher()
            patcher.model.diffusion_model = UNetModel(width).eval()
            observer = DenoiserObserver(patcher, ["block.layer"], max_calls=1, elements=3)
            self.assertEqual(observer._labels_checked_family(), checked, width)

    def test_hunyuan_video_15_image_prompt_tokens_come_before_the_text(self):
        class HunyuanVideo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(4, 4)
                self.vision_in = torch.nn.Identity()
            def forward(self, value):
                return self.layer(value)
        patcher = Patcher()
        patcher.model.diffusion_model = HunyuanVideo().eval()
        observer = DenoiserObserver(patcher, ["layer"], max_calls=1, elements=3, attention_block="double:0")
        arguments = {"input": torch.zeros(1, 32, 1, 4, 4), "c": {"c_crossattn": torch.zeros(1, 2, 8), "clip_fea": torch.zeros(1, 3, 8)}}
        current = observer._attention_context(0, arguments)
        self.assertEqual((current["text_tokens"], current["text_lead"]), (2, 3))
        # Keys run three SigCLIP tokens, two prompt tokens, then a 2x2 image.
        q = torch.zeros(1, 1, 9, 4)
        k = torch.zeros(1, 1, 9, 4)
        k[0, 0, 4, 0] = 8.0  # the second prompt token
        q[0, 0, 5:, 0] = 1.0
        observer._current = current
        observer.attention_override(lambda *args, **kwargs: None, q, k, k, 1, transformer_options={"block_type": "double", "block_index": 0})
        self.assertIsNone(current["reason"])
        self.assertIsNotNone(current["record"])

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
        self.assertEqual(attempts[:2], [1, 0])  # retried without the map, then without slices, then gave up quietly
        self.assertTrue(any("fits" in s["reason"] for s in observer.trace["cross_attention"]["skipped"]))

    def test_section_value_budget(self):
        from model_run import capture_limits
        patcher = Patcher()
        with capture_limits.override(cross_attention_values=3 * 16):
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


class ReplacedAttentionTests(unittest.TestCase):
    def test_a_patch_that_bypasses_the_override_is_routed_through_it_for_one_call(self):
        import sys
        import types
        from model_run.comfy_bridge import reach_replaced_attention
        calls = []
        def optimized_attention(q, k, v, heads, *args, **kwargs):
            calls.append(("kernel", "transformer_options" in kwargs, kwargs.get("mask")))
            return q
        attention = types.ModuleType("comfy.ldm.modules.attention"); attention.optimized_attention = optimized_attention
        plugin = types.ModuleType("fake_ipadapter"); plugin.optimized_attention = optimized_attention
        def patch(q, k, v, extra_options):
            return plugin.optimized_attention(q, k, v, extra_options["n_heads"])
        patch.__module__ = "fake_ipadapter"
        stubs = {"comfy": types.ModuleType("comfy"), "comfy.ldm": types.ModuleType("comfy.ldm"),
                 "comfy.ldm.modules": types.ModuleType("comfy.ldm.modules"), "comfy.ldm.modules.attention": attention, "fake_ipadapter": plugin}
        saved = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        try:
            def override(func, q, k, v, heads, *args, **kwargs):
                calls.append(("override", kwargs["transformer_options"]["block"]))
                return func(q, k, v, heads, *args, mask="masked")
            options = {"patches_replace": {"attn2": {("output", 4, 0): patch}}}
            with reach_replaced_attention(options, override) as routed:
                routed["patches_replace"]["attn2"][("output", 4, 0)](torch.ones(1), None, None, {"n_heads": 2, "block": ("output", 4)})
                self.assertIs(options["patches_replace"]["attn2"][("output", 4, 0)], patch)  # the caller's options are not changed
            self.assertEqual(calls, [("override", ("output", 4)), ("kernel", False, "masked")])
            self.assertIs(plugin.optimized_attention, optimized_attention)
            with reach_replaced_attention({}, override) as untouched:
                self.assertEqual(untouched, {})
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
