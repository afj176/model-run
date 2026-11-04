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
