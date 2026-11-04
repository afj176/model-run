import unittest

try:
    import torch
except ImportError:
    torch = None

from model_run.capture import capture_forward
from model_run.control import RunControl, ObservationCancelled


@unittest.skipIf(torch is None, "optional PyTorch dependency unavailable")
class CaptureTests(unittest.TestCase):
    def test_cancelled_partial_trace_and_hook_cleanup(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Identity()
                self.second = torch.nn.Identity()
                self.second_called = False
            def forward(self, x):
                x = self.first(x)
                self.second_called = True
                return self.second(x)
        model = Model().eval()
        control = RunControl(on_progress=lambda event: control.cancel() if event["stage"] == "module:first" else None)
        with self.assertRaises(ObservationCancelled) as result:
            capture_forward(model, {"x": torch.ones(2)}, ["first", "second"], control=control)
        self.assertEqual(result.exception.trace["status"], "cancelled")
        self.assertEqual(len(result.exception.trace["events"]), 1)
        self.assertFalse(model.second_called)
        for layer in (model.first, model.second):
            self.assertFalse(layer._forward_hooks)
            self.assertFalse(layer._forward_pre_hooks)
        capture_forward(model, {"x": torch.ones(2)}, ["first"])
    def test_multimodal_output_selection(self):
        class Multimodal(torch.nn.Module):
            def forward(self, x):
                return {"video": x + 1, "audio": (None, x + 2)}
        model = Multimodal().eval()
        _, trace = capture_forward(model, {"x": torch.ones(2)}, [""], output_selectors={"": ["audio", 1]})
        self.assertEqual(trace["events"][0]["output_path"], ["audio", 1])
        self.assertEqual(trace["events"][0]["values"], [3., 3.])
        with self.assertRaises(ValueError):
            capture_forward(model, {"x": torch.ones(2)}, [""], output_selectors={"": ["audio", 9]})
        self.assertEqual(len(model._forward_hooks), 0)

    def test_matches_reference_and_removes_hooks(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(3, 2)
            def forward(self, x):
                return self.layer(x)
        model = Model().eval()
        inputs = {"x": torch.tensor([[1.0, 2.0, 3.0]])}
        with torch.inference_mode(): expected = model(**inputs)
        output, trace = capture_forward(model, inputs, ["layer"], max_tensor_elements=1)
        torch.testing.assert_close(output, expected)
        self.assertEqual(trace["events"][0]["shape"], [1, 2])
        self.assertTrue(trace["events"][0]["truncated"])
        self.assertEqual(len(model.layer._forward_hooks), 0)

    def test_cleanup_after_failure(self):
        class Broken(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Identity()
            def forward(self, x):
                self.layer(x)
                raise RuntimeError("expected")
        model = Broken().eval()
        with self.assertRaisesRegex(RuntimeError, "expected") as error:
            capture_forward(model, {"x": torch.ones(1)}, ["layer"])
        self.assertEqual(error.exception.trace["status"], "failed")
        self.assertEqual(len(error.exception.trace["events"]), 1)
        self.assertEqual(len(model.layer._forward_hooks), 0)

    def test_repeated_calls_and_capability_limits(self):
        class Repeated(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Identity()
            def forward(self, x):
                return self.layer(self.layer(x))
        model = Repeated().eval()
        _, trace = capture_forward(model, {"x": torch.ones(1)}, ["layer"], max_events=1)
        self.assertEqual(trace["dropped_events"], 1)
        with self.assertRaises(ValueError): capture_forward(model, {}, ["missing"])

    def test_nonfinite_and_complex_outputs(self):
        model = torch.nn.Identity().eval()
        _, trace = capture_forward(model, {"input": torch.tensor([float("nan"), float("inf")])}, [""])
        self.assertEqual(trace["events"][0]["values"], [None, None])
        _, trace = capture_forward(model, {"input": torch.tensor([1j])}, [""])
        self.assertEqual(trace["events"][0]["status"], "unsupported_tensor")
