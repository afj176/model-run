import tempfile
import unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None
from model_run.diffusion_capture import DiffusionCapture
from model_run.trace_store import save_trace, load_trace


@unittest.skipIf(torch is None, "optional PyTorch dependency unavailable")
class DiffusionTests(unittest.TestCase):
    def test_limits_passthrough_and_roundtrip(self):
        pipe = object()
        capture = DiffusionCapture(every_n_steps=2, max_events=1, max_tensor_elements=2)
        data = {"latents": torch.tensor([1., 2., 3.])}
        self.assertIs(capture(pipe, 0, torch.tensor(9), data), data)
        capture(pipe, 1, 8, data)
        capture(pipe, 2, 7, data)
        trace = capture.trace(pipe)
        self.assertEqual(trace["dropped_events"], 1)
        self.assertEqual(trace["events"][0]["values"], [1., 2.])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path = Path(tmp) / "trace.json"
            save_trace(path, trace)
            self.assertEqual(load_trace(path), trace)
        with self.assertRaises(ValueError): capture(object(), 3, 5, data)
        with self.assertRaises(ValueError): capture(pipe, 0, 9, data)

    def test_missing_latents(self):
        with self.assertRaises(ValueError): DiffusionCapture()(object(), 0, 1, {})
