import unittest
try:
    import torch
    from diffusers import DiffusionPipeline
except ImportError:
    torch = None
from model_run.adapters import default_registry, UnsupportedCapability
from model_run.control import RunControl, ObservationCancelled


@unittest.skipIf(torch is None, "optional Diffusers dependency unavailable")
class PipelineAdapterTests(unittest.TestCase):
    def test_declared_callbacks_and_recovery(self):
        class Tiny(DiffusionPipeline):
            _callback_tensor_inputs = ["latents"]
            def __call__(self, num_inference_steps, callback_on_step_end=None, callback_on_step_end_tensor_inputs=None, fail=False):
                if fail:
                    raise RuntimeError("fixture failure")
                latent = torch.ones(1, 2)
                for step in range(num_inference_steps):
                    callback_on_step_end(self, step, num_inference_steps - step, {"latents": latent})
                return latent
        adapter = default_registry().bind("diffusers.pipeline", Tiny())
        self.assertTrue(adapter.negotiate(["denoising_steps"])["accepted"])
        self.assertFalse(adapter.negotiate(["module_outputs"])["accepted"])
        with self.assertRaises(RuntimeError): adapter.capture_steps({"num_inference_steps": 2, "fail": True})
        output, trace = adapter.capture_steps({"num_inference_steps": 2}, max_tensor_elements=1)
        self.assertEqual(list(output.shape), [1, 2])
        self.assertEqual(len(trace["events"]), 2)
        self.assertEqual(trace["adapter_id"], "diffusers.pipeline")
        with self.assertRaises(ValueError): adapter.capture_steps({"num_inference_steps": 129})
        with self.assertRaises(ValueError): adapter.capture_steps({"num_inference_steps": 2, "callback_on_step_end": None})
        control = RunControl(on_progress=lambda event: control.cancel() if event["stage"] == "denoising_step" and event["completed"] == 2 else None)
        with self.assertRaises(ObservationCancelled) as cancelled:
            adapter.capture_steps({"num_inference_steps": 3}, control=control)
        self.assertEqual(cancelled.exception.trace["status"], "cancelled")
        self.assertEqual(len(cancelled.exception.trace["events"]), 1)
        adapter.capture_steps({"num_inference_steps": 1})

    def test_unsupported_pipeline(self):
        class NoCallbacks(DiffusionPipeline):
            def __call__(self):
                return None
        adapter = default_registry().bind("diffusers.pipeline", NoCallbacks())
        self.assertFalse(adapter.negotiate(["denoising_steps"])["accepted"])
        with self.assertRaises(UnsupportedCapability): adapter.capture_steps({"num_inference_steps": 1})
