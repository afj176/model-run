"""Bounded callback observation of supported Diffusers denoising latents."""
import math


class DiffusionCapture:
    """A callback recorder for one pipeline invocation; does not alter pipeline state.

    The host must verify that the pipeline accepts callback_on_step_end and exposes
    ``latents`` in its callback tensor inputs. Never share a recorder across runs.
    """
    def __init__(self, *, every_n_steps=1, max_events=32, max_tensor_elements=1024, control=None):
        if type(every_n_steps) is not int or not 1 <= every_n_steps <= 10000:
            raise ValueError("every_n_steps must be between 1 and 10000")
        if type(max_events) is not int or not 1 <= max_events <= 256:
            raise ValueError("max_events must be between 1 and 256")
        if type(max_tensor_elements) is not int or not 1 <= max_tensor_elements <= 65536:
            raise ValueError("max_tensor_elements must be between 1 and 65536")
        self.every_n_steps = every_n_steps
        self.max_events = max_events
        self.max_tensor_elements = max_tensor_elements
        self.events = []
        self.dropped_events = 0
        self._pipeline_identity = None
        self._last_step = -1
        self.control = control

    def __call__(self, pipeline, step, timestep, callback_kwargs):
        import torch
        if self._pipeline_identity is None:
            self._pipeline_identity = id(pipeline)
        elif self._pipeline_identity != id(pipeline):
            raise ValueError("Recorder cannot be shared across pipelines")
        if self.control is not None:
            self.control.checkpoint("denoising_step", completed=step + 1)
        if type(step) is not int or step < 0 or not isinstance(callback_kwargs, dict):
            raise ValueError("Invalid diffusion callback event")
        if step <= self._last_step:
            raise ValueError("Diffusion step order must increase; use a new recorder for each run")
        self._last_step = step
        if step % self.every_n_steps:
            return callback_kwargs
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return callback_kwargs
        latent = callback_kwargs.get("latents")
        if not isinstance(latent, torch.Tensor):
            raise ValueError("Pipeline did not provide a latents tensor")
        if latent.layout != torch.strided or latent.is_complex() or latent.is_quantized:
            raise ValueError("Unsupported latent tensor representation")
        t = timestep.item() if isinstance(timestep, torch.Tensor) and timestep.numel() == 1 else timestep
        if type(t) not in (int, float) or not math.isfinite(t):
            raise ValueError("Timestep must be a finite scalar")
        flat = latent.detach().reshape(-1)
        sample = flat[:self.max_tensor_elements].to(device="cpu", dtype=torch.float32).tolist()
        nonfinite = sum(not math.isfinite(value) for value in sample)
        sample = [value if math.isfinite(value) else None for value in sample]
        self.events.append({"step": step, "timestep": t, "shape": list(latent.shape),
                            "dtype": str(latent.dtype), "device": str(latent.device),
                            "selection": "flattened_prefix", "values": sample,
                            "total_elements": flat.numel(), "truncated": len(sample) < flat.numel(),
                            "nonfinite_values_replaced_with_null": nonfinite})
        return callback_kwargs

    def trace(self, pipeline, *, status="completed"):
        if self._pipeline_identity != id(pipeline):
            raise ValueError("No events recorded for this pipeline")
        import torch
        return {"schema_version": "1.0", "kind": "diffusion_step_capture", "status": status,
                "model_class": f"{type(pipeline).__module__}.{type(pipeline).__qualname__}",
                "torch_version": str(torch.__version__), "capture": {"every_n_steps": self.every_n_steps,
                "max_events": self.max_events, "max_tensor_elements": self.max_tensor_elements},
                "events": list(self.events), "dropped_events": self.dropped_events,
                "limitations": ["Latents are not decoded images or audio.",
                                "Flattened prefixes converted to float32 are not representative samples.",
                                "Non-contiguous tensors may require a full reshape copy.",
                                "Pipeline must support callback_on_step_end with latents."]}
