"""Opt-in tiny Diffusers integration smoke; accepts a local pipeline directory."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import diffusers
from diffusers import StableDiffusionPipeline
from model_run.adapters import default_registry
from model_run.control import RunControl, ObservationCancelled
from model_run.host import ModelHost

path = sys.argv[1]
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(path, local_files_only=True, use_safetensors=True, safety_checker=None, requires_safety_checker=False).to(device)
pipe.set_progress_bar_config(disable=True)
kwargs = {"prompt": "a red cube", "num_inference_steps": 3, "guidance_scale": 1.0, "output_type": "latent"}
baseline = pipe(**kwargs, generator=torch.Generator(device=device).manual_seed(37)).images
adapter = default_registry().bind("diffusers.pipeline", pipe)
result, trace = adapter.capture_steps({**kwargs, "generator": torch.Generator(device=device).manual_seed(37)}, max_tensor_elements=16)
observed = result.images
torch.testing.assert_close(observed, baseline)
assert len(trace["events"]) == 3
torch.testing.assert_close(torch.tensor(trace["events"][-1]["values"]), observed.flatten()[:16].float().cpu())
control = RunControl(on_progress=lambda event: control.cancel() if event["stage"] == "denoising_step" and event["completed"] == 2 else None)
try:
    adapter.capture_steps({**kwargs, "generator": torch.Generator(device=device).manual_seed(37)}, control=control)
    raise AssertionError("Expected cancellation")
except ObservationCancelled as exc:
    assert exc.trace["status"] == "cancelled" and len(exc.trace["events"]) == 1
recovered, _ = adapter.capture_steps({**kwargs, "generator": torch.Generator(device=device).manual_seed(37)})
torch.testing.assert_close(recovered.images, baseline)
with ModelHost() as host:
    host.register("diffusion-smoke", adapter, release=lambda: pipe.to("cpu"))
    host.submit("diffusion-smoke", "denoising_steps", inputs={**kwargs, "generator": torch.Generator(device=device).manual_seed(37)}, options={"max_tensor_elements": 16})
    assert host.wait(60)["run"]["status"] == "completed"
    hosted = host.take_trace()
    torch.testing.assert_close(torch.tensor(hosted["events"][-1]["values"]), baseline.flatten()[:16].float().cpu())
assert next(pipe.unet.parameters()).device.type == "cpu"
print(json.dumps({"device": device, "diffusers": diffusers.__version__, "steps": len(trace["events"]),
                  "latent_shape": list(observed.shape)}))
