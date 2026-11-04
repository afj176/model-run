"""Manual native denoiser forward comparison on small synthetic latent inputs."""
import argparse
import json
import sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("comfy_root", type=Path)
parser.add_argument("artifact", type=Path)
parser.add_argument("--lora", type=Path)
parser.add_argument("--bridge", action="store_true", help="Also verify the native MODEL wrapper against the same forward")
args = parser.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(args.comfy_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import comfy.sd
import comfy.model_management as mm
from model_run.adapters import default_registry

patcher = comfy.sd.load_diffusion_model(str(args.artifact), disable_dynamic=True)
if args.lora:
    import comfy.utils
    state = comfy.utils.load_torch_file(str(args.lora), safe_load=True)
    patcher, _ = comfy.sd.load_lora_for_models(patcher, None, state, 0.7, 0.0)
mm.load_models_gpu([patcher], force_full_load=True)
model = patcher.model.diffusion_model.eval()
family = type(patcher.model).__name__
device = patcher.load_device
dtype = patcher.model.get_dtype_inference()
channels = patcher.model.latent_format.latent_channels
if family in ("Flux2", "QwenImage", "QwenImage21"):
    context_dim = model.txt_in.weight.shape[1] if hasattr(model.txt_in, "weight") else model.txt_in.text_norm.weight.shape[0]
    inputs = {"x": torch.zeros((1, channels, 4, 4), device=device, dtype=dtype),
              "timestep": torch.tensor([0.5], device=device, dtype=dtype),
              "context": torch.zeros((1, 4, context_dim), device=device, dtype=dtype)}
    if family == "QwenImage":
        inputs["x"] = inputs["x"].unsqueeze(2)
    if family == "Flux2":
        inputs["guidance"] = torch.tensor([3.5], device=device, dtype=dtype)
    selected = "double_blocks.0" if family == "Flux2" else "transformer_blocks.0"
elif family == "SeedVR2":
    inputs = {"x": torch.zeros((1, channels, 1, 4, 4), device=device, dtype=dtype),
              "timestep": torch.tensor([0.5], device=device, dtype=dtype), "context": None,
              "condition": torch.zeros((1, channels + 1, 1, 4, 4), device=device, dtype=dtype),
              "disable_cache": False, "transformer_options": {"cond_or_uncond": [0]}}
    selected = "blocks.0"
elif family == "LTXAV":
    context_dim = model.inner_dim + model.audio_inner_dim if model.caption_proj_before_connector else model.caption_channels * 2
    inputs = {"x": [torch.zeros((1, channels, 1, 4, 4), device=device, dtype=dtype),
                      torch.zeros((1, model.num_audio_channels, 2, model.audio_frequency_bins), device=device, dtype=dtype)],
              "timestep": torch.tensor([0.5], device=device, dtype=dtype),
              "context": torch.zeros((1, 4, context_dim), device=device, dtype=dtype)}
    selected = "transformer_blocks.0"
elif family == "HiDreamO1":
    from comfy.ldm.hidream_o1.conditioning import build_extra_conds
    x = torch.zeros((1, 3, 64, 64), device=device, dtype=dtype)
    inputs = {"x": x, "timesteps": torch.tensor([500.0], device=device, dtype=dtype),
              **build_extra_conds(torch.tensor([[1, 2]], device=device), x)}
    selected = "language_model.layers.0"
else:
    raise ValueError(f"No validated synthetic-input contract for {family}")
adapter = default_registry().bind("pytorch.module", model)
with torch.inference_mode():
    reference = model(**inputs)
observed, trace = adapter.capture(inputs, [selected], max_tensor_elements=32)
torch.testing.assert_close(observed, reference)
outputs = observed if isinstance(observed, (list, tuple)) else [observed]
assert all(torch.isfinite(tensor).all().item() for tensor in outputs)
assert trace["events"][0]["status"] == "captured"
audio_shape = None
if args.bridge:
    from model_run.comfy_bridge import DenoiserObserver
    observer = DenoiserObserver(patcher, [selected], max_calls=1, elements=32)
    # Invoke the real loaded denoiser through the exact sampler wrapper signature.
    def native_call(_input, _timestep, **_conditioning):
        return model(**inputs)
    with torch.inference_mode():
        wrapped = observer(native_call, {"input": None, "timestep": None, "c": {}})
    torch.testing.assert_close(wrapped, reference)
    assert observer.trace["events"][0]["module"] == selected
    assert observer.trace["source"] == "comfyui_native_denoiser"
if family == "LTXAV":
    audio_observed, audio_trace = adapter.capture(inputs, [selected], max_tensor_elements=32, output_selectors={selected: [1]})
    torch.testing.assert_close(audio_observed, reference)
    assert audio_trace["events"][0]["output_path"] == [1]
    audio_shape = audio_trace["events"][0]["shape"]
print(json.dumps({"artifact": args.artifact.name, "status": "passed", "stage": "native_denoiser_forward_capture",
                  "family": family, "device": str(device), "dtype": str(dtype), "output_shapes": [list(t.shape) for t in outputs],
                  "module": selected, "activation_shape": trace["events"][0]["shape"], "audio_activation_shape": audio_shape,
                  "lora": args.lora.name if args.lora else None, "native_bridge": args.bridge}))
