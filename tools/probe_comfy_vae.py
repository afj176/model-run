"""Manual native-checkpoint probe; all installation and artifact paths are arguments."""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("comfy_root", type=Path)
parser.add_argument("artifact", type=Path)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(args.comfy_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import comfy.utils
import comfy.sd
from model_run.capture import capture_forward

state, metadata = comfy.utils.load_torch_file(str(args.artifact), safe_load=True, return_metadata=True)
vae = comfy.sd.VAE(sd=state, metadata=metadata, device=torch.device("cpu"), dtype=torch.float32)
vae.throw_exception_if_invalid()
model = vae.first_stage_model.eval()

class Decoder(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    def forward(self, latent):
        return self.inner.decode(latent)

wrapper = Decoder(model).eval()
shape = [1, vae.latent_channels] + ([4, 4] if vae.latent_dim == 2 else [1, 4, 4] if vae.latent_dim == 3 else [16])
if vae.extra_1d_channel is not None:
    shape = [1, vae.latent_channels, 8, vae.extra_1d_channel]
latent = torch.zeros(shape)
with torch.inference_mode():
    baseline = wrapper(latent=latent)
observed, trace = capture_forward(wrapper, {"latent": latent}, [""], max_tensor_elements=16)
torch.testing.assert_close(observed, baseline)
assert torch.isfinite(observed).all().item()
print(json.dumps({"artifact": args.artifact.name, "status": "passed", "stage": "native_vae_decode_capture",
                  "model_class": type(model).__name__, "latent_shape": shape, "output_shape": list(observed.shape),
                  "device": "cpu", "dtype": "float32", "captured_events": len(trace["events"])}))
