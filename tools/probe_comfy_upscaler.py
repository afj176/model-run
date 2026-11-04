"""Manual LTX latent upscaler forward/capture comparison."""
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
import comfy.ops
from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
from model_run.capture import capture_forward
state, metadata = comfy.utils.load_torch_file(str(args.artifact), safe_load=True, return_metadata=True)
config = json.loads(metadata["config"])
model = LatentUpsampler.from_config(config, operations=comfy.ops.disable_weight_init).float().eval()
model.load_state_dict(state, strict=True)
latent = torch.zeros((1, config["in_channels"], 1, 4, 4))
with torch.inference_mode():
    baseline = model(latent)
import inspect
argument = next(iter(inspect.signature(model.forward).parameters))
output, trace = capture_forward(model, {argument: latent}, [""], max_tensor_elements=16)
torch.testing.assert_close(output, baseline)
assert torch.isfinite(output).all().item()
print(json.dumps({"artifact": args.artifact.name, "status": "passed", "stage": "native_upscaler_forward_capture",
                  "latent_shape": list(latent.shape), "output_shape": list(output.shape), "device": "cpu", "dtype": "float32"}))
