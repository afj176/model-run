"""Manual native LoRA key resolution and patch registration against an explicit base."""
import argparse
import json
import sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("comfy_root", type=Path)
parser.add_argument("artifact", type=Path)
parser.add_argument("base", type=Path)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(args.comfy_root))
import torch
import comfy.sd
import comfy.lora
import comfy.utils
model = comfy.sd.load_diffusion_model(str(args.base), model_options={"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}, disable_dynamic=True)
state = comfy.utils.load_torch_file(str(args.artifact), safe_load=True)
key_map = comfy.lora.model_lora_keys_unet(model.model, {})
patches = comfy.lora.load_lora(state, key_map)
applied = model.add_patches(patches, 1.0)
if not patches or len(applied) != len(patches):
    raise ValueError(f"Incomplete native patch registration: {len(applied)}/{len(patches)}")
print(json.dumps({"artifact": args.artifact.name, "base": args.base.name, "status": "passed",
                  "stage": "native_lora_resolution_and_patch_registration", "patch_count": len(patches),
                  "limitations": ["Patch registration does not verify numerical patch application or generated output."]}))
