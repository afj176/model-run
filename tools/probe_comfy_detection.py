"""Manual native architecture detection from shape-only meta tensors; no weights loaded."""
import argparse
import json
import sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("comfy_root", type=Path)
parser.add_argument("roots", nargs="+", type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(args.comfy_root))
import torch
from safetensors import safe_open
import comfy.model_detection as detection
results = []
for root in args.roots:
    for path in sorted(root.glob("*.safetensors")):
        result = {"artifact": path.name, "stage": "shape_only_native_detection", "inference": "not_run"}
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                # Detection uses shapes/key names. Meta tensors deliberately contain no weights.
                state = {name: torch.empty(handle.get_slice(name).get_shape(), device="meta") for name in handle.keys()}
                metadata = handle.metadata()
            prefix = detection.unet_prefix_from_state_dict(state)
            stripped = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            if not stripped:
                stripped = state
            config = detection.model_config_from_unet(stripped, "", metadata=metadata)
            method = "native"
            if config is None:
                converted = detection.convert_diffusers_mmdit(stripped, "")
                if converted is not None:
                    config = detection.model_config_from_unet(converted, "", metadata=metadata)
                    method = "diffusers_mmdit_conversion"
                else:
                    config = detection.model_config_from_diffusers_unet(stripped)
                    method = "diffusers_unet_detection"
            result.update({"status": "recognized" if config is not None else "unrecognized",
                           "method": method,
                           "config_class": type(config).__name__ if config is not None else None})
        except Exception as exc:
            result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        results.append(result)
        print(json.dumps(result), flush=True)
args.output.write_text(json.dumps({"results": results, "limitations": ["Synthetic float32 meta tensors represent shapes only; dtype and numeric detection behavior are not validated."]}, indent=2), encoding="utf-8")
