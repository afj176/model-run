"""Manual native text-encoder load probe with an explicit conditioning family."""
import argparse
import json
import sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("comfy_root", type=Path)
parser.add_argument("artifact", type=Path)
parser.add_argument("clip_type")
parser.add_argument("--forward", action="store_true")
args = parser.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(args.comfy_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import comfy.sd
from model_run.structure import inspect_module_tree
from model_run.capture import capture_forward
clip = comfy.sd.load_clip([str(args.artifact)], clip_type=comfy.sd.CLIPType[args.clip_type],
    model_options={"load_device": torch.device("cuda" if args.forward else "cpu"), "offload_device": torch.device("cpu")}, disable_dynamic=True)
graph = inspect_module_tree(clip.cond_stage_model, max_modules=30000)
if args.forward:
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = clip.cond_stage_model
        def forward(self):
            return clip.encode_from_tokens(tokens)
    tokens = clip.tokenize("a red cube")
    clip.load_model(tokens)
    wrapper = Encoder().eval()
    candidates = [name for name, _ in wrapper.named_modules() if name.endswith(".layers.0")]
    if not candidates:
        raise ValueError("No supported first-layer capture point")
    selected = candidates[0]
    with torch.inference_mode():
        reference = wrapper()
    observed, trace = capture_forward(wrapper, {}, [selected], max_tensor_elements=32)
    torch.testing.assert_close(observed, reference)
    assert torch.isfinite(observed).all().item()
    assert trace["events"][0]["status"] == "captured"
    print(json.dumps({"artifact": args.artifact.name, "status": "passed", "stage": "native_encoder_forward_capture",
                      "clip_type": args.clip_type, "output_shape": list(observed.shape), "module": selected,
                      "activation_shape": trace["events"][0]["shape"], "device": "cuda"}))
    raise SystemExit(0)
print(json.dumps({"artifact": args.artifact.name, "status": "passed", "stage": "native_encoder_load_and_module_tree",
                  "clip_type": args.clip_type, "module_count": len(graph["nodes"]), "truncated": graph["truncated"],
                  "model_class": type(clip.cond_stage_model).__name__, "inference": "not_run"}))
