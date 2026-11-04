"""Manual isolated native diffusion loading probe. Does not run generation."""
import argparse
import json
import sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("comfy_root", type=Path)
parser.add_argument("artifact", type=Path)
parser.add_argument("--max-bytes", type=int, default=9 * 1024**3)
args = parser.parse_args()
if args.artifact.stat().st_size > args.max_bytes:
    print(json.dumps({"artifact": args.artifact.name, "status": "deferred", "reason": "Above configured isolated load budget", "inference": "not_run"}))
    raise SystemExit(0)
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(args.comfy_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import comfy.sd
from model_run.structure import inspect_module_tree

patcher = comfy.sd.load_diffusion_model(str(args.artifact), model_options={"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}, disable_dynamic=True)
graph = inspect_module_tree(patcher.model, max_modules=30000)
print(json.dumps({"artifact": args.artifact.name, "status": "passed", "stage": "native_weight_load_and_module_tree",
                  "module_count": len(graph["nodes"]), "truncated": graph["truncated"],
                  "model_class": type(patcher.model).__name__, "inference": "not_run"}))
