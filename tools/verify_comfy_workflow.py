"""Opt-in real ComfyUI Flux2 workflow equivalence check; filenames supplied by caller."""
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("url")
parser.add_argument("denoiser")
parser.add_argument("encoder")
parser.add_argument("output_directory", type=Path)
args = parser.parse_args()
base = args.url.rstrip("/")
def call(route, data=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token: headers["X-Model-Run-Token"] = token
    req = Request(base + route, data=None if data is None else json.dumps(data).encode(), headers=headers)
    with urlopen(req, timeout=30) as response: return json.load(response)

graph = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": args.denoiser, "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": args.encoder, "type": "flux2", "device": "default"}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a red cube"}},
    "4": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
    "5": {"class_type": "ModelRunObserve", "inputs": {"model": ["1", 0], "modules": "double_blocks.0", "max_calls": 2, "elements": 256}},
}
for key, model, prefix in (("6", "1", "model-run-reference"), ("7", "5", "model-run-observed")):
    graph[key] = {"class_type": "KSampler", "inputs": {"model": [model, 0], "seed": 31, "steps": 2, "cfg": 1.,
        "sampler_name": "euler", "scheduler": "simple", "positive": ["3", 0], "negative": ["3", 0],
        "latent_image": ["4", 0], "denoise": 1.}}
    graph[str(int(key) + 2)] = {"class_type": "SaveLatent", "inputs": {"samples": [key, 0], "filename_prefix": prefix}}
result = call("/prompt", {"prompt": graph, "client_id": "model-run-verification"})
identifier = result["prompt_id"]
deadline = time.monotonic() + 480
while time.monotonic() < deadline:
    history = call("/history/" + identifier)
    if identifier in history:
        record = history[identifier]
        assert record["status"]["status_str"] == "success", record["status"]
        break
    time.sleep(1)
else: raise TimeoutError("ComfyUI workflow exceeded verification deadline")
from safetensors.torch import load_file
import torch
outputs = []
for key in ("8", "9"):
    file = record["outputs"][key]["latents"][0]
    path = (args.output_directory / file["subfolder"] / file["filename"]).resolve()
    assert path.is_relative_to(args.output_directory.resolve())
    outputs.append(load_file(path)["latent_tensor"])
torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)
token = call("/model-run/api/session")["token"]
state = call("/model-run/api/state", token=token)
trace = call("/model-run/api/traces/" + state["traces"][-1]["id"], token=token)
assert trace["source"] == "comfyui_native_denoiser" and trace["captured_calls"] == 2
print(json.dumps({"status": "passed", "prompt_id": identifier, "captured_calls": trace["captured_calls"],
                  "shape": list(outputs[0].shape), "max_abs_difference": float((outputs[0]-outputs[1]).abs().max())}))
