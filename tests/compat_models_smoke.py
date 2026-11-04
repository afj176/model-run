"""Opt-in real-checkpoint compatibility probes; input tensors are synthetic."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForSpeechSeq2Seq
from model_run.capture import capture_forward

kind, path, device = sys.argv[1:4]
classes = {"text": AutoModelForCausalLM, "vision": AutoModelForImageClassification, "audio": AutoModelForSpeechSeq2Seq}
model = classes[kind].from_pretrained(path, local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32).to(device).eval()
if kind == "text":
    inputs = {"input_ids": torch.tensor([[1, 2, 3]], device=device), "use_cache": False}
    path_name = "model.layers.0"
elif kind == "vision":
    inputs = {"pixel_values": torch.zeros((1, 3, model.config.image_size, model.config.image_size), device=device)}
    path_name = "vit.layers.0"
else:
    inputs = {"input_features": torch.zeros((1, model.config.num_mel_bins, model.config.max_source_positions * 2), device=device),
              "decoder_input_ids": torch.tensor([[model.config.decoder_start_token_id]], device=device), "use_cache": False}
    path_name = "model.encoder.layers.0"
with torch.inference_mode():
    reference = model(**inputs).logits
observed, trace = capture_forward(model, inputs, [path_name], max_tensor_elements=32)
torch.testing.assert_close(observed.logits, reference)
assert trace["events"][0]["status"] == "captured"
print(json.dumps({"kind": kind, "model_type": model.config.model_type, "device": device,
                  "transformers": transformers.__version__, "module": path_name,
                  "activation_shape": trace["events"][0]["shape"], "logits_shape": list(reference.shape)}))
