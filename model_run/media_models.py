"""Local image-classification and speech-recognition bindings for the standalone host.

Only local files are read and no remote code runs. Everything reported is measured on the one
input the person supplied; attention weights describe where a layer looked, not why it decided.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import math
from pathlib import Path

from .adapters import TorchModuleAdapter
from .capture import capture_forward
from .tensor_slices import SliceRecorder, _clean

MAX_MEDIA_BYTES = 11 * 1024 * 1024
MAX_IMAGE_PIXELS = 64_000_000
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP")
SPEECH_RATE = 16000
MAX_SPEECH_SECONDS = 30
MAX_TRANSCRIPT_TOKENS = 128
CLS_METHOD = "class_token_attention_row"


def task_for_config(config):
    """The task a config declares: text, image, audio, or inventory when it is none Model Run can load."""
    if not isinstance(config, dict):
        return "inventory"
    architectures = config.get("architectures")
    names = [a for a in architectures if isinstance(a, str)] if isinstance(architectures, list) else []
    if any(a.endswith("ForCausalLM") for a in names):
        return "text"
    if any(a.endswith("ForImageClassification") for a in names):
        return "image"
    if config.get("model_type") == "whisper":
        return "audio"
    return "inventory"


def declared_name(config, fallback):
    """The published name a package was saved from, when the folder name does not say what it is."""
    origin = config.get("_name_or_path") if isinstance(config, dict) else None
    if isinstance(origin, str) and origin and not Path(origin).is_absolute():
        return origin.replace("\\", "/").rstrip("/").split("/")[-1][:80] or fallback
    return fallback


def _decode_bytes(media):
    if not isinstance(media, dict) or not isinstance(media.get("base64"), str):
        raise ValueError("Choose a file to run this model on")
    if len(media["base64"]) > MAX_MEDIA_BYTES * 4 // 3 + 8:
        raise ValueError(f"Files can be at most {MAX_MEDIA_BYTES // (1024 * 1024)} MiB")
    try:
        data = base64.b64decode(media["base64"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The file could not be read") from exc
    if not data:
        raise ValueError("The file is empty")
    return data


def decode_image(media):
    """A bounded RGB image from an uploaded PNG, JPEG or WebP."""
    from PIL import Image
    try:
        image = Image.open(io.BytesIO(_decode_bytes(media)), formats=IMAGE_FORMATS)
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError(f"Images can be at most {MAX_IMAGE_PIXELS // 1_000_000} megapixels")
        return image.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("That file is not a readable PNG, JPEG or WebP image") from exc


def decode_audio(media):
    """Mono 16 kHz samples and what was done to get them, from an audio file libsndfile can read."""
    import numpy as np
    import soundfile
    from scipy.signal import resample_poly
    try:
        samples, rate = soundfile.read(io.BytesIO(_decode_bytes(media)), dtype="float32", always_2d=True)
    except (soundfile.LibsndfileError, RuntimeError, TypeError) as exc:
        raise ValueError("That file is not readable audio; use WAV, FLAC, OGG or MP3") from exc
    if not len(samples) or rate < 1:
        raise ValueError("The audio file has no samples")
    channels = samples.shape[1]
    duration = len(samples) / rate
    mono = samples.mean(axis=1)
    if rate != SPEECH_RATE:
        divisor = math.gcd(int(rate), SPEECH_RATE)
        mono = resample_poly(mono, SPEECH_RATE // divisor, int(rate) // divisor).astype(np.float32)
    kept = mono[:MAX_SPEECH_SECONDS * SPEECH_RATE]
    return kept, {"kind": "audio", "source_rate": int(rate), "source_channels": int(channels),
                  "source_seconds": round(duration, 3), "used_seconds": round(len(kept) / SPEECH_RATE, 3),
                  "rate": SPEECH_RATE, "clipped": len(kept) < len(mono)}


def _image_processor(auto, path, shared):
    """The model's image processor; Transformers' PIL version when torchvision is not installed."""
    try:
        return auto.from_pretrained(path, **shared)
    except ImportError:
        import transformers
        config = read_config(path) or {}
        for architecture in config.get("architectures") or []:
            backend = getattr(transformers, f"{str(architecture).removesuffix('ForImageClassification')}ImageProcessorPil", None)
            if backend is not None:
                return backend.from_pretrained(path, **shared)
        raise ValueError("This image model's processor needs torchvision, which is not installed") from None


def load_local_media_model(directory, task, *, device="cpu"):
    """Load a local image or speech model with eager attention; returns (model, processor)."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModelForSpeechSeq2Seq, AutoProcessor
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    path = str(Path(directory).resolve(strict=True))
    shared = {"local_files_only": True, "trust_remote_code": False}
    if task == "image":
        processor = _image_processor(AutoImageProcessor, path, shared)
        loader = AutoModelForImageClassification
    elif task == "audio":
        processor = AutoProcessor.from_pretrained(path, **shared)
        loader = AutoModelForSpeechSeq2Seq
    else:
        raise ValueError("Unknown media task")
    model = loader.from_pretrained(path, use_safetensors=True, dtype=torch.float32, attn_implementation="eager", **shared)
    model.to(device)
    model.eval()
    return model, processor


def read_config(directory):
    path = Path(directory) / "config.json"
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return None


def _entropy_bits(probabilities):
    import torch
    positive = probabilities[probabilities > 0]
    return float(-(positive * torch.log2(positive)).sum())


def _patch_grid(config):
    """(rows, cols) of an image model's patches, or None when its config does not state them."""
    image, patch = getattr(config, "image_size", None), getattr(config, "patch_size", None)
    pair = lambda v: tuple(v) if isinstance(v, (list, tuple)) and len(v) == 2 else (v, v)
    (height, width), (patch_h, patch_w) = pair(image), pair(patch)
    if not all(type(v) is int and v > 0 for v in (height, width, patch_h, patch_w)):
        return None
    return height // patch_h, width // patch_w


class ImageClassifierAdapter(TorchModuleAdapter):
    adapter_id = "transformers.image_classification"

    def __init__(self, model):
        super().__init__(model)
        self.grid = _patch_grid(model.config)

    def classify(self, inputs, module_paths, *, top_k=5, max_tensor_elements=1024, control=None):
        import torch
        self._require("module_outputs")
        grid = self.grid
        pixels = inputs["pixel_values"]
        token_grid = (1, grid[0], grid[1]) if grid else None
        def forward():
            return capture_forward(self.model, {"pixel_values": pixels, "output_attentions": True}, module_paths, control=control,
                                   max_tensor_elements=max_tensor_elements, slices=SliceRecorder(token_grid=token_grid))
        output, trace = self._run(forward)
        logits = output.logits[0].detach().to(device="cpu", dtype=torch.float32)
        probabilities = torch.softmax(logits, dim=-1)
        values, indices = probabilities.topk(min(top_k, probabilities.numel()))
        labels = getattr(self.model.config, "id2label", None) or {}
        record = {"top_labels": [{"index": int(i), "label": str(labels.get(int(i), labels.get(str(int(i)), f"class {int(i)}")))[:120], "probability": float(p)}
                                 for p, i in zip(values.tolist(), indices.tolist())],
                  "entropy_bits": _entropy_bits(probabilities), "class_count": int(probabilities.numel()),
                  "pixel_shape": list(pixels.shape), "method": CLS_METHOD, "attention_layers": [], "grid": list(grid) if grid else None,
                  "limitations": ["Attention is the class token's row of each layer's measured attention, averaged over heads. It shows where that layer looked, not why the model decided.",
                                  "The image shown is what the model received after its processor resized it."]}
        attentions = getattr(output, "attentions", None)
        expected = grid[0] * grid[1] + 1 if grid else None
        if attentions and expected:
            for layer, weights in enumerate(attentions):
                if not isinstance(weights, torch.Tensor) or weights.ndim != 4 or list(weights.shape[-2:]) != [expected, expected]:
                    record["attention_layers"] = []
                    record["attention_unavailable"] = "This model's attention does not line up with one class token followed by its patches."
                    break
                row = weights[0].mean(dim=0)[0].detach().to(device="cpu", dtype=torch.float32)
                record["attention_layers"].append({"layer": layer, "self": _clean([row[0].item()])[0], "patches": _clean(row[1:].tolist())})
        else:
            record["attention_unavailable"] = "The model returned no attention weights for this input."
        trace["image_classification"] = record
        trace["adapter_id"] = self.adapter_id
        return output, trace


class SpeechAdapter(TorchModuleAdapter):
    adapter_id = "transformers.speech_recognition"

    def transcribe(self, inputs, module_paths, *, top_k=5, max_tensor_elements=1024, max_new_tokens=MAX_TRANSCRIPT_TOKENS, control=None):
        """Greedy transcript with per-token scores, then one forward pass recording layer outputs."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        self._require("module_outputs")
        features = inputs["input_features"]
        class Cancellable(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if control is not None:
                    control.check()
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        def forward():
            with torch.inference_mode():
                generated = self.model.generate(features, task="transcribe", do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                                return_dict_in_generate=True, output_scores=True, stopping_criteria=StoppingCriteriaList([Cancellable()]))
            sequence = generated.sequences
            output, trace = capture_forward(self.model, {"input_features": features, "decoder_input_ids": sequence}, module_paths, control=control,
                                            max_tensor_elements=max_tensor_elements)
            return generated, sequence, output, trace
        generated, sequence, output, trace = self._run(forward)
        ids = sequence[0].tolist()
        scores = generated.scores
        prefix = len(ids) - len(scores)
        if prefix < 1:
            raise ValueError("The decoder returned fewer tokens than scores")
        steps = []
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        for index, score in enumerate(scores):
            probabilities = torch.softmax(score[0].detach().to(device="cpu", dtype=torch.float32), dim=-1)
            token = ids[prefix + index]
            values, indices = probabilities.topk(min(top_k, probabilities.numel()))
            steps.append({"index": index, "position": prefix + index, "token_id": token, "probability": float(probabilities[token]),
                          "entropy_bits": _entropy_bits(probabilities),
                          "top_tokens": [{"token_id": int(i), "probability": float(p)} for p, i in zip(values.tolist(), indices.tolist())]})
        trace["transcript_generation"] = {
            "schema_version": "1.0", "kind": "generation_capture", "status": "completed", "decoding": "greedy", "provenance": False,
            "stopped": "eos" if ids[-1] == eos else "max_new_tokens", "input": {"token_count": prefix}, "steps": steps,
            "prompt_token_ids": ids[:prefix], "model_class": trace["model_class"], "torch_version": trace["torch_version"],
            "limitations": ["Probabilities are softmax of the decoder's scores after Whisper's own token rules, which is what greedy decoding chose from.",
                            "Whisper picked the language itself; the tokens before the transcript show what it chose.",
                            "Only the first 30 seconds of audio are used."]}
        trace["adapter_id"] = self.adapter_id
        return output, trace


def validate_image_classification(record):
    """Raise ValueError unless ``record`` is a consistent image-classification section."""
    if not isinstance(record, dict) or record.get("method") != CLS_METHOD:
        raise ValueError("Invalid image classification")
    labels = record.get("top_labels")
    if (not isinstance(labels, list) or not 1 <= len(labels) <= 20 or type(record.get("entropy_bits")) not in (int, float)
            or not math.isfinite(record["entropy_bits"])):
        raise ValueError("Invalid image classification labels")
    for entry in labels:
        if (not isinstance(entry, dict) or type(entry.get("index")) is not int or not isinstance(entry.get("label"), str)
                or type(entry.get("probability")) not in (int, float) or not 0 <= entry["probability"] <= 1):
            raise ValueError("Invalid image classification label")
    layers, grid = record.get("attention_layers"), record.get("grid")
    if not isinstance(layers, list) or len(layers) > 128:
        raise ValueError("Invalid image attention")
    if layers and (not isinstance(grid, list) or len(grid) != 2 or any(type(v) is not int or not 1 <= v <= 256 for v in grid)):
        raise ValueError("Invalid image attention grid")
    for layer in layers:
        patches = layer.get("patches") if isinstance(layer, dict) else None
        if (type(layer.get("layer") if isinstance(layer, dict) else None) is not int or not isinstance(patches, list) or len(patches) != grid[0] * grid[1]
                or any(v is not None and (type(v) not in (int, float) or not 0 <= v <= 1.0001) for v in patches)):
            raise ValueError("Invalid image attention layer")
    image = record.get("input_image")
    if image is not None and (not isinstance(image, dict) or image.get("mime") != "image/png" or not isinstance(image.get("base64"), str) or len(image["base64"]) > 6 * 1024 * 1024):
        raise ValueError("Invalid input image")
