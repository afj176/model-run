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
from pathlib import Path, PurePosixPath, PureWindowsPath

from .adapters import TorchModuleAdapter
from .capture import capture_forward
from .tensor_slices import SliceRecorder, _clean

MAX_MEDIA_BYTES = 11 * 1024 * 1024
MAX_IMAGE_PIXELS = 64_000_000
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP")
SPEECH_RATE = 16000
MAX_SPEECH_SECONDS = 30
MAX_TRANSCRIPT_TOKENS = 256
CLS_METHOD = "class_token_attention_row"
AUDIO_METHOD = "whisper_cross_attention"
MEL_COLS = 300
TIME_BINS = 150
MAX_AUDIO_MAP_VALUES = 200000


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
    # Gemma 3 and 4 ship as image-and-text models that Transformers also loads as causal language models.
    if any(a.endswith("ForConditionalGeneration") for a in names) and not config.get("is_encoder_decoder") and config.get("model_type") in _causal_lm_types():
        return "text"
    return "inventory"


def _causal_lm_types():
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    except Exception:  # noqa: BLE001 - without Transformers nothing loads as text anyway
        return frozenset()
    return frozenset(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)


def declared_name(config, fallback):
    """The published name a package was saved from, when the folder name does not say what it is."""
    origin = config.get("_name_or_path") if isinstance(config, dict) else None
    # A local path from any system, not only this one, is private and says nothing about the model.
    local = isinstance(origin, str) and (PurePosixPath(origin).is_absolute() or PureWindowsPath(origin).is_absolute() or PureWindowsPath(origin).drive)
    if isinstance(origin, str) and origin and not local:
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


def _read_soundfile(data):
    """(samples [frames, channels], rate, channels, total frames) through libsndfile."""
    import soundfile
    try:
        with soundfile.SoundFile(io.BytesIO(data)) as source:
            rate, channels, total = source.samplerate, source.channels, source.frames
            # Only what will be used is decoded: a small compressed file can hold hours of audio.
            samples = source.read(frames=min(total, (MAX_SPEECH_SECONDS + 1) * rate), dtype="float32", always_2d=True)
    except (soundfile.LibsndfileError, RuntimeError, TypeError) as exc:
        raise ValueError("That file is not readable audio; use WAV, FLAC, OGG or MP3") from exc
    return samples, rate, channels, total


def _read_pyav(data):
    """The same through PyAV, which ComfyUI always installs where soundfile may be missing."""
    import av
    import numpy as np
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = container.streams.audio[0]
            rate, channels = int(stream.codec_context.sample_rate or 0), int(stream.codec_context.channels or 0)
            resampler = av.AudioResampler(format="fltp", layout=stream.codec_context.layout.name, rate=rate)
            limit, parts, got = (MAX_SPEECH_SECONDS + 1) * rate, [], 0
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    parts.append(out.to_ndarray())
                    got += parts[-1].shape[1]
                if got >= limit:
                    break
            known = container.duration if container.duration and container.duration > 0 else None
    except (av.error.FFmpegError, IndexError, ValueError, AttributeError) as exc:
        raise ValueError("That file is not readable audio; use WAV, FLAC, OGG or MP3") from exc
    samples = np.concatenate(parts, axis=1).T[:limit].astype(np.float32) if parts else np.zeros((0, max(channels, 1)), np.float32)
    total = round(known / 1_000_000 * rate) if known and got >= limit else got
    return samples, rate, channels or samples.shape[1], total


def decode_audio(media):
    """Mono 16 kHz samples and what was done to get them, from an audio file libsndfile or FFmpeg can read."""
    import numpy as np
    from scipy.signal import resample_poly
    try:
        import soundfile  # noqa: F401
        reader = _read_soundfile
    except ImportError:
        reader = _read_pyav
    samples, rate, channels, total = reader(_decode_bytes(media))
    if not len(samples) or rate < 1:
        raise ValueError("The audio file has no samples")
    duration = total / rate
    mono = samples.mean(axis=1)
    if rate != SPEECH_RATE:
        divisor = math.gcd(int(rate), SPEECH_RATE)
        mono = resample_poly(mono, SPEECH_RATE // divisor, int(rate) // divisor).astype(np.float32)
    kept = mono[:MAX_SPEECH_SECONDS * SPEECH_RATE]
    return kept, {"kind": "audio", "source_rate": int(rate), "source_channels": int(channels),
                  "source_seconds": round(duration, 3), "used_seconds": round(len(kept) / SPEECH_RATE, 3),
                  "rate": SPEECH_RATE, "clipped": duration > MAX_SPEECH_SECONDS}


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


def _audio_view(features, output, steps, info, used_mel, used_frames, encoder_length):
    """The spectrogram the model received and, per written token, where in the audio each decoder layer read.

    A decoder position predicts the next token, so a token's map is the attention row of the position before it.
    """
    import torch
    mel = features[0, :, :used_mel].detach().to(device="cpu", dtype=torch.float32)
    mel_cols = min(MEL_COLS, used_mel)
    if mel_cols != used_mel:
        mel = torch.nn.functional.adaptive_avg_pool1d(mel.unsqueeze(0), mel_cols)[0]
    bins = min(TIME_BINS, used_frames)
    edges = [round(i * used_frames / bins) for i in range(bins + 1)]
    view = {"method": AUDIO_METHOD, "seconds": info["used_seconds"], "time_bins": bins, "encoder_frames": used_frames, "encoder_length": encoder_length,
            "mel": {"bins": int(mel.shape[0]), "cols": mel_cols, "values": _clean(mel.reshape(-1).tolist())},
            "tokens": [{"step": s["index"], "token_id": s["token_id"], "probability": s["probability"]} for s in steps], "layers": [],
            "limitations": ["Each map is that decoder layer's measured cross-attention, averaged over heads, for the position that predicted the token. It shows where the layer read, not a word timing.",
                            "Whisper's window always covers 30 seconds. Attention landing past the end of the audio is reported as outside the audio."]}
    attentions = getattr(output, "cross_attentions", None)
    if not attentions or not steps:
        view["attention_unavailable"] = "The model returned no cross-attention for this input."
        return view
    keep = max(1, min(len(attentions), MAX_AUDIO_MAP_VALUES // max(1, len(steps) * bins)))
    chosen = sorted({round(i * (len(attentions) - 1) / max(1, keep - 1)) for i in range(keep)}) if keep > 1 else [len(attentions) - 1]
    for layer in chosen:
        weights = attentions[layer]
        if not isinstance(weights, torch.Tensor) or weights.ndim != 4 or int(weights.shape[-1]) != encoder_length:
            view["attention_unavailable"] = "This model's cross-attention does not line up with its encoder frames."
            view["layers"] = []
            return view
        mean = weights[0].mean(dim=0).detach().to(device="cpu", dtype=torch.float32)
        maps, outside = [], []
        for step in steps:
            row = mean[step["position"] - 1]
            total = row[:used_frames].cumsum(0)
            at = lambda frame: float(total[frame - 1]) if frame > 0 else 0.0
            maps.append(_clean([at(edges[i + 1]) - at(edges[i]) for i in range(bins)]))
            outside.append(_clean([max(0.0, float(row[used_frames:].sum()))])[0])
        view["layers"].append({"layer": layer, "maps": maps, "outside": outside})
    return view


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
        # Each layer keeps every patch by its features, and its attention is recorded beside it.
        from .layer_parts import PART_FEATURES, layer_parts, order_parts
        from .tensor_slices import ALL_FEATURES
        parts = [p for p in layer_parts(self.model, module_paths) if p[2] == "attention"] if token_grid else []
        watched = [*module_paths, *(p for p, _layer, _kind in parts)]
        def forward():
            return capture_forward(self.model, {"pixel_values": pixels, "output_attentions": True}, watched, control=control,
                                   max_tensor_elements=max_tensor_elements, max_events=max(64, len(watched)),
                                   slices=SliceRecorder(token_grid=token_grid, patch_features=ALL_FEATURES,
                                                        parts=[p for p, _layer, _kind in parts], part_features=PART_FEATURES))
        output, trace = self._run(forward)
        order_parts(trace, parts)
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

    def transcribe(self, inputs, module_paths, *, audio_info=None, top_k=5, max_tensor_elements=1024, max_new_tokens=MAX_TRANSCRIPT_TOKENS, control=None):
        """Greedy transcript with per-token scores, then one forward pass recording layer outputs and cross-attention."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        self._require("module_outputs")
        features = inputs["input_features"]
        info = audio_info or {"used_seconds": features.shape[-1] * 160 / SPEECH_RATE, "hop_length": 160, "rate": SPEECH_RATE}
        mel_per_second = info["rate"] / info["hop_length"]
        used_mel = max(1, min(int(features.shape[-1]), math.ceil(info["used_seconds"] * mel_per_second)))
        encoder_length = int(features.shape[-1]) // 2
        used_frames = max(1, min(encoder_length, math.ceil(used_mel / 2)))
        note = f"Shows the {used_frames} encoder frames that cover the audio. The rest of Whisper's fixed window encodes padding and is not shown."
        recorder = SliceRecorder(sequence=lambda length: (used_frames, "time", note) if length == encoder_length else (length, "token", None))
        class Cancellable(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if control is not None:
                    control.check()
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        generation_config = self.model.generation_config
        multilingual = bool(getattr(generation_config, "is_multilingual", False))
        # English-only models refuse a task; only models that can translate need one.
        language = {"task": "transcribe"} if multilingual else {}
        # The decoder's position limit counts the tokens it starts from, so leave room for them.
        room = int(getattr(self.model.config, "max_target_positions", max_new_tokens + 8)) - 8
        limit = max(1, min(max_new_tokens, room))
        def forward():
            with torch.inference_mode():
                generated = self.model.generate(features, **language, do_sample=False, num_beams=1, max_new_tokens=limit,
                                                return_dict_in_generate=True, output_scores=True, stopping_criteria=StoppingCriteriaList([Cancellable()]))
            sequence = generated.sequences
            output, trace = capture_forward(self.model, {"input_features": features, "decoder_input_ids": sequence, "output_attentions": True}, module_paths, control=control,
                                            max_tensor_elements=max_tensor_elements, slices=recorder)
            return generated, sequence, output, trace
        generated, sequence, output, trace = self._run(forward)
        ids = sequence[0].tolist()
        scores = generated.scores
        prefix = len(ids) - len(scores)
        if prefix < 1:
            raise ValueError("The decoder returned fewer tokens than scores")
        steps = []
        eos = generation_config.eos_token_id
        ends = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        for index, score in enumerate(scores):
            probabilities = torch.softmax(score[0].detach().to(device="cpu", dtype=torch.float32), dim=-1)
            token = ids[prefix + index]
            values, indices = probabilities.topk(min(top_k, probabilities.numel()))
            steps.append({"index": index, "position": prefix + index, "token_id": token, "probability": float(probabilities[token]),
                          "entropy_bits": _entropy_bits(probabilities),
                          "top_tokens": [{"token_id": int(i), "probability": float(p)} for p, i in zip(values.tolist(), indices.tolist())]})
        trace["audio_view"] = _audio_view(features, output, steps, info, used_mel, used_frames, encoder_length)
        trace["transcript_generation"] = {
            "schema_version": "1.0", "kind": "generation_capture", "status": "completed", "decoding": "greedy", "provenance": False,
            "stopped": "eos" if ids[-1] in ends else "max_new_tokens", "input": {"token_count": prefix}, "steps": steps,
            "prompt_token_ids": ids[:prefix], "model_class": trace["model_class"], "torch_version": trace["torch_version"],
            "limitations": ["Probabilities are softmax of the decoder's scores after Whisper's own token rules, which is what greedy decoding chose from.",
                            "Whisper picked the language itself; the tokens before the transcript show what it chose." if multilingual else "This model only transcribes English.",
                            "Only the first 30 seconds of audio are used.",
                            "Whisper can invent text on silence, tones or noise, sometimes with high probability; these probabilities are not a measure of accuracy."]}
        trace["adapter_id"] = self.adapter_id
        return output, trace


def _text_list(values, name):
    if not isinstance(values, list) or len(values) > 32 or any(not isinstance(v, str) or len(v) > 4096 for v in values):
        raise ValueError(f"Invalid {name}")


def validate_input_media(media):
    """What an imported trace says about its input is shown in the header, so it must be well formed."""
    if not isinstance(media, dict) or media.get("kind") not in ("image", "audio") or not isinstance(media.get("name"), str) or len(media["name"]) > 256:
        raise ValueError("Invalid input description")
    fields = ("source_width", "source_height") if media["kind"] == "image" else ("source_rate", "source_seconds", "used_seconds", "rate")
    if any(type(media.get(f)) not in (int, float) or not math.isfinite(media[f]) or media[f] < 0 for f in fields):
        raise ValueError("Invalid input description")


def validate_audio_view(view):
    """Raise ValueError unless ``view`` is a consistent audio section."""
    _text_list(view.get("limitations", []) if isinstance(view, dict) else None, "audio limitations")
    if isinstance(view, dict) and view.get("attention_unavailable") is not None and not isinstance(view["attention_unavailable"], str):
        raise ValueError("Invalid audio view")
    def numbers(values, low=None, high=None):
        return isinstance(values, list) and all(v is None or (type(v) in (int, float) and math.isfinite(v) and (low is None or low <= v) and (high is None or v <= high)) for v in values)
    if not isinstance(view, dict) or view.get("method") != AUDIO_METHOD or type(view.get("seconds")) not in (int, float) or not 0 < view["seconds"] <= MAX_SPEECH_SECONDS + 1:
        raise ValueError("Invalid audio view")
    mel, bins, tokens, layers = view.get("mel"), view.get("time_bins"), view.get("tokens"), view.get("layers")
    if (not isinstance(mel, dict) or type(mel.get("bins")) is not int or type(mel.get("cols")) is not int or not 1 <= mel["bins"] <= 256 or not 1 <= mel["cols"] <= MEL_COLS
            or not numbers(mel.get("values")) or len(mel["values"]) != mel["bins"] * mel["cols"]):
        raise ValueError("Invalid spectrogram")
    if type(bins) is not int or not 1 <= bins <= TIME_BINS or not isinstance(tokens, list) or len(tokens) > 512 or not isinstance(layers, list) or len(layers) > 64:
        raise ValueError("Invalid audio view")
    for token in tokens:
        if (not isinstance(token, dict) or type(token.get("token_id")) is not int or type(token.get("probability")) not in (int, float) or not 0 <= token["probability"] <= 1
                or type(token.get("step")) is not int or not isinstance(token.get("text", ""), str)):
            raise ValueError("Invalid audio token")
    for layer in layers:
        maps, outside = (layer.get("maps"), layer.get("outside")) if isinstance(layer, dict) else (None, None)
        if (type(layer.get("layer") if isinstance(layer, dict) else None) is not int or not isinstance(maps, list) or len(maps) != len(tokens)
                or not numbers(outside, 0, 1.001) or len(outside) != len(tokens) or any(not numbers(row, -0.001, 1.001) or len(row) != bins for row in maps)):
            raise ValueError("Invalid audio attention layer")
    audio = view.get("audio")
    if audio is not None and (not isinstance(audio, dict) or audio.get("mime") != "audio/wav" or not isinstance(audio.get("base64"), str) or len(audio["base64"]) > 2 * 1024 * 1024):
        raise ValueError("Invalid audio clip")


def encode_wav(samples, rate):
    """16-bit WAV of the exact samples the model received, for playback."""
    import wave
    import numpy as np
    pcm = np.asarray(samples, dtype=np.float32)
    pcm = np.round(np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
        out.setsampwidth(2)
        out.setframerate(int(rate))
        out.writeframes(pcm.tobytes())
    return {"mime": "audio/wav", "base64": base64.b64encode(buffer.getvalue()).decode()}


def validate_image_classification(record):
    """Raise ValueError unless ``record`` is a consistent image-classification section."""
    if not isinstance(record, dict) or record.get("method") != CLS_METHOD:
        raise ValueError("Invalid image classification")
    _text_list(record.get("limitations", []), "image limitations")
    if record.get("attention_unavailable") is not None and not isinstance(record["attention_unavailable"], str):
        raise ValueError("Invalid image classification")
    if record.get("class_count") is not None and type(record["class_count"]) is not int:
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
                or any(v is not None and (type(v) not in (int, float) or not 0 <= v <= 1.0001) for v in patches)
                or (layer.get("self") is not None and type(layer["self"]) not in (int, float))):
            raise ValueError("Invalid image attention layer")
    image = record.get("input_image")
    if image is not None and (not isinstance(image, dict) or image.get("mime") != "image/png" or not isinstance(image.get("base64"), str) or len(image["base64"]) > 6 * 1024 * 1024):
        raise ValueError("Invalid input image")
