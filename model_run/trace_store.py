"""Explicit bounded JSON trace export and import."""
import json
import os
import tempfile
import math
from pathlib import Path
from .tensor_slices import validate_slices

MAX_TRACE_BYTES = 16 * 1024 * 1024
SUPPORTED_KINDS = {"pytorch_module_capture", "attention_capture", "diffusion_step_capture", "head_ablation", "activation_patching", "generation_capture"}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_cross_attention(section):
    """Measured image-to-text attention maps attached to a diffusion or module trace."""
    if (not isinstance(section, dict) or section.get("method") != "measured_attention_probabilities" or not isinstance(section.get("source"), str)
            or not isinstance(section.get("selection"), str) or not isinstance(section.get("events"), list) or len(section["events"]) > 32
            or not isinstance(section.get("limitations"), list) or type(section.get("grid_limit")) is not int):
        raise ValueError("Invalid cross-attention section")
    labels = section.get("text_labels")
    if labels is not None and (not isinstance(labels, list) or len(labels) > 4096 or any(not isinstance(l, str) or len(l) > 48 for l in labels)):
        raise ValueError("Invalid cross-attention token labels")
    skipped = section.get("skipped")
    if skipped is not None and (not isinstance(skipped, list) or len(skipped) > 32 or any(not isinstance(s, dict) or not isinstance(s.get("reason"), str) for s in skipped)):
        raise ValueError("Invalid cross-attention skip list")
    for event in section["events"]:
        if (not isinstance(event, dict) or type(event.get("denoiser_call")) is not int or event["denoiser_call"] < 0
                or event.get("attention_kind") not in ("cross", "joint") or type(event.get("text_tokens")) is not int or not 1 <= event["text_tokens"] <= 4096
                or not isinstance(event.get("grid"), list) or len(event["grid"]) != 2 or any(type(n) is not int or not 1 <= n <= 64 for n in event["grid"])
                or not isinstance(event.get("source_grid"), list) or len(event["source_grid"]) != 2 or any(type(n) is not int or n < 1 for n in event["source_grid"])
                or type(event.get("heads")) is not int or not isinstance(event.get("batch"), str) or not isinstance(event.get("block"), str)):
            raise ValueError("Invalid cross-attention event")
        frames = event.get("frames", 1)
        if type(frames) is not int or not 1 <= frames <= 64:
            raise ValueError("Invalid cross-attention frame count")
        cells = frames * event["grid"][0] * event["grid"][1]
        maps = event.get("maps")
        if (not isinstance(maps, list) or len(maps) != event["text_tokens"] or event["text_tokens"] * cells > 262144
                or any(not isinstance(row, list) or len(row) != cells or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in row) for row in maps)):
            raise ValueError("Invalid cross-attention maps")
        for key, length in (("text_mass", cells), ("token_mass", event["text_tokens"])):
            values = event.get(key)
            if not isinstance(values, list) or len(values) != length or any(type(v) not in (int, float) or not 0 <= v <= 1.0001 for v in values):
                raise ValueError(f"Invalid cross-attention {key}")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _optional_number(value):
    return value is None or _number(value)


DENOISING_IMAGES = {"latent", "estimate", "noise", "positive", "negative", "guidance"}


def _validate_denoising(section):
    """Per-step sampler records: measured numbers plus small JPEG panels."""
    import base64
    if not isinstance(section, dict) or section.get("method") != "sampler_post_cfg" or not isinstance(section.get("steps"), list) or len(section["steps"]) > 64:
        raise ValueError("Invalid denoising section")
    if not isinstance(section.get("limitations"), list) or section.get("projection") not in ("latent_rgb_factors", "first_channels"):
        raise ValueError("Invalid denoising section")
    if not isinstance(section.get("skipped", ""), str):
        raise ValueError("Invalid denoising section")
    for step in section["steps"]:
        if not isinstance(step, dict) or type(step.get("index")) is not int or not _optional_number(step.get("sigma")):
            raise ValueError("Invalid denoising step")
        if not isinstance(step.get("rms"), dict) or any(not _optional_number(v) for v in step["rms"].values()):
            raise ValueError("Invalid denoising step")
        for key in ("estimate_change", "guidance_rms", "guidance_cosine", "guidance_scale"):
            if not _optional_number(step.get(key)):
                raise ValueError("Invalid denoising step")
        images = step.get("images", {})
        if not isinstance(images, dict) or set(images) - DENOISING_IMAGES:
            raise ValueError("Invalid denoising images")
        for encoded in images.values():
            if not isinstance(encoded, str) or len(encoded) > 512 * 1024:
                raise ValueError("Invalid denoising image")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ValueError("Invalid denoising image") from exc
            if not raw.startswith(b"\xff\xd8\xff"):
                raise ValueError("Denoising images must be JPEG")


def _validate_text_encoder(section):
    if not isinstance(section, dict) or not isinstance(section.get("encoders"), list) or len(section["encoders"]) > 8 or not isinstance(section.get("limitations"), list):
        raise ValueError("Invalid text encoder section")
    for encoder in section["encoders"]:
        if not isinstance(encoder, dict) or not isinstance(encoder.get("encoder"), str) or not isinstance(encoder.get("tokens"), list) or not isinstance(encoder.get("layers"), list):
            raise ValueError("Invalid text encoder")
        count = len(encoder["tokens"])
        if not 1 <= count <= 256 or len(encoder["layers"]) > 128:
            raise ValueError("Invalid text encoder size")
        for token in encoder["tokens"]:
            if not isinstance(token, dict) or not isinstance(token.get("text"), str) or len(token["text"]) > 48 or not (token.get("id") is None or type(token["id"]) is int) or not _number(token.get("weight")) or type(token.get("special", False)) is not bool:
                raise ValueError("Invalid text encoder token")
        for layer in encoder["layers"]:
            units = layer.get("units") if isinstance(layer, dict) else None
            rows = layer.get("unit_values") if isinstance(layer, dict) else None
            if (not isinstance(layer.get("path") if isinstance(layer, dict) else None, str) or not isinstance(layer.get("token_rms"), list) or len(layer["token_rms"]) != count
                    or not all(_number(v) for v in layer["token_rms"]) or not isinstance(units, list) or len(units) > 256 or not all(type(u) is int for u in units)
                    or not isinstance(rows, list) or len(rows) != count or any(not isinstance(r, list) or len(r) != len(units) or not all(_number(v) for v in r) for r in rows)):
                raise ValueError("Invalid text encoder layer")
        projection = encoder.get("map")
        if not isinstance(projection, dict) or projection.get("method") != "pca" or not isinstance(projection.get("explained"), list) or not all(_number(v) for v in projection["explained"]):
            raise ValueError("Invalid text encoder map")
        points = projection.get("points")
        if (not isinstance(points, list) or len(points) != len(encoder["layers"])
                or any(not isinstance(layer, list) or len(layer) != count or any(not isinstance(p, list) or not all(_number(c) for c in p) for p in layer) for layer in points)):
            raise ValueError("Invalid text encoder map")


def _validate_logits(view):
    if not isinstance(view, dict) or type(view.get("vocab_size")) is not int or not isinstance(view.get("top"), list) or len(view["top"]) > 512:
        raise ValueError("Invalid logits view")
    for token in view["top"]:
        if not isinstance(token, dict) or type(token.get("token_id")) is not int or not _number(token.get("logit")):
            raise ValueError("Invalid logits token")
    grid = view.get("temperatures")
    if not isinstance(grid, list) or not grid or any(not isinstance(g, dict) or not _number(g.get("temperature")) or not _number(g.get("log_total")) for g in grid):
        raise ValueError("Invalid logits temperatures")
    histogram = view.get("histogram")
    if (not isinstance(histogram, dict) or not isinstance(histogram.get("edges"), list) or not isinstance(histogram.get("counts"), list)
            or len(histogram["edges"]) != len(histogram["counts"]) + 1 or not all(_number(v) for v in histogram["edges"] + histogram["counts"])):
        raise ValueError("Invalid logits histogram")


def _validate_generation(trace):
    """A greedy response with per-token provenance."""
    def _tokens_ok(tokens):
        return (isinstance(tokens, list) and 1 <= len(tokens) <= 100
                and all(isinstance(t, dict) and type(t.get("token_id")) is int and type(t.get("probability")) in (int, float) and 0 <= t["probability"] <= 1 for t in tokens))
    steps = trace.get("steps")
    if (trace.get("decoding") != "greedy" or trace.get("stopped") not in ("max_new_tokens", "eos", "context_limit")
            or not isinstance(trace.get("input"), dict) or type(trace["input"].get("token_count")) is not int
            or not isinstance(steps, list) or not 1 <= len(steps) <= 256 or type(trace.get("provenance")) is not bool):
        raise ValueError("Invalid generation capture")
    for index, step in enumerate(steps):
        if (not isinstance(step, dict) or step.get("index") != index or type(step.get("position")) is not int or step["position"] < 1
                or type(step.get("token_id")) is not int or type(step.get("probability")) not in (int, float) or not 0 <= step["probability"] <= 1
                or type(step.get("entropy_bits")) not in (int, float) or not math.isfinite(step["entropy_bits"]) or not _tokens_ok(step.get("top_tokens"))):
            raise ValueError("Invalid generation step")
        attention = step.get("attention")
        if attention is not None and (not isinstance(attention, list) or len(attention) != step["position"] or any(type(v) not in (int, float) or not 0 <= v <= 1.0001 for v in attention)):
            raise ValueError("Invalid generation attention")
        lens = step.get("lens")
        if lens is not None:
            track = lens.get("chosen_probability_by_layer") if isinstance(lens, dict) else None
            if (not isinstance(track, list) or not 1 <= len(track) <= 1025 or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in track)
                    or any(lens.get(k) is not None and (type(lens.get(k)) is not int or not 0 <= lens[k] < len(track)) for k in ("first_layer", "settled_layer"))):
                raise ValueError("Invalid generation lens")
    tokens = trace.get("tokens")
    if tokens is not None and (not isinstance(tokens, list) or len(tokens) != trace["input"]["token_count"]):
        raise ValueError("Invalid generation prompt tokens")


def _validate_patching(block):
    """An activation-patching result attached to a text capture."""
    def _dist_ok(entry, extra=()):
        tokens = entry.get("top_tokens") if isinstance(entry, dict) else None
        return (isinstance(tokens, list) and 1 <= len(tokens) <= 100
                and all(isinstance(t, dict) and type(t.get("token_id")) is int and type(t.get("probability")) in (int, float) and 0 <= t["probability"] <= 1 for t in tokens)
                and type(entry.get("entropy_bits")) in (int, float) and math.isfinite(entry["entropy_bits"])
                and all(type(entry.get(k)) in (int, float) and math.isfinite(entry[k]) for k in extra))
    if (not isinstance(block, dict) or block.get("method") != "residual_stream_patch" or block.get("positions") not in ("last", "all")
            or type(block.get("layer_count")) is not int or not 1 <= block["layer_count"] <= 1024
            or not _dist_ok(block.get("source_run")) or type(block["source_run"].get("token_count")) is not int
            or not _dist_ok(block.get("target_run"), ("source_answer_probability", "target_answer_probability")) or type(block["target_run"].get("token_count")) is not int
            or not isinstance(block.get("layers"), list) or not 1 <= len(block["layers"]) <= block["layer_count"]
            or not isinstance(block.get("limitations"), list)):
        raise ValueError("Invalid activation patching block")
    tokens = block["source_run"].get("tokens")
    if tokens is not None and (not isinstance(tokens, list) or len(tokens) != block["source_run"]["token_count"] or any(not isinstance(t, dict) or type(t.get("id")) is not int for t in tokens)):
        raise ValueError("Invalid patching source tokens")
    seen = set()
    for entry in block["layers"]:
        if (not _dist_ok(entry, ("kl_bits", "source_answer_probability", "target_answer_probability", "recovery"))
                or type(entry.get("layer")) is not int or not 0 <= entry["layer"] < block["layer_count"] or entry["layer"] in seen
                or entry["kl_bits"] < 0 or type(entry.get("top_is_source")) is not bool):
            raise ValueError("Invalid activation patching layer")
        seen.add(entry["layer"])


def _validate_ablation(ablation, layer):
    """A zero-ablation result: a baseline next-token distribution and one entry per ablated head."""
    def _tokens_ok(entry):
        tokens = entry.get("top_tokens") if isinstance(entry, dict) else None
        return (isinstance(tokens, list) and 1 <= len(tokens) <= 100
                and all(isinstance(t, dict) and type(t.get("token_id")) is int and type(t.get("probability")) in (int, float) and 0 <= t["probability"] <= 1 for t in tokens)
                and type(entry.get("entropy_bits")) in (int, float) and math.isfinite(entry["entropy_bits"]))
    if (not isinstance(ablation, dict) or ablation.get("method") != "zero_ablation" or type(layer) is not int or ablation.get("layer") != layer
            or type(ablation.get("head_count")) is not int or not 1 <= ablation["head_count"] <= 1024
            or not _tokens_ok(ablation.get("baseline")) or not isinstance(ablation.get("heads"), list) or not 1 <= len(ablation["heads"]) <= ablation["head_count"]
            or not isinstance(ablation.get("limitations"), list)):
        raise ValueError("Invalid head ablation block")
    seen = set()
    for entry in ablation["heads"]:
        if (not _tokens_ok(entry) or type(entry.get("head")) is not int or not 0 <= entry["head"] < ablation["head_count"] or entry["head"] in seen
                or type(entry.get("kl_bits")) not in (int, float) or not math.isfinite(entry["kl_bits"]) or entry["kl_bits"] < 0
                or type(entry.get("baseline_top_probability")) not in (int, float) or not 0 <= entry["baseline_top_probability"] <= 1
                or type(entry.get("top_token_changed")) is not bool):
            raise ValueError("Invalid head ablation entry")
        seen.add(entry["head"])


def _validate(trace):
    if not isinstance(trace, dict) or trace.get("schema_version") != "1.0" or trace.get("kind") not in SUPPORTED_KINDS:
        raise ValueError("Unsupported trace schema or kind")
    if not isinstance(trace.get("model_class"), str) or not isinstance(trace.get("torch_version"), str):
        raise ValueError("Trace requires model and runtime provenance")
    if not isinstance(trace.get("limitations"), list) or any(not isinstance(item, str) for item in trace["limitations"]):
        raise ValueError("Trace requires a list of limitations")
    for field in ("model_name", "source", "status", "input_mode", "response"):
        if field in trace and not isinstance(trace[field], str):
            raise ValueError(f"Invalid {field}")
    if "cross_attention" in trace:
        if trace["kind"] not in ("pytorch_module_capture", "diffusion_step_capture"):
            raise ValueError("Cross-attention maps belong to denoiser traces")
        _validate_cross_attention(trace["cross_attention"])
    if "denoising" in trace:
        _validate_denoising(trace["denoising"])
    if "text_encoder" in trace:
        _validate_text_encoder(trace["text_encoder"])
    if "logits_view" in trace:
        _validate_logits(trace["logits_view"])
    if "token_ablation" in trace:
        block = trace["token_ablation"]
        if (trace["kind"] not in ("pytorch_module_capture", "diffusion_step_capture") or not isinstance(block, dict)
                or block.get("method") != "cross_attention_key_mask" or not isinstance(block.get("tokens"), list) or not 1 <= len(block["tokens"]) <= 4096
                or any(type(t) is not int or not 0 <= t < 4096 for t in block["tokens"])
                or not isinstance(block.get("labels"), list) or len(block["labels"]) != len(block["tokens"]) or any(not isinstance(l, str) or len(l) > 48 for l in block["labels"])
                or type(block.get("applied_attention_calls")) is not int or block["applied_attention_calls"] < 0
                or not isinstance(block.get("limitations"), list)):
            raise ValueError("Invalid token ablation block")
    if "patching" in trace:
        if trace["kind"] not in ("pytorch_module_capture", "attention_capture"):
            raise ValueError("Activation patching belongs to text captures")
        _validate_patching(trace["patching"])
    structure = trace.get("structure")
    if structure is not None:
        if not isinstance(structure, dict) or not isinstance(structure.get("nodes"), list) or len(structure["nodes"]) > 10000:
            raise ValueError("Invalid runtime structure")
        if "name" in structure and not isinstance(structure["name"], str):
            raise ValueError("Invalid structure name")
        for node in structure["nodes"]:
            if not isinstance(node, dict) or not isinstance(node.get("path"), str) or not isinstance(node.get("type"), str):
                raise ValueError("Invalid runtime component")
            for field in ("parameters", "buffers"):
                if field not in node:
                    continue
                if not isinstance(node[field], list):
                    raise ValueError("Invalid component tensor metadata")
                for tensor in node[field]:
                    if not isinstance(tensor, dict) or not isinstance(tensor.get("name"), str) or not isinstance(tensor.get("dtype"), str) or not isinstance(tensor.get("shape"), list) or any(type(d) is not int or d < 0 for d in tensor["shape"]):
                        raise ValueError("Invalid component tensor metadata")
    if "tokens" in trace:
        if not isinstance(trace["tokens"], list) or len(trace["tokens"]) > 4096 or any(not isinstance(t, dict) or type(t.get("id")) is not int or not isinstance(t.get("text"), str) for t in trace["tokens"]):
            raise ValueError("Invalid decoded tokens")
    if "input_media" in trace and trace["input_media"] is not None:
        from .media_models import validate_input_media
        validate_input_media(trace["input_media"])
    if "audio_view" in trace:
        from .media_models import validate_audio_view
        validate_audio_view(trace["audio_view"])
    if "image_classification" in trace:
        from .media_models import validate_image_classification
        validate_image_classification(trace["image_classification"])
    if "output" in trace:
        output = trace["output"]
        if not isinstance(output, dict) or not isinstance(output.get("top_tokens"), list) or len(output["top_tokens"]) > 100:
            raise ValueError("Invalid token distribution")
        if "entropy_bits" in output and (type(output["entropy_bits"]) not in (int, float) or not math.isfinite(output["entropy_bits"])):
            raise ValueError("Invalid output entropy")
        for token in output["top_tokens"]:
            if not isinstance(token, dict) or type(token.get("token_id")) is not int or type(token.get("probability")) not in (int, float) or not 0 <= token["probability"] <= 1 or ("text" in token and not isinstance(token["text"], str)):
                raise ValueError("Invalid predicted token")
    if "logit_lens" in trace:
        lens = trace["logit_lens"]
        if not isinstance(lens, dict) or not isinstance(lens.get("layers"), list) or len(lens["layers"]) > 1025 or type(lens.get("final_token_id")) is not int:
            raise ValueError("Invalid logit lens")
        for entry in lens["layers"]:
            if (not isinstance(entry, dict) or type(entry.get("index")) is not int or not isinstance(entry.get("top_tokens"), list) or len(entry["top_tokens"]) > 10
                    or any(type(entry.get(k)) not in (int, float) or not math.isfinite(entry[k]) for k in ("entropy_bits", "final_token_probability", "residual_norm"))):
                raise ValueError("Invalid logit lens layer")
            for token in entry["top_tokens"]:
                if not isinstance(token, dict) or type(token.get("token_id")) is not int or type(token.get("probability")) not in (int, float) or not 0 <= token["probability"] <= 1 or ("text" in token and not isinstance(token["text"], str)):
                    raise ValueError("Invalid logit lens token")
    def _check_png(encoded):
        import base64
        if not isinstance(encoded, str):
            raise ValueError("Invalid preview")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("Invalid preview encoding") from exc
        if len(raw) > 4 * 1024 * 1024 or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Invalid or oversized PNG preview")
    if "preview" in trace:
        preview = trace["preview"]
        if not isinstance(preview, dict) or preview.get("mime") != "image/png":
            raise ValueError("Invalid preview")
        _check_png(preview.get("base64"))
    if "preview_frames" in trace:
        block = trace["preview_frames"]
        frames = block.get("frames") if isinstance(block, dict) else None
        indices = block.get("frame_indices") if isinstance(block, dict) else None
        if (not isinstance(frames, list) or not 2 <= len(frames) <= 64 or not isinstance(indices, list) or len(indices) != len(frames)
                or any(type(i) is not int or i < 0 for i in indices) or type(block.get("source_frames")) is not int):
            raise ValueError("Invalid preview frames")
        for frame in frames:
            _check_png(frame)
    if trace["kind"] == "attention_capture":
        shape, values = trace.get("shape"), trace.get("values")
        if (not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or not 1 <= n <= 128 for n in shape)
                or not isinstance(values, list) or len(values) != shape[0]):
            raise ValueError("Invalid attention matrix shape")
        for row in values:
            if not isinstance(row, list) or len(row) != shape[1] or any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
                raise ValueError("Invalid attention matrix values")
        if trace.get("axes") != ["query_position", "key_position"] or any(type(trace.get(k)) is not int or trace[k] < 0 for k in ("layer", "head")):
            raise ValueError("Invalid attention selection or axes")
        if "heads" in trace:
            heads = trace["heads"]
            if not isinstance(heads, list) or not 1 <= len(heads) <= 256 or len(heads) * shape[0] * shape[1] > 262144:
                raise ValueError("Invalid attention head gallery")
            for matrix in heads:
                if not isinstance(matrix, list) or len(matrix) != shape[0] or any(not isinstance(row, list) or len(row) != shape[1] or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in row) for row in matrix):
                    raise ValueError("Invalid attention head matrix")
        if "attribution" in trace:
            attribution = trace["attribution"]
            if not isinstance(attribution, dict) or not isinstance(attribution.get("values"), list) or len(attribution["values"]) != shape[1] or any(type(v) not in (int, float) or not math.isfinite(v) for v in attribution["values"]):
                raise ValueError("Invalid attention attribution")
            received = attribution.get("last_position_attention")
            if received is not None and (not isinstance(received, list) or len(received) != shape[1] or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in received)):
                raise ValueError("Invalid last-position attention")
        if "ablation" in trace:
            _validate_ablation(trace["ablation"], trace["layer"])
        if "ablations" in trace:
            blocks = trace["ablations"]
            if not isinstance(blocks, list) or len(blocks) > 1024:
                raise ValueError("Invalid ablation list")
            seen = set()
            for block in blocks:
                layer = block.get("layer") if isinstance(block, dict) else None
                if layer in seen:
                    raise ValueError("Duplicate ablation layer")
                seen.add(layer)
                _validate_ablation(block, layer)
        if "layer_stacks" in trace:
            stacks = trace["layer_stacks"]
            if not isinstance(stacks, list) or not 1 <= len(stacks) <= 1024:
                raise ValueError("Invalid layer stacks")
            budget = 0
            for position, stack in enumerate(stacks):
                if not isinstance(stack, dict) or stack.get("index") != position:
                    raise ValueError("Layer stacks must be indexed in order")
                heads = stack.get("heads")
                mean = stack.get("mean")
                if heads is not None:
                    if not isinstance(heads, list) or not 1 <= len(heads) <= 256 or any(not isinstance(m, list) or len(m) != shape[0] or any(not isinstance(r, list) or len(r) != shape[1] or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in r) for r in m) for m in heads):
                        raise ValueError("Invalid layer stack heads")
                    budget += len(heads) * shape[0] * shape[1]
                elif mean is not None:
                    if not isinstance(mean, list) or len(mean) != shape[0] or any(not isinstance(r, list) or len(r) != shape[1] or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in r) for r in mean):
                        raise ValueError("Invalid layer stack mean")
                    budget += shape[0] * shape[1]
                else:
                    raise ValueError("Layer stack needs heads or mean")
            if budget > 1200000:
                raise ValueError("Layer stacks exceed the value bound")
        if "layers" in trace:
            if not isinstance(trace["layers"], list) or len(trace["layers"]) > 1024 or any(not isinstance(e, dict) or type(e.get("index")) is not int or any(type(e.get(k)) not in (int, float) or not math.isfinite(e[k]) for k in ("sink_mass", "diagonal_mass", "previous_token_mass", "entropy_bits")) for e in trace["layers"]):
                raise ValueError("Invalid attention layer summaries")
    elif trace["kind"] == "head_ablation":
        _validate_ablation(trace, trace.get("layer"))
    elif trace["kind"] == "activation_patching":
        _validate_patching(trace)
    elif trace["kind"] == "generation_capture":
        _validate_generation(trace)
    elif trace["kind"] == "pytorch_module_capture":
        events, capture = trace.get("events"), trace.get("capture")
        if not isinstance(events, list) or len(events) > 1024 or not isinstance(capture, dict):
            raise ValueError("Invalid capture events")
        for event in events:
            if not isinstance(event, dict) or not isinstance(event.get("module"), str) or event.get("status") not in ("captured", "no_tensor_output", "unsupported_tensor"):
                raise ValueError("Invalid capture event")
            for field in ("dtype", "device", "selection"):
                if field in event and not isinstance(event[field], str):
                    raise ValueError(f"Invalid event {field}")
            if "denoiser_call" in event and (type(event["denoiser_call"]) is not int or event["denoiser_call"] < 0):
                raise ValueError("Invalid denoiser call index")
            if event["status"] == "captured":
                values = event.get("values")
                shape = event.get("shape")
                if (not isinstance(values, list) or len(values) > 65536 or not isinstance(shape, list)
                        or any(type(d) is not int or d < 0 for d in shape)
                        or any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in values)):
                    raise ValueError("Invalid captured tensor")
                total = event.get("total_elements")
                if type(total) is not int or total != math.prod(shape) or len(values) > total or type(event.get("truncated")) is not bool or event["truncated"] != (len(values) < total):
                    raise ValueError("Captured tensor count or truncation is inconsistent")
                if "slices" in event:
                    validate_slices(event["slices"])
    else:
        events = trace.get("events")
        if not isinstance(events, list) or len(events) > 256 or not isinstance(trace.get("capture"), dict):
            raise ValueError("Invalid diffusion events")
        for event in events:
            if not isinstance(event, dict) or type(event.get("step")) is not int or event["step"] < 0 or type(event.get("timestep")) not in (int, float) or not math.isfinite(event["timestep"]):
                raise ValueError("Invalid diffusion step")
            values, shape = event.get("values"), event.get("shape")
            if not isinstance(values, list) or len(values) > 65536 or not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape) or any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in values):
                raise ValueError("Invalid diffusion tensor")
            total = event.get("total_elements")
            if type(total) is not int or total != math.prod(shape) or len(values) > total or type(event.get("truncated")) is not bool or event["truncated"] != (len(values) < total):
                raise ValueError("Diffusion tensor count or truncation is inconsistent")
            if "slices" in event:
                validate_slices(event["slices"])


def encode_trace(trace):
    """Validate and encode while bounding the accumulated serialized output."""
    _validate(trace)
    data = bytearray()
    try:
        encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        for chunk in encoder.iterencode(trace):
            encoded = chunk.encode("utf-8")
            if len(data) + len(encoded) > MAX_TRACE_BYTES:
                raise ValueError("Trace exceeds the size limit")
            data.extend(encoded)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"Trace cannot be serialized: {exc}") from exc
    return bytes(data)


def save_trace(path, trace):
    """Atomically replace a caller-chosen local path; never save without an explicit call."""
    data = encode_trace(trace)
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".model-run-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_trace(path):
    path = Path(path)
    with path.open("rb") as stream:
        data = stream.read(MAX_TRACE_BYTES + 1)
    if len(data) > MAX_TRACE_BYTES:
        raise ValueError("Trace exceeds the size limit")
    try:
        trace = json.loads(data, object_pairs_hook=_unique_pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Invalid JSON number")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"Invalid trace JSON: {exc}") from exc
    _validate(trace)
    return trace
