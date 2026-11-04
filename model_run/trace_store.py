"""Explicit bounded JSON trace export and import."""
import json
import os
import tempfile
import math
from pathlib import Path

MAX_TRACE_BYTES = 16 * 1024 * 1024
SUPPORTED_KINDS = {"pytorch_module_capture", "attention_capture", "diffusion_step_capture", "head_ablation"}


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
    if labels is not None and (not isinstance(labels, list) or len(labels) > 1024 or any(not isinstance(l, str) or len(l) > 48 for l in labels)):
        raise ValueError("Invalid cross-attention token labels")
    skipped = section.get("skipped")
    if skipped is not None and (not isinstance(skipped, list) or len(skipped) > 32 or any(not isinstance(s, dict) or not isinstance(s.get("reason"), str) for s in skipped)):
        raise ValueError("Invalid cross-attention skip list")
    for event in section["events"]:
        if (not isinstance(event, dict) or type(event.get("denoiser_call")) is not int or event["denoiser_call"] < 0
                or event.get("attention_kind") not in ("cross", "joint") or type(event.get("text_tokens")) is not int or not 1 <= event["text_tokens"] <= 1024
                or not isinstance(event.get("grid"), list) or len(event["grid"]) != 2 or any(type(n) is not int or not 1 <= n <= 64 for n in event["grid"])
                or not isinstance(event.get("source_grid"), list) or len(event["source_grid"]) != 2 or any(type(n) is not int or n < 1 for n in event["source_grid"])
                or type(event.get("heads")) is not int or not isinstance(event.get("batch"), str) or not isinstance(event.get("block"), str)):
            raise ValueError("Invalid cross-attention event")
        cells = event["grid"][0] * event["grid"][1]
        maps = event.get("maps")
        if (not isinstance(maps, list) or len(maps) != event["text_tokens"] or event["text_tokens"] * cells > 262144
                or any(not isinstance(row, list) or len(row) != cells or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in row) for row in maps)):
            raise ValueError("Invalid cross-attention maps")
        for key, length in (("text_mass", cells), ("token_mass", event["text_tokens"])):
            values = event.get(key)
            if not isinstance(values, list) or len(values) != length or any(type(v) not in (int, float) or not 0 <= v <= 1.0001 for v in values):
                raise ValueError(f"Invalid cross-attention {key}")


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
    for field in ("model_name", "source", "status", "input_mode"):
        if field in trace and not isinstance(trace[field], str):
            raise ValueError(f"Invalid {field}")
    if "cross_attention" in trace:
        if trace["kind"] not in ("pytorch_module_capture", "diffusion_step_capture"):
            raise ValueError("Cross-attention maps belong to denoiser traces")
        _validate_cross_attention(trace["cross_attention"])
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
    if "preview" in trace:
        import base64
        preview = trace["preview"]
        if not isinstance(preview, dict) or preview.get("mime") != "image/png" or not isinstance(preview.get("base64"), str):
            raise ValueError("Invalid preview")
        try:
            raw = base64.b64decode(preview["base64"], validate=True)
        except ValueError as exc:
            raise ValueError("Invalid preview encoding") from exc
        if len(raw) > 4 * 1024 * 1024 or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Invalid or oversized PNG preview")
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
        if "layers" in trace:
            if not isinstance(trace["layers"], list) or len(trace["layers"]) > 1024 or any(not isinstance(e, dict) or type(e.get("index")) is not int or any(type(e.get(k)) not in (int, float) or not math.isfinite(e[k]) for k in ("sink_mass", "diagonal_mass", "previous_token_mass", "entropy_bits")) for e in trace["layers"]):
                raise ValueError("Invalid attention layer summaries")
    elif trace["kind"] == "head_ablation":
        _validate_ablation(trace, trace.get("layer"))
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
