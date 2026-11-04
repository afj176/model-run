"""Text-model adapter for already-loaded Hugging Face causal language models."""
import math

from .capture import capture_forward

# Aggregate bound for optional per-head matrices returned with an attention capture.
MAX_HEAD_VALUES = 262144
ROUND_DIGITS = 5


def _entropy_bits(probabilities):
    """Shannon entropy in bits of a 1-D probability tensor."""
    import torch
    p = probabilities[probabilities > 0]
    return float(-(p * torch.log2(p)).sum().item())


def _final_norm(model):
    """Locate the decoder's final normalization module, if the architecture exposes one."""
    candidates = []
    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        try:
            candidates.append(get_decoder())
        except Exception:
            pass
    candidates.append(getattr(model, "base_model", None))
    candidates.append(getattr(model, "model", None))
    candidates.append(getattr(model, "transformer", None))
    import torch
    for module in candidates:
        if not isinstance(module, torch.nn.Module):
            continue
        for name in ("norm", "final_layer_norm", "ln_f", "final_norm", "layer_norm"):
            norm = getattr(module, name, None)
            if isinstance(norm, torch.nn.Module):
                return norm
    return None


def _logit_lens(model, hidden_states, final_logits, top_k):
    """Project the last-position residual stream of every layer through the output head."""
    import torch
    norm = _final_norm(model)
    head = model.get_output_embeddings()
    final_token = int(torch.argmax(final_logits).item())
    layers = []
    with torch.inference_mode():
        for index, state in enumerate(hidden_states):
            if not isinstance(state, torch.Tensor) or state.ndim != 3 or state.shape[0] != 1:
                raise ValueError("Unexpected hidden state shape")
            vector = state[0, -1].detach()
            residual_norm = float(vector.float().norm().item())
            projected = vector.unsqueeze(0).unsqueeze(0)
            if norm is not None:
                projected = norm(projected)
            logits = head(projected)[0, -1].float()
            if not torch.isfinite(logits).all().item():
                raise ValueError("Logit lens produced non-finite values")
            probabilities = torch.softmax(logits, dim=-1)
            values, indices = torch.topk(probabilities, min(top_k, probabilities.numel()))
            layers.append({"index": index, "label": "embeddings" if index == 0 else f"layer {index - 1}",
                           "top_tokens": [{"token_id": int(i), "probability": float(p)} for i, p in zip(indices.tolist(), values.tolist())],
                           "entropy_bits": _entropy_bits(probabilities),
                           "final_token_probability": float(probabilities[final_token].item()),
                           "residual_norm": residual_norm})
    return {"method": "final_norm_and_output_embeddings" if norm is not None else "output_embeddings_without_norm",
            "position": "last_input_token", "final_token_id": final_token, "layers": layers,
            "limitations": ["Intermediate predictions apply the final norm and output head to earlier residual states; they are a lens, not the model's own output.",
                            "Index 0 is the embedding output; index k is the residual stream after decoder layer k-1."]}


def capture_next_token(model, input_ids, module_paths, *, max_tokens=256, top_k=10, max_tensor_elements=4096, logit_lens=True, control=None):
    """Observe a single eager forward pass; no model loading or tokenization is performed."""
    import torch

    if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
        raise ValueError("max_tokens must be between 1 and 4096")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must be a batch of one token sequence")
    if input_ids.dtype not in (torch.int32, torch.int64) or not 1 <= input_ids.shape[1] <= max_tokens:
        raise ValueError("input_ids must contain integer tokens within the configured length limit")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    if type(logit_lens) is not bool:
        raise ValueError("logit_lens must be boolean")
    inputs = {"input_ids": input_ids, "use_cache": False}
    if logit_lens:
        inputs["output_hidden_states"] = True
    output, trace = capture_forward(model, inputs, module_paths, max_tensor_elements=max_tensor_elements, control=control)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("Model did not return causal language model logits")
    final_logits = logits[0, -1].detach().float()
    if not torch.isfinite(final_logits).all().item():
        raise ValueError("Model returned non-finite logits")
    probabilities = torch.softmax(final_logits, dim=-1)
    values, indices = torch.topk(probabilities, min(top_k, probabilities.numel()))
    trace["task"] = "causal_lm_next_token"
    trace["input"] = {"token_count": int(input_ids.shape[1]), "token_ids_included": False}
    trace["output"] = {"position": int(input_ids.shape[1] - 1), "distribution": "raw_softmax",
                       "entropy_bits": _entropy_bits(probabilities),
                       "top_tokens": [{"token_id": int(i), "probability": float(p)} for i, p in zip(indices.tolist(), values.tolist())]}
    hidden_states = getattr(output, "hidden_states", None)
    if logit_lens and isinstance(hidden_states, (tuple, list)) and hidden_states and len(hidden_states) <= 1025:
        trace["logit_lens"] = _logit_lens(model, hidden_states, final_logits, min(top_k, 5))
    elif logit_lens:
        trace["limitations"].append("The model did not return per-layer hidden states; no logit lens was recorded.")
    trace["limitations"].append("No tokenizer, chat template, sampling or generated text is captured.")
    return output, trace


def _rollout(attentions):
    """Attention rollout (Abnar & Zuidema 2020): residual-adjusted head-mean attention multiplied across layers."""
    import torch
    rolled = None
    for layer in attentions:
        mean = layer[0].detach().float().mean(0)
        adjusted = mean + torch.eye(mean.shape[0])
        adjusted = adjusted / adjusted.sum(-1, keepdim=True)
        rolled = adjusted if rolled is None else adjusted @ rolled
    return rolled


def _layer_summaries(attentions):
    """Per-layer head-averaged attention statistics on the full matrix: sink mass, previous-token mass, entropy."""
    import torch
    summaries = []
    for index, layer in enumerate(attentions):
        matrix = layer[0].detach().float()  # heads × T × T
        heads, size, _ = matrix.shape
        rows = torch.arange(size)
        sink = matrix[:, :, 0].mean().item()
        diagonal = matrix[:, rows, rows].mean().item()
        previous = matrix[:, rows[1:], rows[1:] - 1].mean().item() if size > 1 else 0.0
        clamped = matrix.clamp_min(1e-12)
        entropy = -(matrix * torch.log2(clamped)).sum(-1).mean().item()
        summaries.append({"index": index, "heads": heads, "sink_mass": sink, "diagonal_mass": diagonal,
                          "previous_token_mass": previous, "entropy_bits": entropy})
    return summaries


def capture_attention(model, input_ids, layer, head, *, max_tokens=128, all_heads=True, control=None):
    """Capture one returned attention matrix; the runtime may materialize all layers first.

    When ``all_heads`` is set and the result fits the aggregate bound, every head of the
    selected layer is returned (rounded), together with attention-rollout attribution for the
    final input position and per-layer head-averaged summaries across all layers.
    """
    import torch
    if any(module.training for module in model.modules()):
        raise ValueError("all model modules must already be in evaluation mode")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 128:
        raise ValueError("max_tokens must be between 1 and 128")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype not in (torch.int32, torch.int64) or not 1 <= input_ids.shape[1] <= max_tokens:
        raise ValueError("input_ids must be a bounded batch of one integer token sequence")
    if type(layer) is not int or layer < 0 or type(head) is not int or head < 0:
        raise ValueError("layer and head must be nonnegative integers")
    if type(all_heads) is not bool:
        raise ValueError("all_heads must be boolean")
    if control is not None:
        control.checkpoint("attention_start")
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False, output_attentions=True)
    if control is not None:
        control.checkpoint("attention_forward_complete")
    attentions = getattr(output, "attentions", None)
    if attentions is None or layer >= len(attentions) or attentions[layer] is None:
        raise ValueError("Requested attention is unavailable; use a compatible attention implementation")
    matrix = attentions[layer]
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 4 or matrix.shape[0] != 1 or head >= matrix.shape[1]:
        raise ValueError("Requested attention head or tensor shape is unavailable")
    selected = matrix[0, head].detach().float().cpu()
    if not torch.isfinite(selected).all().item():
        raise ValueError("Attention contains non-finite values")
    trace = {"schema_version": "1.0", "kind": "attention_capture", "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
             "torch_version": str(torch.__version__), "layer": layer, "head": head, "head_count": int(matrix.shape[1]),
             "layer_count": len(attentions),
             "attention_implementation": getattr(model.config, "_attn_implementation", None),
             "axes": ["query_position", "key_position"], "shape": list(selected.shape), "values": selected.tolist(),
             "input": {"token_count": int(input_ids.shape[1]), "token_ids_included": False},
             "limitations": ["Attention weights alone do not establish causality.", "Runtime may materialize attention for every layer and head.", "No tokenizer or input text is recorded."]}
    if all_heads:
        complete = all(isinstance(a, torch.Tensor) and a.ndim == 4 and a.shape[0] == 1 for a in attentions)
        if not complete:
            trace["limitations"].append("Some layers returned no attention; head gallery, rollout and layer summaries were skipped.")
        else:
            cpu = [a.detach().float().cpu() for a in attentions]
            if not all(torch.isfinite(a).all().item() for a in cpu):
                raise ValueError("Attention contains non-finite values")
            size = int(selected.shape[0])
            if int(matrix.shape[1]) * size * size <= MAX_HEAD_VALUES:
                trace["heads"] = [[[round(v, ROUND_DIGITS) for v in row] for row in h.tolist()] for h in cpu[layer][0]]
                trace["limitations"].append(f"Per-head matrices are rounded to {ROUND_DIGITS} decimals; the selected head keeps full precision in values.")
            else:
                trace["limitations"].append("Per-head matrices exceed the aggregate bound and were not recorded.")
            rolled = _rollout(cpu)
            # Directly measured: how much the final query position attends to each key, averaged over every layer and head.
            received = torch.stack([a[0, :, -1, :].mean(0) for a in cpu]).mean(0)
            trace["attribution"] = {"method": "attention_rollout", "position": size - 1,
                                    "values": [float(v) for v in rolled[-1].tolist()],
                                    "last_position_attention": [float(v) for v in received.tolist()],
                                    "limitations": ["Rollout assumes equal head weighting and a residual identity term; it is an attention-flow estimate, not a causal attribution.",
                                                    "last_position_attention is the measured head- and layer-averaged attention from the final query position."]}
            trace["layers"] = _layer_summaries(cpu)
    return trace


MAX_ABLATION_HEADS = 128
_ATTENTION_NAMES = ("self_attn", "attn", "attention", "self_attention")
_PROJECTION_NAMES = ("o_proj", "out_proj", "c_proj", "dense", "wo", "proj")


def _attention_projection(model, layer):
    """Find the output projection of decoder layer ``layer`` and the per-head width of its input."""
    import torch
    attention = None
    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-1] in _ATTENTION_NAMES and parts[-2] == str(layer):
            attention = module
            break
    if attention is None:
        raise ValueError("Requested attention layer is unavailable")
    projection = next((getattr(attention, name) for name in _PROJECTION_NAMES
                       if isinstance(getattr(attention, name, None), torch.nn.Module)), None)
    if projection is None:
        raise ValueError("The attention module has no recognised output projection; head ablation is unavailable")
    weight = getattr(projection, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("The attention output projection has no 2-D weight")
    in_features = weight.shape[1] if isinstance(projection, torch.nn.Linear) else weight.shape[0]
    config = getattr(model, "config", None)
    head_count = getattr(attention, "num_heads", None) or getattr(config, "num_attention_heads", None)
    head_dim = getattr(attention, "head_dim", None) or getattr(config, "head_dim", None)
    if type(head_count) is not int or head_count < 1:
        raise ValueError("The attention head count is unavailable")
    if type(head_dim) is not int or head_dim < 1:
        head_dim = in_features // head_count
    if head_count * head_dim != in_features:
        raise ValueError("The attention output projection width does not match head_count x head_dim")
    return projection, head_count, head_dim


def _distribution(logits, top_k):
    import torch
    probabilities = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probabilities, min(top_k, probabilities.numel()))
    return probabilities, {"top_tokens": [{"token_id": int(i), "probability": float(p)} for i, p in zip(indices.tolist(), values.tolist())],
                           "entropy_bits": _entropy_bits(probabilities)}


def capture_head_ablation(model, input_ids, layer, heads=None, *, max_tokens=128, top_k=10, control=None):
    """Zero one attention head's output at a time and record how the next-token distribution moves.

    The head's slice of the input to the layer's output projection is zeroed during a full
    forward pass, so the measurement is a zero ablation of the head's contribution at every
    position. ``heads`` lists head indices; ``None`` sweeps every head of the layer.
    """
    import torch
    if any(module.training for module in model.modules()):
        raise ValueError("all model modules must already be in evaluation mode")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 128:
        raise ValueError("max_tokens must be between 1 and 128")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype not in (torch.int32, torch.int64) or not 1 <= input_ids.shape[1] <= max_tokens:
        raise ValueError("input_ids must be a bounded batch of one integer token sequence")
    if type(layer) is not int or layer < 0:
        raise ValueError("layer must be a nonnegative integer")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    projection, head_count, head_dim = _attention_projection(model, layer)
    if heads is None:
        heads = list(range(head_count))
    if not isinstance(heads, list) or not heads or len(heads) != len(set(heads)) or any(type(h) is not int or not 0 <= h < head_count for h in heads):
        raise ValueError("heads must list distinct head indices of the selected layer")
    if len(heads) > MAX_ABLATION_HEADS:
        raise ValueError(f"At most {MAX_ABLATION_HEADS} heads can be ablated per request")
    if control is not None:
        control.checkpoint("ablation_start")

    def forward():
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError("Model did not return causal language model logits")
        last = logits[0, -1].detach().float().cpu()
        if not torch.isfinite(last).all().item():
            raise ValueError("Model returned non-finite logits")
        return last

    baseline_probabilities, baseline = _distribution(forward(), top_k)
    baseline_top = baseline["top_tokens"][0]["token_id"]
    baseline_log = torch.log(baseline_probabilities.clamp_min(1e-30))
    results = []
    for position, head in enumerate(heads):
        if control is not None:
            control.check()
        start, stop = head * head_dim, (head + 1) * head_dim

        def zero_head(_module, args):
            value = args[0]
            if not isinstance(value, torch.Tensor) or value.shape[-1] != head_count * head_dim:
                raise ValueError("Unexpected input to the attention output projection")
            value = value.clone()
            value[..., start:stop] = 0
            return (value, *args[1:])

        handle = projection.register_forward_pre_hook(zero_head)
        try:
            probabilities, ablated = _distribution(forward(), top_k)
        finally:
            handle.remove()
        kl = float((baseline_probabilities * (baseline_log - torch.log(probabilities.clamp_min(1e-30)))).sum().item())
        results.append({"head": head, **ablated, "kl_bits": max(0.0, kl / math.log(2)),
                        "baseline_top_probability": float(probabilities[baseline_top].item()),
                        "top_token_changed": ablated["top_tokens"][0]["token_id"] != baseline_top})
        if control is not None:
            control.checkpoint(f"ablation:head:{head}", completed=position + 1, total=len(heads))
    return {"schema_version": "1.0", "kind": "head_ablation", "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "torch_version": str(torch.__version__), "method": "zero_ablation", "layer": layer, "head_count": head_count, "head_dim": head_dim,
            "position": int(input_ids.shape[1] - 1), "baseline": baseline, "heads": results,
            "limitations": ["Zero ablation removes the head's output at every position; it is an intervention, not the model's natural behaviour, and a zeroed head is off-distribution.",
                            "Effects of ablating several heads together are not additive and were not measured.",
                            "kl_bits is the KL divergence from the baseline to the ablated next-token distribution at the last input position."]}


_LAYER_PATTERN = r"(?:^|\.)(?:layers|blocks|h|transformer_blocks)\.(\d+)$"


def _decoder_layers(model):
    """Decoder layer modules in index order, found by name; refuses ambiguous trees."""
    import re
    found = {}
    for name, module in model.named_modules():
        match = re.search(_LAYER_PATTERN, name)
        if match:
            index = int(match.group(1))
            if index in found:
                raise ValueError("Ambiguous decoder layer naming; activation patching is unavailable")
            found[index] = module
    if not found or sorted(found) != list(range(len(found))):
        raise ValueError("Decoder layers are not contiguous; activation patching is unavailable")
    return [found[i] for i in range(len(found))]


def _hidden_of(output):
    import torch
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ValueError("Decoder layer output is not a hidden-state tensor")


def capture_activation_patching(model, source_ids, target_ids, *, layers=None, positions="last", max_tokens=128, top_k=10, control=None):
    """Copy the source run's residual stream into the target run, one decoder layer at a time.

    For each layer the layer's output hidden state at the patched positions is replaced with the
    source run's value, the forward continues, and the next-token distribution at the last target
    position is recorded. ``positions`` is "last" (the final position of each prompt) or "all"
    (every position; prompts must tokenise to the same length).
    """
    import torch
    if any(module.training for module in model.modules()):
        raise ValueError("all model modules must already be in evaluation mode")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 128:
        raise ValueError("max_tokens must be between 1 and 128")
    for ids in (source_ids, target_ids):
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype not in (torch.int32, torch.int64) or not 1 <= ids.shape[1] <= max_tokens:
            raise ValueError("input_ids must be a bounded batch of one integer token sequence")
    if positions not in ("last", "all"):
        raise ValueError("positions must be last or all")
    if positions == "all" and source_ids.shape[1] != target_ids.shape[1]:
        raise ValueError("Patching every position needs prompts of the same token length")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    modules = _decoder_layers(model)
    if layers is None:
        layers = list(range(len(modules)))
    if not isinstance(layers, list) or not layers or len(layers) != len(set(layers)) or any(type(l) is not int or not 0 <= l < len(modules) for l in layers):
        raise ValueError("layers must list distinct decoder layer indices")
    if len(layers) > 256:
        raise ValueError("At most 256 layers can be patched per request")
    source_slice = slice(None) if positions == "all" else slice(source_ids.shape[1] - 1, source_ids.shape[1])
    target_slice = slice(None) if positions == "all" else slice(target_ids.shape[1] - 1, target_ids.shape[1])
    if control is not None:
        control.checkpoint("patching_start")

    def forward(ids):
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False).logits
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError("Model did not return causal language model logits")
        last = logits[0, -1].detach().float().cpu()
        if not torch.isfinite(last).all().item():
            raise ValueError("Model returned non-finite logits")
        return last

    # Source run: keep each layer's output at the patched positions.
    stored = {}
    handles = []
    try:
        for index, module in enumerate(modules):
            def keep(_module, _args, output, index=index):
                stored[index] = _hidden_of(output)[:, source_slice].detach().clone()
            handles.append(module.register_forward_hook(keep))
        source_probabilities, source = _distribution(forward(source_ids), top_k)
    finally:
        for handle in handles:
            handle.remove()
    if len(stored) != len(modules):
        raise ValueError("Some decoder layers produced no hidden state during the source run")
    target_probabilities, target = _distribution(forward(target_ids), top_k)
    source_top = source["top_tokens"][0]["token_id"]
    target_top = target["top_tokens"][0]["token_id"]
    base_source = float(target_probabilities[source_top].item())
    base_target = float(target_probabilities[target_top].item())
    source_own = float(source_probabilities[source_top].item())
    target_log = torch.log(target_probabilities.clamp_min(1e-30))
    results = []
    for position, layer in enumerate(layers):
        if control is not None:
            control.check()
        replacement = stored[layer]

        def patch_output(_module, _args, output, replacement=replacement):
            hidden = _hidden_of(output)
            patched = hidden.clone()
            patched[:, target_slice] = replacement.to(device=hidden.device, dtype=hidden.dtype)
            if isinstance(output, torch.Tensor):
                return patched
            return (patched, *tuple(output)[1:])

        handle = modules[layer].register_forward_hook(patch_output)
        try:
            probabilities, patched = _distribution(forward(target_ids), top_k)
        finally:
            handle.remove()
        kl = float((target_probabilities * (target_log - torch.log(probabilities.clamp_min(1e-30)))).sum().item())
        after_source = float(probabilities[source_top].item())
        after_target = float(probabilities[target_top].item())
        span = source_own - base_source
        results.append({"layer": layer, **patched, "kl_bits": max(0.0, kl / math.log(2)),
                        "source_answer_probability": after_source, "target_answer_probability": after_target,
                        "recovery": (after_source - base_source) / span if abs(span) > 1e-9 else 0.0,
                        "top_is_source": patched["top_tokens"][0]["token_id"] == source_top})
        if control is not None:
            control.checkpoint(f"patching:layer:{layer}", completed=position + 1, total=len(layers))
    return {"schema_version": "1.0", "kind": "activation_patching", "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "torch_version": str(torch.__version__), "method": "residual_stream_patch", "positions": positions,
            "layer_count": len(modules), "source_run": {**source, "token_count": int(source_ids.shape[1])},
            "target_run": {**target, "token_count": int(target_ids.shape[1]),
                       "source_answer_probability": base_source, "target_answer_probability": base_target},
            "layers": results,
            "limitations": ["Each layer is patched on its own; effects of patching several layers together are not measured.",
                            "The patched state is the layer's output residual stream at the chosen positions; attention from later positions still sees the target prompt.",
                            "recovery is the fraction of the gap between the target's and the source's own probability of the source answer that the patch restores; it can exceed 1 or go negative.",
                            "kl_bits is the KL divergence from the target baseline to the patched next-token distribution."]}
