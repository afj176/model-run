"""Attention weights of a ComfyUI text model's prompt pass.

ComfyUI's text models call a fused attention kernel that never returns its weights. During the prompt
pass, each text layer's query and key are taken as the kernel receives them, and softmax(QK^T * scale +
mask) is computed beside it, with the same mask and scale. The kernel's own output is what the model
uses; nothing it computes is changed.
"""
import logging
import re
import sys
from contextlib import contextmanager

from . import capture_limits

# Standard prompt length; the active one is the capture size's prompt tokens.
MAX_TOKENS = 128
_LAYER = re.compile(r"^(.*?(?:^|\.)(?:layers?|blocks?))\.(\d+)\.")


def layer_slot(path):
    """(stack container, layer number) of an attention module's path, or None."""
    match = _LAYER.match(path)
    return (match.group(1), int(match.group(2))) if match else None


class AttentionWatch:
    """Records the first call of each listed attention module, while ``watching`` is active."""

    def __init__(self, stack, attention_paths):
        self.paths = list(attention_paths)
        self.names = dict(stack.named_modules())
        self.current = None
        self.found = {}
        self.skipped = None
        self.side = capture_limits.get("attention_side")
        self.longest = capture_limits.get("prompt_tokens")
        self.length = None  # the prompt's tokens, before any shrinking
        self.ranges = None

    @contextmanager
    def watching(self):
        handles, patched = [], []
        for path in self.paths:
            module = self.names.get(path)
            if module is None:
                continue
            handles.append(module.register_forward_pre_hook(lambda _m, _a, p=path: self._enter(p)))
            handles.append(module.register_forward_hook(lambda _m, _a, _o: self._leave()))
        # Text models look the factory up by name at call time, so replacing it in their modules is enough.
        for name, module in list(sys.modules.items()):
            factory = getattr(module, "optimized_attention_for_device", None) if name.startswith("comfy.text_encoders.") else None
            if callable(factory):
                setattr(module, "optimized_attention_for_device", self._factory(factory))
                patched.append((module, factory))
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()
            for module, factory in patched:
                setattr(module, "optimized_attention_for_device", factory)
            self.current = None

    def _enter(self, path):
        self.current = path

    def _leave(self):
        self.current = None

    def _factory(self, original):
        def for_device(*args, **kwargs):
            return self._wrap(original(*args, **kwargs))
        return for_device

    def _wrap(self, kernel):
        def attention(q, k, v, heads, *args, **kwargs):
            out = kernel(q, k, v, heads, *args, **kwargs)
            path = self.current
            if path is not None and path not in self.found:
                try:
                    weights = self._weights(q, k, heads, kwargs)
                except Exception as exc:  # noqa: BLE001 - observation must never break the model
                    self.skipped = f"{type(exc).__name__}: {exc}"[:200]
                    weights = None
                if weights is not None:
                    self.found[path] = weights
            return out
        return attention

    def _weights(self, q, k, heads, options):
        import torch
        if options.get("skip_reshape"):
            query, key = q, k  # [B, H, T, D]
        else:
            batch, length, width = q.shape
            depth = width // heads
            query = q.view(batch, length, heads, depth).transpose(1, 2)
            key = k.view(k.shape[0], k.shape[1], -1, depth).transpose(1, 2)
        length, depth = int(query.shape[2]), int(query.shape[3])
        # A single query is a generated token reading the cache; only the prompt pass is recorded.
        if length < 2 or length > self.longest or int(key.shape[2]) != length:
            return None
        query = query[:1].float()
        key = key[:1].float()
        if key.shape[1] != query.shape[1]:
            key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        scale = options.get("scale")
        scores = (query @ key.transpose(-1, -2)) * (scale if scale is not None else depth ** -0.5)
        mask = options.get("mask")
        if mask is not None:
            mask = mask[:1] if mask.ndim == 4 else mask
            mask = mask[..., -length:, -length:]
            scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask.float()
        weights = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        # A long prompt is shrunk to blocks layer by layer, so every layer's full matrix is never held at once.
        pooled, ranges = capture_limits.pool_attention(weights[0], self.side)
        self.length, self.ranges = length, ranges
        return pooled.cpu()  # [H, T, T]


def attention_trace(watch, tokens, model_class):
    """An attention_capture trace from the watched layers, the same shape as a standalone attention probe."""
    import torch
    from .transformers_capture import _layer_summaries, _rollout, keep_attention_heads, keep_attention_shape, keep_layer_stacks
    # The language model's own stack: the group with the most recorded layers, numbered from 0 without gaps.
    groups = {}
    for path, weights in watch.found.items():
        slot = layer_slot(path)
        if slot is not None:
            groups.setdefault(slot[0], {})[slot[1]] = weights
    if not groups:
        return None
    container = max(groups, key=lambda name: len(groups[name]))
    layers = groups[container]
    cpu, size = [], None
    for index in range(len(layers)):
        weights = layers.get(index)
        if weights is None or (size is not None and weights.shape[-1] != size):
            break
        size = weights.shape[-1]
        cpu.append(weights.unsqueeze(0))
    if not cpu:
        return None
    if not all(torch.isfinite(a).all().item() for a in cpu):
        logging.getLogger(__name__).warning("Model Run skipped attention with non-finite weights")
        return None
    heads = int(cpu[0].shape[1])
    layer, head = 0, 0
    selected = cpu[layer][0, head]
    trace = {"schema_version": "1.0", "kind": "attention_capture", "model_class": model_class, "torch_version": str(torch.__version__),
             "layer": layer, "head": head, "head_count": heads, "layer_count": len(cpu),
             "attention_implementation": "comfyui (weights recomputed beside the fused kernel)",
             "axes": ["query_position", "key_position"], "shape": [size, size], "values": selected.tolist(),
             "input": {"token_count": watch.length or size, "token_ids_included": False},
             "limitations": ["Attention weights alone do not establish causality.",
                             "ComfyUI's fused kernel returns no weights, so each layer's softmax(QK^T * scale + mask) was computed beside it from the same query, key, mask and scale.",
                             "Prompt pass only: generated tokens read the same layers later and are not recorded."]}
    keep_attention_shape(trace, watch.length or size, watch.ranges, heads)
    keep_attention_heads(trace, cpu, layer, size, capture_limits.get("head_values"))
    rolled = _rollout(cpu)
    received = torch.stack([a[0, :, -1, :].mean(0) for a in cpu]).mean(0)
    trace["attribution"] = {"method": "attention_rollout", "position": size - 1,
                            "values": [float(v) for v in rolled[-1].tolist()],
                            "last_position_attention": [min(1.0, max(0.0, float(v))) for v in received.tolist()],
                            "limitations": ["Rollout assumes equal head weighting and a residual identity term; it is an attention-flow estimate, not a causal attribution.",
                                            "last_position_attention is the measured head- and layer-averaged attention from the final query position."]}
    trace["layers"] = _layer_summaries(cpu)
    keep_layer_stacks(trace, cpu, size, capture_limits.get("stack_values"))
    length = watch.length or size
    if tokens and len(tokens) >= length:
        # Real token ids where the tokenizer gave them; the text is what the views show.
        trace["tokens"] = [{"id": t["id"] if type(t.get("id")) is int else i, "text": str(t["text"])[:64]} for i, t in enumerate(tokens[:length])]
        if watch.ranges:
            trace["tokens"] = capture_limits.pooled_tokens(trace["tokens"], watch.ranges)
        trace["limitations"].append("The decoded prompt tokens are included; exported traces may contain private text.")
    trace["stack"] = container  # where these layers live; the caller takes it off
    if len(cpu) < len(layers):
        trace["limitations"].append(f"Recorded {len(cpu)} of {len(layers)} layers in order; the rest were missing or differently sized.")
    if watch.skipped:
        trace["limitations"].append(f"Some layers could not be measured ({watch.skipped}).")
    return trace
