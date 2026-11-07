"""The parts inside a text layer worth recording beside it: its attention and its MLP.

A transformer layer adds what its attention read from other tokens and what its MLP computed for each
token. Recording those two outputs shows how much each part changed every token, not only the result.
"""
import re

ATTENTION = re.compile(r"^(self_attn|self_attention|attn|attention)$")
MLP = re.compile(r"^(mlp|feed_forward|ffn)$")
# Features kept per token for a part: enough for its picture; its RMS per token is exact regardless.
PART_FEATURES = 256


def layer_parts(model, layer_paths):
    """[(part path, layer path, "attention" | "mlp")] for each layer's attention and MLP child."""
    names = dict(model.named_modules())
    parts = []
    for layer in layer_paths:
        module = names.get(layer)
        if module is None:
            continue
        for name, _child in module.named_children():
            kind = "attention" if ATTENTION.match(name) else "mlp" if MLP.match(name) else None
            if kind:
                parts.append((f"{layer}.{name}" if layer else name, layer, kind))
    return parts


def order_parts(trace, parts, prefix=""):
    """Tag part events with their layer and kind, and move them after every layer's own event.

    A module's hook fires when it finishes, so a layer's parts arrive before the layer. Views that look
    for a layer's first event must find the layer's output, not its attention.
    """
    info = {f"{prefix}{path}": (f"{prefix}{layer}", kind) for path, layer, kind in parts}
    own, extra = [], []
    for event in trace.get("events", []):
        found = info.get(event.get("module"))
        if found is None:
            own.append(event)
        else:
            event["part_of"], event["part"] = found
            extra.append(event)
    trace["events"] = own + extra
    return trace
