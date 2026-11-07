"""The parts inside a text layer worth recording beside it: its attention and its MLP.

A transformer layer adds what its attention read from other tokens and what its MLP computed for each
token. Recording those two outputs shows how much each part changed every token, not only the result.
"""
import re

# Text layers name them self_attn and mlp; image transformer blocks also use img_attn, img_mlp and ff.
ATTENTION = re.compile(r"^(self_attn|self_attention|attn|attention|img_attn)$")
MLP = re.compile(r"^(mlp|feed_forward|ffn|ff|img_mlp)$")
# Where an attention module's output leaves it, for blocks that call its parts rather than the module.
PROJECTIONS = ("proj", "o_proj", "out_proj", "o", "to_out")
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


def block_part_meter(model, blocks):
    """Hooks that measure, on the call they see first, the RMS of each block's output, of what the block
    changed (output minus input), and of its attention and MLP outputs. Returns (results(), remove());
    results() lists [{block, output_rms, update_rms?, parts}].

    A diffusion transformer's stream can carry a few huge channels, so a part is compared with the
    block's change, which those channels do not swamp, when the input can be matched."""
    import torch
    names = dict(model.named_modules())
    seen, handles = {}, []

    def stream(output):
        # A block that returns text and image streams: the image has the most tokens, whichever comes first.
        if isinstance(output, (tuple, list)):
            tensors = [item for item in output if isinstance(item, torch.Tensor)]
            return max(tensors, key=lambda item: item.numel()) if tensors else None
        return output

    def rms(output):
        tensor = stream(output)
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or tensor.numel() == 0:
            return None
        value = float(tensor.detach().float().pow(2).mean().sqrt())
        return value if value == value and value != float("inf") else None

    def keep(key):
        def hook(_module, _args, output):
            if key not in seen:
                seen[key] = rms(output)
        return hook

    inputs = {}

    def remember(block):
        def hook(_module, args, kwargs):
            if block not in inputs and (block, "output") not in seen:
                inputs[block] = [v for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor) and v.is_floating_point()]
        return hook

    def finish(block):
        def hook(_module, _args, output):
            if (block, "output") in seen:
                return
            seen[(block, "output")] = rms(output)
            after = stream(output)
            before = next((t for t in inputs.pop(block, []) if isinstance(after, torch.Tensor) and t.shape == after.shape), None)
            if before is not None:
                seen[(block, "update")] = rms(after.detach().float() - before.detach().float())
        return hook

    order = []
    for block in blocks:
        if block not in names or not block:
            continue
        parts = layer_parts(model, [block])
        if not parts:
            continue
        order.append((block, parts))
        handles.append(names[block].register_forward_pre_hook(remember(block), with_kwargs=True))
        handles.append(names[block].register_forward_hook(finish(block)))
        for path, _layer, kind in parts:
            handles.append(names[path].register_forward_hook(keep((path, "module"))))
            for name in PROJECTIONS:
                child = getattr(names[path], name, None)
                if isinstance(child, torch.nn.Module) and not isinstance(child, torch.nn.ModuleList):
                    handles.append(child.register_forward_hook(keep((path, "projection"))))
                    break

    def results():
        found = []
        for block, parts in order:
            output = seen.get((block, "output"))
            measured = []
            for path, _layer, kind in parts:
                value = seen.get((path, "module"), seen.get((path, "projection")))
                if value is not None:
                    measured.append({"kind": kind, "path": path, "rms": value})
            if output is not None and measured:
                # Attention first, then the MLP, in the order a block runs them.
                measured.sort(key=lambda part: part["kind"] != "attention")
                entry = {"block": block, "output_rms": output, "parts": measured[:4]}
                if seen.get((block, "update")) is not None:
                    entry["update_rms"] = seen[(block, "update")]
                found.append(entry)
        return found

    def remove():
        for handle in handles:
            handle.remove()
        handles.clear()
        inputs.clear()

    return results, remove
