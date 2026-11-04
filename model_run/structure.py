"""Inspect an already-instantiated PyTorch model; never load arbitrary model files."""


def inspect_module_tree(model, *, max_modules=10000):
    """Return containment, not an execution graph. Shared modules appear once per path."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for module inspection") from exc
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Expected an instantiated torch.nn.Module")
    if not isinstance(max_modules, int) or isinstance(max_modules, bool) or max_modules < 1:
        raise ValueError("max_modules must be a positive integer")
    nodes, pending = [], [("", model, frozenset())]
    truncated = False
    while pending:
        if len(nodes) >= max_modules:
            truncated = True
            break
        name, module, ancestors = pending.pop()
        if id(module) in ancestors:
            nodes.append({"path": name, "type": type(module).__name__, "cycle": True})
            continue
        parameters = [{"name": key, "shape": list(value.shape), "dtype": str(value.dtype)}
                      for key, value in module.named_parameters(recurse=False)]
        buffers = [{"name": key, "shape": list(value.shape), "dtype": str(value.dtype)}
                   for key, value in module.named_buffers(recurse=False)]
        nodes.append({"path": name, "type": f"{type(module).__module__}.{type(module).__qualname__}",
                      "parameters": parameters, "buffers": buffers})
        next_ancestors = ancestors | {id(module)}
        for key, child in reversed(list(module._modules.items())):
            if child is not None:
                pending.append((f"{name}.{key}" if name else key, child, next_ancestors))
    return {"schema_version": "1.0", "kind": "module_containment", "nodes": nodes,
            "truncated": truncated, "provenance": "instantiated PyTorch module hierarchy",
            "limitations": ["Containment is not execution order or a computation graph.",
                            "Model construction is the caller's responsibility and may execute code."]}
