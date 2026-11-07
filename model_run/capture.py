"""Bounded observation of explicitly selected modules in an instantiated PyTorch model."""
from __future__ import annotations

import time
import math
from .control import ObservationCancelled
from .tensor_slices import SliceRecorder


def capture_forward(model, inputs, module_paths, *, max_tensor_elements=4096, max_events=64, output_selectors=None, control=None, slices=None, latent_shape=None, require_eval=True,
                    calls_per_module=None):
    """Run once and copy selected module output tensors to CPU, without mutating the model.

    Model construction, device placement, tokenization and trust remain the caller's responsibility.
    This is a module-output observer, not an operation-level execution trace.
    """
    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instantiated torch.nn.Module")
    if require_eval and any(module.training for module in model.modules()):
        raise ValueError("all model modules must already be in evaluation mode")
    if not isinstance(inputs, dict):
        raise TypeError("inputs must be a dictionary of keyword arguments")
    if not isinstance(module_paths, (list, tuple)) or not module_paths or any(not isinstance(p, str) for p in module_paths):
        raise ValueError("module_paths must be a nonempty list of names")
    if len(set(module_paths)) != len(module_paths):
        raise ValueError("duplicate module paths are not allowed")
    selectors = {} if output_selectors is None else output_selectors
    if not isinstance(selectors, dict) or any(key not in module_paths for key in selectors):
        raise ValueError("output_selectors must map selected modules to tensor paths")
    for path in selectors.values():
        if not isinstance(path, (list, tuple)) or any(not isinstance(part, str) and (type(part) is not int or part < 0) for part in path):
            raise ValueError("output selector paths require string keys or nonnegative indices")
    if type(max_tensor_elements) is not int or not 1 <= max_tensor_elements <= 65536:
        raise ValueError("max_tensor_elements must be between 1 and 65536")
    if type(max_events) is not int or not 1 <= max_events <= 1024:
        raise ValueError("max_events must be between 1 and 1024")
    if calls_per_module is not None and (type(calls_per_module) is not int or calls_per_module < 1):
        raise ValueError("calls_per_module must be a positive integer")
    calls = {}
    modules = dict(model.named_modules())
    missing = [name for name in module_paths if name not in modules]
    if missing:
        raise ValueError(f"Unknown module paths: {missing}")
    events, handles = [], []
    dropped = 0
    slices = SliceRecorder() if slices is None else slices

    def first_tensor(value, path=()):
        if isinstance(value, torch.Tensor):
            return value, list(path)
        if isinstance(value, (tuple, list)):
            return next((t for t in (first_tensor(v, (*path, index)) for index, v in enumerate(value)) if t is not None), None)
        if isinstance(value, dict):
            return next((t for t in (first_tensor(v, (*path, key)) for key, v in value.items()) if t is not None), None)
        return None

    def hook_for(path):
        def hook(_module, _args, output):
            nonlocal dropped
            if control is not None:
                control.check()
            if len(events) >= max_events:
                dropped += 1
                return
            # A module's later calls (a language model's generated tokens) can be left out.
            calls[path] = calls.get(path, 0) + 1
            if calls_per_module is not None and calls[path] > calls_per_module:
                return
            if path in selectors:
                value = output
                try:
                    for part in selectors[path]:
                        if isinstance(value, dict) or isinstance(value, (list, tuple)) and type(part) is int:
                            value = value[part]
                        else:
                            raise ValueError("Selector does not address an output container")
                except (KeyError, IndexError, TypeError) as exc:
                    raise ValueError(f"Unavailable output selector for {path}") from exc
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"Output selector does not identify a tensor for {path}")
                selected = (value, list(selectors[path]))
            else:
                selected = first_tensor(output)
                if isinstance(output, (tuple, list)) and latent_shape is not None:
                    # Some blocks return text before image; the image stream is the one worth reading.
                    for index, item in enumerate(output):
                        if isinstance(item, torch.Tensor) and slices.shows_image(item, latent_shape):
                            selected = (item, [index])
                            break
            if selected is None:
                events.append({"module": path, "status": "no_tensor_output"})
                return
            tensor, tensor_path = selected
            if tensor.is_complex() or tensor.layout != torch.strided or tensor.is_quantized:
                events.append({"module": path, "status": "unsupported_tensor", "dtype": str(tensor.dtype),
                               "layout": str(tensor.layout), "shape": list(tensor.shape), "output_path": tensor_path})
                return
            flattened = tensor.detach().reshape(-1)
            count = min(flattened.numel(), max_tensor_elements)
            sample = flattened[:count].to(device="cpu", dtype=torch.float32).tolist()
            nonfinite = sum(not math.isfinite(value) for value in sample)
            sample = [value if math.isfinite(value) else None for value in sample]
            events.append({"module": path, "status": "captured", "shape": list(tensor.shape),
                           "dtype": str(tensor.dtype), "device": str(tensor.device),
                           "selection": "flattened_prefix", "values": sample, "output_path": tensor_path,
                           "total_elements": flattened.numel(), "truncated": count < flattened.numel(),
                           "nonfinite_values_replaced_with_null": nonfinite})
            record = slices.record(tensor, latent_shape, path=path)
            if record is not None:
                events[-1]["slices"] = record
            if control is not None:
                control.checkpoint(f"module:{path}", completed=len(events), total=None)
        return hook

    started = time.perf_counter()
    def make_trace(status):
        return {"schema_version": "1.0", "kind": "pytorch_module_capture", "status": status,
                "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
                "torch_version": str(torch.__version__), "capture": {"modules": list(module_paths),
                "output_selectors": {key: list(value) for key, value in selectors.items()},
                "max_tensor_elements": max_tensor_elements, "max_events": max_events},
                "elapsed_host_seconds": time.perf_counter() - started, "dropped_events": dropped,
                "events": events, "limitations": ["Module outputs only; not an operation-level graph.",
                "Flattened prefixes converted to float32; values are not a representative sample.",
                "Non-finite samples are represented as null. Non-contiguous tensors may require a full reshape copy.",
                "Host elapsed time includes capture overhead and may not synchronize accelerator work.",
                "Cancellation is cooperative at selected module boundaries; kernels are not interrupted."]}

    try:
        if control is not None:
            control.checkpoint("forward_start")
        for name in module_paths:
            handles.append(modules[name].register_forward_hook(hook_for(name)))
            if control is not None:
                handles.append(modules[name].register_forward_pre_hook(lambda *_: control.check()))
        with torch.inference_mode():
            output = model(**inputs)
        if control is not None:
            control.checkpoint("forward_complete")
    except ObservationCancelled as exc:
        exc.trace = make_trace("cancelled")
        raise
    except Exception as exc:
        exc.trace = make_trace("failed")
        raise
    finally:
        for handle in handles:
            handle.remove()
    manifest = make_trace("completed")
    return output, manifest
