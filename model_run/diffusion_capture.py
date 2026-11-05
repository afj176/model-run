"""Bounded callback observation of supported Diffusers denoising latents."""
import math
from .tensor_slices import SliceRecorder


class DiffusionCapture:
    """A callback recorder for one pipeline invocation; does not alter pipeline state.

    The host must verify that the pipeline accepts callback_on_step_end and exposes
    ``latents`` in its callback tensor inputs. Never share a recorder across runs.
    """
    def __init__(self, *, every_n_steps=1, max_events=32, max_tensor_elements=1024, control=None):
        if type(every_n_steps) is not int or not 1 <= every_n_steps <= 10000:
            raise ValueError("every_n_steps must be between 1 and 10000")
        if type(max_events) is not int or not 1 <= max_events <= 256:
            raise ValueError("max_events must be between 1 and 256")
        if type(max_tensor_elements) is not int or not 1 <= max_tensor_elements <= 65536:
            raise ValueError("max_tensor_elements must be between 1 and 65536")
        self.every_n_steps = every_n_steps
        self.max_events = max_events
        self.max_tensor_elements = max_tensor_elements
        self._slices = SliceRecorder()
        self.events = []
        self.dropped_events = 0
        self._pipeline_identity = None
        self._last_step = -1
        self.control = control

    def __call__(self, pipeline, step, timestep, callback_kwargs):
        import torch
        if self._pipeline_identity is None:
            self._pipeline_identity = id(pipeline)
        elif self._pipeline_identity != id(pipeline):
            raise ValueError("Recorder cannot be shared across pipelines")
        if self.control is not None:
            self.control.checkpoint("denoising_step", completed=step + 1)
        if type(step) is not int or step < 0 or not isinstance(callback_kwargs, dict):
            raise ValueError("Invalid diffusion callback event")
        if step <= self._last_step:
            raise ValueError("Diffusion step order must increase; use a new recorder for each run")
        self._last_step = step
        if step % self.every_n_steps:
            return callback_kwargs
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return callback_kwargs
        latent = callback_kwargs.get("latents")
        if not isinstance(latent, torch.Tensor):
            raise ValueError("Pipeline did not provide a latents tensor")
        if latent.layout != torch.strided or latent.is_complex() or latent.is_quantized:
            raise ValueError("Unsupported latent tensor representation")
        t = timestep.item() if isinstance(timestep, torch.Tensor) and timestep.numel() == 1 else timestep
        if type(t) not in (int, float) or not math.isfinite(t):
            raise ValueError("Timestep must be a finite scalar")
        flat = latent.detach().reshape(-1)
        sample = flat[:self.max_tensor_elements].to(device="cpu", dtype=torch.float32).tolist()
        nonfinite = sum(not math.isfinite(value) for value in sample)
        sample = [value if math.isfinite(value) else None for value in sample]
        self.events.append({"step": step, "timestep": t, "shape": list(latent.shape),
                            "dtype": str(latent.dtype), "device": str(latent.device),
                            "selection": "flattened_prefix", "values": sample,
                            "total_elements": flat.numel(), "truncated": len(sample) < flat.numel(),
                            "nonfinite_values_replaced_with_null": nonfinite})
        record = self._slices.record(latent, list(latent.shape))
        if record is not None:
            self.events[-1]["slices"] = record
        return callback_kwargs

    def trace(self, pipeline, *, status="completed"):
        if self._pipeline_identity != id(pipeline):
            raise ValueError("No events recorded for this pipeline")
        import torch
        return {"schema_version": "1.0", "kind": "diffusion_step_capture", "status": status,
                "model_class": f"{type(pipeline).__module__}.{type(pipeline).__qualname__}",
                "torch_version": str(torch.__version__), "capture": {"every_n_steps": self.every_n_steps,
                "max_events": self.max_events, "max_tensor_elements": self.max_tensor_elements},
                "events": list(self.events), "dropped_events": self.dropped_events,
                "limitations": ["Latents are not decoded images or audio.",
                                "Flattened prefixes converted to float32 are not representative samples.",
                                "Non-contiguous tensors may require a full reshape copy.",
                                "Pipeline must support callback_on_step_end with latents."]}


class CrossAttentionProbe:
    """Measure image-to-text attention of one classic cross-attention module in a Diffusers UNet.

    Forward hooks on the module's ``to_q`` and ``to_k`` capture the exact projected tensors;
    the module's own forward hook turns them into head-averaged probabilities with the
    module's scale. Rotary or joint-attention transformers are refused, since their kernel
    inputs are not module outputs.
    """
    MAX_CALLS = 4

    def __init__(self, pipeline, module_path, *, max_grid=32, token_labels=None, ablated_tokens=None):
        import torch
        from .cross_attention import new_section, parse_token_list
        self.ablated_tokens = parse_token_list(ablated_tokens)
        denoiser = getattr(pipeline, "unet", None)
        if not isinstance(denoiser, torch.nn.Module):
            raise ValueError("Cross-attention maps need a pipeline with a UNet denoiser")
        if not isinstance(module_path, str) or not 1 <= len(module_path) <= 512:
            raise ValueError("Select one cross-attention module path")
        module = dict(denoiser.named_modules()).get(module_path)
        if module is None:
            raise ValueError("Unknown cross-attention module path")
        to_q, to_k = getattr(module, "to_q", None), getattr(module, "to_k", None)
        heads, scale = getattr(module, "heads", None), getattr(module, "scale", None)
        if not isinstance(to_q, torch.nn.Module) or not isinstance(to_k, torch.nn.Module) or type(heads) is not int or type(scale) not in (int, float):
            raise ValueError("Module is not a Diffusers attention block with to_q/to_k projections")
        if getattr(module, "is_cross_attention", None) is False:
            raise ValueError("Select a cross-attention (attn2) module; self-attention has no text keys")
        self.denoiser, self.module, self.to_q, self.to_k = denoiser, module, to_q, to_k
        self.heads, self.scale, self.path = heads, float(scale), module_path
        self.section_data = new_section(module_path, "diffusers_module_hooks", token_labels=token_labels, max_grid=max_grid)
        self.section_data["max_calls"] = self.MAX_CALLS
        self.section_data["limitations"].append("Diffusers classifier-free guidance batches the negative prompt first; the positive element is recorded when the batch has two entries.")
        self.max_grid = max_grid
        self.handles, self.call, self.latent_shape, self.stash, self.skipped = [], -1, None, {}, []

    def install(self):
        import torch
        def on_denoiser(_module, args, kwargs):
            sample = args[0] if args else kwargs.get("sample")
            self.call += 1
            self.latent_shape = list(sample.shape) if isinstance(sample, torch.Tensor) else None
            self.stash = {}
        def keep(name):
            def hook(_module, _args, output):
                if self.call < self.MAX_CALLS and isinstance(output, torch.Tensor):
                    self.stash[name] = output.detach()
            return hook
        def on_attention(_module, _args, _output):
            if self.call >= self.MAX_CALLS or self.call in {e["denoiser_call"] for e in self.section_data["events"]}:
                return
            q, k = self.stash.get("q"), self.stash.get("k")
            self.stash = {}
            if q is None or k is None or self.latent_shape is None:
                self._skip("Projections were not captured for this call")
                return
            from .cross_attention import measure, block_record, key_mask
            batch = int(q.shape[0])
            index, label = (1, "positive prompt (second of the guidance pair)") if batch == 2 else (0, "batch 0")
            try:
                mask = key_mask(self.ablated_tokens, int(k.shape[1]), device=k.device, dtype=torch.float32) if self.ablated_tokens else None
                result = measure(q, k, heads=self.heads, text_tokens=int(k.shape[1]), latent_shape=self.latent_shape,
                                 scale=self.scale, mask=mask, batch_index=index, max_grid=self.max_grid)
            except Exception as exc:
                self._skip(f"{type(exc).__name__}: {exc}")
                return
            if "unavailable" in result:
                self._skip(result["unavailable"])
            else:
                self.section_data["events"].append(block_record(self.path, denoiser_call=self.call, batch_label=label, measurement=result, block_label=self.path))
        self.handles = [self.denoiser.register_forward_pre_hook(on_denoiser, with_kwargs=True),
                        self.to_q.register_forward_hook(keep("q")), self.to_k.register_forward_hook(keep("k")),
                        self.module.register_forward_hook(on_attention)]

    def _skip(self, reason):
        if len(self.skipped) < 32:
            self.skipped.append({"denoiser_call": self.call, "reason": str(reason)[:200]})

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def section(self):
        data = dict(self.section_data)
        if self.skipped:
            data["skipped"] = list(self.skipped)
        return data


class TokenAblation:
    """Mask chosen text tokens out of every Diffusers cross-attention module for one run."""

    def __init__(self, pipeline, tokens, *, token_labels=None):
        import torch
        from .cross_attention import parse_token_list, ABLATION_LIMITATIONS
        denoiser = getattr(pipeline, "unet", None)
        if not isinstance(denoiser, torch.nn.Module):
            raise ValueError("Token ablation needs a pipeline with a UNet denoiser")
        self.tokens = parse_token_list(tokens)
        if not self.tokens:
            raise ValueError("Select at least one token to ablate")
        self.modules = [m for m in denoiser.modules() if getattr(m, "is_cross_attention", False) and hasattr(m, "set_processor") and hasattr(m, "processor")]
        if not self.modules:
            raise ValueError("The denoiser has no cross-attention modules to mask")
        labels = token_labels or []
        self.block = {"method": "cross_attention_key_mask", "tokens": list(self.tokens),
                      "labels": [labels[t] if t < len(labels) else str(t) for t in self.tokens],
                      "attention_modules": len(self.modules), "applied_attention_calls": 0, "limitations": list(ABLATION_LIMITATIONS)}
        self._originals = []

    def install(self):
        import torch
        ablation = self
        for module in self.modules:
            inner = module.processor
            self._originals.append((module, inner))

            class Masked:
                def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
                    if encoder_hidden_states is not None:
                        from .cross_attention import key_mask, MASK_BIAS
                        keys = int(encoder_hidden_states.shape[1])
                        bias = key_mask(ablation.tokens, keys, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)[0, 0]  # [1, keys]
                        if attention_mask is None:
                            attention_mask = bias.expand(encoder_hidden_states.shape[0], 1, keys)
                        else:
                            # Keep the pipeline's own mask shape ([B, keys] or [B, 1, keys]); add the bias along the key axis.
                            current = attention_mask
                            if current.dtype == torch.bool:
                                current = torch.zeros(current.shape, device=current.device, dtype=bias.dtype).masked_fill(~current, MASK_BIAS)
                            attention_mask = current.to(bias.dtype) + bias.reshape(*([1] * (current.ndim - 1)), keys)
                        ablation.block["applied_attention_calls"] += 1
                    return inner(attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs)
            module.set_processor(Masked())

    def remove(self):
        for module, inner in self._originals:
            module.set_processor(inner)
        self._originals = []

    def section(self):
        return dict(self.block)
