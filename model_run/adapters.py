"""Explicit runtime bindings for already-loaded models, independent of rendering and hosts."""
from dataclasses import asdict, dataclass
from threading import Lock

from .capture import capture_forward
from .structure import inspect_module_tree
from .control import ObservationCancelled


@dataclass(frozen=True)
class Capability:
    status: str
    reason: str


class UnsupportedCapability(ValueError):
    pass


def _negotiate(report, requested):
    if not isinstance(requested, (list, tuple)) or any(not isinstance(item, str) for item in requested):
        raise ValueError("Requested capabilities must be a list of names")
    denied = {name: report["features"].get(name, {"status": "unsupported", "reason": "Unknown capability"})
              for name in requested if report["features"].get(name, {}).get("status") != "available"}
    return {"adapter_id": report["adapter_id"], "accepted": not denied, "denied": denied}


class TorchModuleAdapter:
    adapter_id = "pytorch.module"

    def __init__(self, model):
        import torch
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Binding requires an already-loaded torch.nn.Module")
        self.model = model
        self._capture_lock = Lock()

    def capabilities(self):
        import torch
        compiled = isinstance(self.model, torch.jit.ScriptModule) or hasattr(self.model, "_orig_mod")
        training = any(module.training for module in self.model.modules())
        outputs = Capability("unverified", "Compiled/scripted module hooks require a separate verified adapter") if compiled else Capability(
            "blocked" if training else "available",
            "Put all modules into evaluation mode before observation" if training else "Bounded module hooks; the caller supplies compatible inputs")
        features = {"module_tree": Capability("available", "Containment from the loaded module, not execution order"),
                    "module_outputs": outputs,
                    "next_token": Capability("unsupported", "Requires an explicit causal language-model binding"),
                    "attention": Capability("unsupported", "Generic hooks do not identify attention semantics")}
        return {"schema_version": "1.0", "adapter_id": self.adapter_id,
                "model_class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
                "features": {name: asdict(feature) for name, feature in features.items()},
                "verification": "Mechanism readiness only; no checkpoint or workload certification is inferred",
                "limitations": ["The host owns model loading, device placement and memory.",
                                "External inference on the same instance must be excluded by the host."]}

    def negotiate(self, requested):
        return _negotiate(self.capabilities(), requested)

    def structure(self, *, max_modules=10000):
        result = inspect_module_tree(self.model, max_modules=max_modules)
        result["adapter_id"] = self.adapter_id
        return result

    def _require(self, feature):
        result = self.negotiate([feature])
        if not result["accepted"]:
            raise UnsupportedCapability(result["denied"][feature]["reason"])

    def _run(self, function):
        if not self._capture_lock.acquire(blocking=False):
            raise RuntimeError("An observation is already running on this adapter")
        try:
            return function()
        except Exception as exc:
            partial = getattr(exc, "trace", None)
            if isinstance(partial, dict):
                partial["adapter_id"] = self.adapter_id
            raise
        finally:
            self._capture_lock.release()

    def capture(self, inputs, module_paths, **limits):
        self._require("module_outputs")
        output, trace = self._run(lambda: capture_forward(self.model, inputs, module_paths, **limits))
        trace["adapter_id"] = self.adapter_id
        return output, trace


class CausalLMAdapter(TorchModuleAdapter):
    adapter_id = "transformers.causal_lm"

    def __init__(self, model):
        from transformers import PreTrainedModel
        if not isinstance(model, PreTrainedModel) or not model.can_generate() or model.get_output_embeddings() is None:
            raise TypeError("Binding requires a Transformers model with generation and output embeddings")
        if getattr(model.config, "is_encoder_decoder", False):
            raise TypeError("Encoder-decoder generation requires a separate task adapter")
        super().__init__(model)

    def capabilities(self):
        result = super().capabilities()
        result["features"]["next_token"] = dict(result["features"]["module_outputs"])
        result["features"]["attention"] = {"status": "probe_required", "reason": "Returned attention depends on architecture and attention implementation; explicitly probe a selected layer/head"}
        return result

    def next_token(self, input_ids, module_paths, **limits):
        from .transformers_capture import capture_next_token
        self._require("next_token")
        output, trace = self._run(lambda: capture_next_token(self.model, input_ids, module_paths, **limits))
        trace["adapter_id"] = self.adapter_id
        return output, trace

    def probe_attention(self, input_ids, layer, head, **limits):
        from .transformers_capture import capture_attention
        self._require("module_outputs")
        trace = self._run(lambda: capture_attention(self.model, input_ids, layer, head, **limits))
        trace["adapter_id"] = self.adapter_id
        return trace

    def ablate_heads(self, input_ids, layer, heads=None, **limits):
        from .transformers_capture import capture_head_ablation
        self._require("module_outputs")
        trace = self._run(lambda: capture_head_ablation(self.model, input_ids, layer, heads, **limits))
        trace["adapter_id"] = self.adapter_id
        return trace


class AdapterRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, adapter_id, factory):
        if not isinstance(adapter_id, str) or not adapter_id or not callable(factory):
            raise ValueError("Adapter registration requires a name and a callable factory")
        if adapter_id in self._factories:
            raise ValueError(f"Adapter already registered: {adapter_id}")
        self._factories[adapter_id] = factory

    def bind(self, adapter_id, model):
        if adapter_id not in self._factories:
            raise UnsupportedCapability(f"Unknown runtime adapter: {adapter_id}")
        return self._factories[adapter_id](model)

    def available(self):
        return sorted(self._factories)


def default_registry():
    registry = AdapterRegistry()
    registry.register(TorchModuleAdapter.adapter_id, TorchModuleAdapter)
    registry.register(CausalLMAdapter.adapter_id, CausalLMAdapter)
    registry.register(DiffusersPipelineAdapter.adapter_id, DiffusersPipelineAdapter)
    return registry


class DiffusersPipelineAdapter:
    adapter_id = "diffusers.pipeline"

    def __init__(self, pipeline):
        from diffusers import DiffusionPipeline
        if not isinstance(pipeline, DiffusionPipeline):
            raise TypeError("Binding requires an already-loaded Diffusers pipeline")
        self.pipeline = pipeline
        self._capture_lock = Lock()

    def capabilities(self):
        import inspect
        try:
            parameters = inspect.signature(type(self.pipeline).__call__).parameters
        except (TypeError, ValueError):
            parameters = {}
        supported = all(name in parameters for name in ("callback_on_step_end", "callback_on_step_end_tensor_inputs")) and "latents" in (getattr(self.pipeline, "_callback_tensor_inputs", None) or [])
        return {"schema_version": "1.0", "adapter_id": self.adapter_id,
                "features": {"denoising_steps": {"status": "available" if supported else "unsupported",
                "reason": "Pipeline declares latent step callbacks" if supported else "Pipeline does not declare the required latent callback interface"}},
                "verification": "Declared callback readiness; checkpoint and workload require a runtime probe"}

    def capture_steps(self, inputs, *, every_n_steps=1, max_events=32, max_tensor_elements=1024, control=None,
                      cross_attention=None, attention_grid=32, token_labels=None):
        from .diffusion_capture import DiffusionCapture, CrossAttentionProbe
        if self.capabilities()["features"]["denoising_steps"]["status"] != "available":
            raise UnsupportedCapability("Pipeline does not expose the required latent callback interface")
        if not isinstance(inputs, dict) or any(key in inputs for key in ("callback_on_step_end", "callback_on_step_end_tensor_inputs")):
            raise ValueError("Pipeline inputs must be a dictionary without callback overrides")
        steps = inputs.get("num_inference_steps")
        if type(steps) is not int or not 1 <= steps <= 128:
            raise ValueError("Provide num_inference_steps between 1 and 128")
        recorder = DiffusionCapture(every_n_steps=every_n_steps, max_events=max_events, max_tensor_elements=max_tensor_elements, control=control)
        probe = CrossAttentionProbe(self.pipeline, cross_attention, max_grid=attention_grid, token_labels=token_labels) if cross_attention else None
        if not self._capture_lock.acquire(blocking=False):
            raise RuntimeError("An observation is already running on this adapter")
        try:
            if control is not None:
                control.checkpoint("pipeline_start")
            if probe is not None:
                probe.install()
            try:
                output = self.pipeline(**inputs, callback_on_step_end=recorder, callback_on_step_end_tensor_inputs=["latents"])
            finally:
                if probe is not None:
                    probe.remove()
            if control is not None:
                control.checkpoint("pipeline_complete")
            trace = recorder.trace(self.pipeline)
            trace["adapter_id"] = self.adapter_id
            if probe is not None:
                trace["cross_attention"] = probe.section()
            return output, trace
        except ObservationCancelled as exc:
            try:
                exc.trace = recorder.trace(self.pipeline, status="cancelled")
                exc.trace["adapter_id"] = self.adapter_id
            except ValueError:
                pass
            raise
        except Exception as exc:
            try:
                exc.trace = recorder.trace(self.pipeline, status="failed")
                exc.trace["adapter_id"] = self.adapter_id
            except ValueError:
                pass
            raise
        finally:
            self._capture_lock.release()

    def negotiate(self, requested):
        return _negotiate(self.capabilities(), requested)
