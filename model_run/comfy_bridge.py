"""Opt-in native MODEL observation. Never loads, clones weights, or changes sampling."""
import re
import base64
import io
import logging
import os
import sys
from contextlib import contextmanager
from threading import Lock
import time

from .capture import capture_forward
from .comfy_paths import NOT_INSTALL_FOLDERS
from .denoising import DenoisingRecorder
from .structure import inspect_module_tree
from .tensor_slices import SliceRecorder
from .cross_attention import stable_audio_labels, collapse_padding, UNCHECKED_LABEL_LIMITATION, joint_by_geometry, REGISTER_LIMITATION, TEXT_MIXING_LIMITATION, place_registers, measure, block_record, new_section, key_mask, text_key_range, parse_token_list, align_labels, ABLATION_LIMITATIONS

MAX_ATTENTION_CALLS = 4
MAX_SECTION_VALUES = 524288  # aggregate cross-attention map values per trace
BLOCK_CONTAINER = re.compile(r"(?:^|\.)[a-z_]*(?:blocks?|layers?)$")
TEXT_LAST_MODELS = {"AceStepConditionGenerationModel", "AsymmDiTJoint", "ErnieImageModel", "Flux3", "JoyImageTransformer3DModel", "LensTransformer2DModel", "MOSSImage"}
# Measured the same way: keys run image, text, then other conditioning, so the text is neither first nor last.
IMAGE_FIRST_MODELS = {"HiDreamImageTransformer2DModel", "StageC"}
# These call ComfyUI's attention without transformer_options, so its override never reaches them.
BYPASS_ATTENTION_MODELS = {"AceStepConditionGenerationModel", "ErnieImageModel", "HunYuanDiT", "HunYuanDiTPlain", "PixArtMS"}
# Their latent runs along time, so a map is a strip, not an image.
AUDIO_MODELS = {"AceStepConditionGenerationModel", "ACEStepTransformer2DModel", "AudioDiffusionTransformer"}
TEXT_LEADS = {"ACEStepTransformer2DModel": 1}  # one speaker token comes first
# These make 3D shapes from a picture, so a map's "text" is image features and its "frames" are depth.
SHAPE_MODELS = {"HunYuanDiTPlain", "Hunyuan3Dv2", "LatentSeqMMFlowModel", "Trellis2"}
SHAPE_LIMITATION = ("This model makes a 3D shape from a picture. The tokens on this map are that picture's features, not words, "
                    "and a voxel model's frames are depth slices through the shape.")
LABELS_CHECKED = {"PixArtMS", "QwenImage21Transformer2DModel", "HiDreamO1Transformer", "LTXAVModel"}
LABELS_CHECKED_UNET_TEXT = {768, 2048}  # SD 1.5 and SDXL
MAX_LAYER_PATHS = 65
MAX_LAYER_EVENTS = 1000  # the trace store accepts 1,024 events
ALL_LAYER_PREFIX_VALUES = 65536
ALL_LAYER_SLICE_VALUES = 524288


def _layer_entry(path, module):
    """A hook on a sequential block may never fire, because ComfyUI's UNet runs its children one by one."""
    import torch
    while isinstance(module, torch.nn.Sequential) and len(module) > 0:
        path, module = f"{path}.{len(module) - 1}", module[-1]
    return path


def all_layer_paths(core):
    """Every top-level block of a denoiser, then its final output; just the output if it has no blocks."""
    import torch
    containers, paths = [], []
    for name, module in core.named_modules():
        if any(name.startswith(parent + ".") for parent in containers) or not BLOCK_CONTAINER.search(name):
            continue
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            containers.append(name)
            paths.extend(_layer_entry(f"{name}.{i}", child) for i, child in enumerate(module))
        elif isinstance(module, torch.nn.Sequential) and len(module) > 0:
            containers.append(name)
            paths.append(_layer_entry(name, module))
    if len(paths) > MAX_LAYER_PATHS - 1:
        paths = [paths[round(i * (len(paths) - 1) / (MAX_LAYER_PATHS - 2))] for i in range(MAX_LAYER_PATHS - 1)]
    return [*paths, ""]


def all_layer_plan(paths, max_calls, elements):
    """Fit a whole-model capture inside the trace bound, whatever the model's size."""
    calls = max(1, min(max_calls, MAX_LAYER_EVENTS // len(paths)))
    events = len(paths) * calls
    return {"max_calls": calls, "elements": max(16, min(elements, ALL_LAYER_PREFIX_VALUES // events)),
            "slice_values": ALL_LAYER_SLICE_VALUES, "slice_per_event": ALL_LAYER_SLICE_VALUES // events}


@contextmanager
def reach_attention(core, override):
    """Send a bypassing model's attention through ``override`` for one call, then put it back.

    Each attention module has a slot ComfyUI consults for its kernel, used by int8 weights. The
    kernel that was there still runs, so the output is unchanged.
    """
    from comfy.ldm.modules.attention import optimized_attention
    def routed(previous):
        def run(q, k, v, heads, *args, **kwargs):
            def kernel(_q, _k, _v, _heads, *inner, **options):
                return (previous or optimized_attention)(q, k, v, heads, *inner, **options)
            return override(kernel, q.peek(), k.peek(), v, heads, *args, **kwargs)
        return run
    slots = [(m, m.function) for m in core.modules() if type(m).__name__ == "ComfyAttention"]
    for module, previous in slots:
        module.function = routed(previous)
    try:
        yield
    finally:
        for module, previous in slots:
            module.function = previous


class DenoiserObserver:
    def __init__(self, patcher, module_paths, *, max_calls=8, elements=1024, publish=None, output_selectors=None,
                 attention_block=None, attention_grid=32, token_labels=None, ablate_tokens=None, slice_values=None, slice_per_event=None):
        from .application import bounded_int
        self.max_calls = bounded_int(max_calls, 1, 32, "max_calls")
        self.elements = bounded_int(elements, 1, 4096, "elements")
        if attention_block is not None and (not isinstance(attention_block, str) or not 1 <= len(attention_block) <= 64):
            raise ValueError("attention_block must name a block such as double:0 or output:1")
        self.attention_block = attention_block or None
        self.attention_grid = bounded_int(attention_grid, 4, 64, "attention_grid")
        if token_labels is not None and (not isinstance(token_labels, list) or len(token_labels) > 1024):
            raise ValueError("token_labels must be a bounded list")
        self.token_labels = token_labels
        self._labels_checked = False
        self._mixes_text = False
        self._registers = False
        core = patcher.model.diffusion_model
        lead = getattr(core, "register_tokens", None)
        # Measured with tools/check_comfy_families.py: where text sits in joint attention, when it is not first.
        self._text_lead = int(lead.shape[1]) if hasattr(lead, "shape") and len(lead.shape) == 3 else TEXT_LEADS.get(type(core).__name__, 0)
        self._text_start = -1 if type(core).__name__ in TEXT_LAST_MODELS else self._text_lead
        self._image_first = type(core).__name__ in IMAGE_FIRST_MODELS
        self._audio_only = type(core).__name__ in AUDIO_MODELS
        self._labels_dropped = False
        self.ablate_tokens = parse_token_list(ablate_tokens)
        self._ablation_applied = 0
        self.previous_attention = None
        self._current = None
        self._skipped = []
        self.core = patcher.model.diffusion_model
        self.base_model = patcher.model
        names = dict(self.core.named_modules())
        if not module_paths or len(module_paths) > MAX_LAYER_PATHS or any(p not in names for p in module_paths):
            raise ValueError(f"Select 1 to {MAX_LAYER_PATHS} valid diffusion module paths; an empty line selects the whole denoiser")
        self.paths = list(module_paths)
        if len(self.paths) * self.max_calls * self.elements > 262144:
            raise ValueError("Capture exceeds the aggregate budget of 262,144 values. Reduce modules, calls, or elements.")
        self.selectors = output_selectors or {}
        if not isinstance(self.selectors, dict) or any(key not in self.paths for key in self.selectors):
            raise ValueError("Output selectors must map selected module paths to lists of output indices or keys")
        self.previous = patcher.model_options.get("model_function_wrapper")
        if isinstance(self.previous, DenoiserObserver):
            raise ValueError("This MODEL already has a Model Run observer")
        self.publish = publish
        self._lock = Lock()
        self._slices = SliceRecorder(**({"max_values": slice_values} if slice_values else {}), per_event=slice_per_event)
        self.calls = 0
        self.trace = None
        self.tree = inspect_module_tree(self.core)

    def __deepcopy__(self, memo):
        # ComfyUI clones model_options when downstream nodes clone a MODEL.
        # Clones share the same loaded core and bounded observer, not copied weights.
        memo[id(self)] = self
        return self

    def __call__(self, function, arguments):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent use of the same observed MODEL is unsupported")
        try:
            if self.calls >= self.max_calls:
                if self.ablate_tokens:
                    # The intervention covers every denoiser call, not just the recorded ones.
                    self._current = self._attention_context(self.calls, arguments, measure_call=False)
                return self._invoke(function, arguments)
            import torch
            observer = self
            class Invocation(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.denoiser = observer.core
                    # Do not call eval(): native module state belongs to ComfyUI.
                    self.training = False
                def forward(self):
                    return observer._invoke(function, arguments)
            call = self.calls
            self.calls += 1
            self._current = self._attention_context(call, arguments)
            try:
                latent_shape = self._latent_shape(arguments)
                self._slices.text_tokens = self._text_token_count(arguments)
                self._slices.reference_tokens = self._reference_token_count(arguments)
                self._slices.conditional_batch = self._conditional_batch(arguments)
                self._slices.sequence = self._token_sequence(latent_shape, self._slices.text_tokens)
                output, trace = capture_forward(Invocation(), {}, ["denoiser" + ("." + p if p else "") for p in self.paths],
                    max_tensor_elements=self.elements, max_events=len(self.paths),
                    output_selectors={"denoiser" + ("." + p if p else ""): value for p, value in self.selectors.items()},
                    slices=self._slices, latent_shape=latent_shape)
            except Exception as exc:
                trace = getattr(exc, "trace", None)
                if trace:
                    self._publish(trace, call)
                raise
            self._publish(trace, call)
            return output
        finally:
            self._current = None
            self._lock.release()

    def _latent_shape(self, arguments):
        """Real shape of the latent this call denoises; audio+video models pack their streams."""
        packed = getattr(self.base_model, "latent_shapes", None)
        if isinstance(packed, (list, tuple)) and len(packed) > 1:
            return list(packed[0])
        shape = getattr(arguments.get("input"), "shape", None)
        return list(shape) if shape is not None else None

    def _labels_checked_family(self):
        name = type(self.core).__name__
        # Klein is the Flux variant checked; Flux 1 uses other text encoders.
        if name == "UNetModel":
            # The same class also runs other text encoders, such as SD 2's and Kolors', which were not checked.
            reader = next((module.attn2.to_k for module in self.core.modules() if hasattr(module, "attn2")), None)
            return getattr(reader, "in_features", None) in LABELS_CHECKED_UNET_TEXT
        return name in LABELS_CHECKED or (name == "Flux" and bool(getattr(getattr(self.core, "params", None), "global_modulation", False)))

    @staticmethod
    def _conditional_batch(arguments):
        """Which batch entry holds the positive prompt, when this call's batch says."""
        conditioning = arguments.get("c") if isinstance(arguments.get("c"), dict) else {}
        order = arguments.get("cond_or_uncond")
        if not isinstance(order, list):
            order = (conditioning.get("transformer_options") or {}).get("cond_or_uncond")
        return order.index(0) if isinstance(order, list) and 0 in order and len(order) > 1 else None

    def _reference_token_count(self, arguments):
        """Tokens an edit model's reference images add, from the references this call actually received."""
        conditioning = arguments.get("c") if isinstance(arguments.get("c"), dict) else {}
        references = conditioning.get("ref_latents")
        if not isinstance(references, (list, tuple)):
            return 0
        patch = getattr(self.core, "patch_size", None) or getattr(getattr(self.core, "params", None), "patch_size", None) or 1
        patch = int(patch[-1]) if isinstance(patch, (list, tuple)) else int(patch)
        total = 0
        for reference in references:
            shape = getattr(reference, "shape", None)
            if shape is not None and len(shape) >= 4:
                total += -(-int(shape[-2]) // patch) * -(-int(shape[-1]) // patch)
        return total

    @staticmethod
    def _text_token_count(arguments):
        conditioning = arguments.get("c") if isinstance(arguments.get("c"), dict) else {}
        context = conditioning.get("c_crossattn", conditioning.get("context"))
        shape = getattr(context, "shape", None)
        return int(shape[1]) if shape is not None and len(shape) == 3 else None

    def _token_sequence(self, latent_shape, text_tokens):
        """How to read a [B, T, D] stream that is not the image: audio along time, or text by token."""
        from .cross_attention import image_grid
        packed = getattr(self.base_model, "latent_shapes", None)
        steps = int(packed[1][2]) if isinstance(packed, (list, tuple)) and len(packed) > 1 and len(packed[1]) > 2 else None
        def plan(length):
            if length == steps:
                return (length, "time", "Tokens are assumed to follow the audio latent's time axis, because their count matches it.")
            # A count that also fits the image grid is ambiguous, so the image reading is kept.
            if length == text_tokens and image_grid(length, latent_shape) is None:
                return (length, "text token", "This stream is taken to be the text tokens, because its count matches the prompt's.")
            return None
        return plan

    def _attention_context(self, call, arguments, measure_call=True):
        """Geometry the attention override needs for this denoiser call, or None when not measuring."""
        measure_call = measure_call and self.attention_block is not None and call < MAX_ATTENTION_CALLS
        if not measure_call and not self.ablate_tokens:
            return None
        import torch
        conditioning = arguments.get("c") if isinstance(arguments.get("c"), dict) else {}
        context = conditioning.get("c_crossattn", conditioning.get("context"))
        latent = arguments.get("input")
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            self._skip(call, "Denoiser call has no text context tensor")
            return None
        if not isinstance(latent, torch.Tensor):
            self._skip(call, f"Denoiser input of type {type(latent).__name__} has no latent tensor this measurement understands")
            return None
        order = arguments.get("cond_or_uncond")
        if not isinstance(order, list):
            order = (conditioning.get("transformer_options") or {}).get("cond_or_uncond")
        batch_index, batch_label = 0, "batch 0"
        if isinstance(order, list) and len(order) == int(latent.shape[0]) and 0 in order:
            batch_index, batch_label = order.index(0), "conditional (cond_or_uncond)"
        # Audio+video models reach the wrapper with their streams packed into one tensor; the sampler
        # records the real shapes on the model, and maps cover the first (video) stream.
        packed = getattr(self.base_model, "latent_shapes", None)
        latent_shape = list(packed[0]) if isinstance(packed, (list, tuple)) and len(packed) > 1 else list(latent.shape)
        audio_steps = int(packed[1][2]) if isinstance(packed, (list, tuple)) and len(packed) > 1 and len(packed[1]) > 2 else None
        if self._audio_only:
            # ACE-Step 1.5 groups time steps, while the first ACE-Step's patch is a pair for the frequency axis.
            patch = getattr(getattr(self.core, "decoder", self.core), "patch_size", 1)
            audio_steps = int(latent_shape[-1]) // (patch if isinstance(patch, int) else 1)
        return {"call": call, "latent_shape": latent_shape, "audio_steps": audio_steps, "reference_tokens": self._reference_token_count(arguments), "text_tokens": int(context.shape[1]),
                "batch_index": batch_index, "batch_label": batch_label, "record": None, "reason": None,
                "measure": measure_call}

    def _skip(self, call, reason):
        if len(self._skipped) < 32:
            self._skipped.append({"denoiser_call": call, "reason": str(reason)[:200]})

    def _image_layout(self, latent_shape, text_tokens):
        """Tokens that are the image and queries before it, for models whose lengths cannot say: (known, lead)."""
        name = type(self.core).__name__
        if name == "HiDreamImageTransformer2DModel":
            patch = int(self.core.patch_size)
            return (int(latent_shape[-2]) // patch) * (int(latent_shape[-1]) // patch), 0
        if name == "MiniMaxH3Model":
            # Tokens run text, video, then audio, and only the video gets an image map.
            frames, height, width = (int(size) // int(patch) for size, patch in zip(latent_shape[-3:], self.core.patch_size))
            return frames * height * width, 0
        if name == "Fibo2":
            # Queries run summary tokens, the image, then any reference images.
            gist = sys.modules[type(self.core).__module__].gist_count(text_tokens, self.core.max_gist_tokens)
            return int(latent_shape[-2]) * int(latent_shape[-1]), gist
        return None, 0

    def attention_override(self, func, q, k, v, heads, *args, **kwargs):
        """ComfyUI optimized_attention_override: measure the selected block, then run the real kernel."""
        current = self._current
        if current is not None:
            current["override_seen"] = True
        if current is not None and self.ablate_tokens:
            try:
                options = kwargs.get("transformer_options") or {}
                key_count = int(k.shape[-2] if k.ndim == 4 else k.shape[1])
                # Joint text+image attention is only assumed where the model labels its blocks as DiT blocks;
                # an unlabelled call with equal query and key counts is self-attention and has no text keys.
                query_count = int(q.shape[-2] if q.ndim == 4 else q.shape[1])
                joint = "block_type" in options or "block" not in options and joint_by_geometry(query_count, key_count, current["text_tokens"], current["latent_shape"], self._text_lead)
                span = text_key_range(query_count, key_count, current["text_tokens"], allow_joint=joint, text_start=self._text_start)
                if self._image_first:
                    start = self._image_layout(current["latent_shape"], current["text_tokens"])[0] or query_count
                    span = (start, start + current["text_tokens"]) if key_count >= start + current["text_tokens"] else None
                elif self._audio_only and query_count != key_count and key_count >= self._text_lead + current["text_tokens"]:
                    start = key_count - current["text_tokens"] if self._text_start == -1 else self._text_lead
                    span = (start, start + current["text_tokens"])
                if span is not None:
                    existing = kwargs["mask"] if "mask" in kwargs else (args[0] if args else None)
                    bias = key_mask(self.ablate_tokens, int(k.shape[-2] if k.ndim == 4 else k.shape[1]), start=span[0], existing=existing, device=q.device, dtype=q.dtype)
                    if "mask" in kwargs or not args:
                        kwargs["mask"] = bias
                    else:
                        args = (bias, *args[1:])
                    self._ablation_applied += 1
            except Exception as exc:  # the intervention must fail loudly in the trace, never in sampling
                current["reason"] = f"ablation: {type(exc).__name__}: {exc}"
        if current is not None and current.get("measure", True) and current["record"] is None:
            try:
                options = kwargs.get("transformer_options") or {}
                if "block_type" in options:
                    label, allow_joint = f"{options.get('block_type')}:{options.get('block_index')}", True
                elif isinstance(options.get("block"), (tuple, list)) and len(options["block"]) == 2:
                    label, allow_joint = f"{options['block'][0]}:{options['block'][1]}", False
                else:
                    # Unlabelled attention: number the calls within this denoiser call so they can still be selected.
                    ordinal = current.get("ordinal", 0)
                    current["ordinal"] = ordinal + 1
                    label, allow_joint = f"call:{ordinal}", False
                query_count = int(q.shape[-2] if q.ndim == 4 else q.shape[1])
                key_count = int(k.shape[-2] if k.ndim == 4 else k.shape[1])
                is_audio = current.get("audio_steps") is not None and (query_count == current["audio_steps"] and key_count == current["text_tokens"]
                            or self._audio_only and query_count >= current["audio_steps"] and query_count != key_count and key_count >= self._text_lead + current["text_tokens"])
                audio_label = None
                if is_audio:
                    # Named by what it is, so the choice does not depend on how many other calls each block makes.
                    audio_label = f"audio:{current.get('audio_ordinal', 0)}"
                    current["audio_ordinal"] = current.get("audio_ordinal", 0) + 1
                    seen = current.setdefault("seen", [])
                    if len(seen) < 64:
                        seen.append(audio_label)
                if label is not None:
                    seen = current.setdefault("seen", [])
                    if label not in seen and len(seen) < 64:
                        seen.append(label)
                if self.attention_block in (label, audio_label):
                    mask = kwargs.get("mask", args[0] if args else None)
                    known, lead = self._image_layout(current["latent_shape"], current["text_tokens"])
                    if self._audio_only and is_audio:
                        known = current["audio_steps"]
                    elif type(self.core).__name__ == "HunYuanDiT" and key_count != query_count:
                        known = query_count
                    measurement = measure(q, k, heads=heads, text_tokens=current["text_tokens"], latent_shape=current["latent_shape"],
                                          mask=mask, batch_index=current["batch_index"], max_grid=self.attention_grid, stream="audio" if is_audio else None, text_start=self._text_start, reference_tokens=current.get("reference_tokens", 0),
                                          image_first=self._image_first, known_image_tokens=known, query_lead=lead)
                    if "unavailable" in measurement:
                        current["reason"] = measurement["unavailable"]
                    elif not (self._image_first or known) and (measurement["attention_kind"] == "joint" and not allow_joint and not label.startswith("call:") or measurement["attention_kind"] == "joint" and not allow_joint and not joint_by_geometry(
                            int(q.shape[-2] if q.ndim == 4 else q.shape[1]), int(k.shape[-2] if k.ndim == 4 else k.shape[1]), current["text_tokens"], current["latent_shape"], self._text_lead)):
                        current["reason"] = "Self-attention in this block has no text keys"
                    else:
                        self._mixes_text = self._mixes_text or bool(measurement.get("text_spreads"))
                        current["record"] = block_record(self.attention_block, denoiser_call=current["call"], batch_label=current["batch_label"],
                                                         measurement=measurement, block_label=audio_label or label)
            except Exception as exc:  # measurement must never break sampling
                current["reason"] = f"{type(exc).__name__}: {exc}"
        if self.previous_attention is not None:
            return self.previous_attention(func, q, k, v, heads, *args, **kwargs)
        return func(q, k, v, heads, *args, **kwargs)

    def _invoke(self, function, args):
        if type(self.core).__name__ in BYPASS_ATTENTION_MODELS:
            with reach_attention(self.core, self.attention_override):
                return self._invoke_reached(function, args)
        return self._invoke_reached(function, args)

    def _invoke_reached(self, function, args):
        if self.previous is not None:
            return self.previous(function, args)
        return function(args["input"], args["timestep"], **args["c"])

    def _publish(self, trace, call):
        for event in trace["events"]:
            event["module"] = event["module"][len("denoiser"):].lstrip(".")
            event["denoiser_call"] = call
        current = self._current
        if current is not None and self.token_labels is not None and not self._labels_checked:
            self._labels_checked = True
            connector = getattr(self.core, "video_embeddings_connector", None)
            self._registers = bool(getattr(connector, "num_learnable_registers", None))
            aligned = (place_registers if self._registers else align_labels)(self.token_labels, current["text_tokens"])
            if aligned is None and type(self.core).__name__ == "AudioDiffusionTransformer":
                aligned = stable_audio_labels(self.token_labels, current["text_tokens"])
            self._labels_dropped = aligned is None
            self.token_labels = aligned
        if self.trace is None:
            self.trace = trace
            self.trace["model_class"] = f"{type(self.core).__module__}.{type(self.core).__qualname__}"
            self.trace["structure"] = self.tree
            self.trace["capture"]["modules"] = self.paths
            self.trace["capture"]["output_selectors"] = self.selectors
            self.trace["capture"]["max_calls"] = self.max_calls
            self.trace["capture"]["max_events_per_call"] = len(self.paths)
            self.trace["capture"]["max_events"] = len(self.paths) * self.max_calls
            self.trace["source"] = "comfyui_native_denoiser"
            if self.attention_block is not None:
                self.trace["cross_attention"] = new_section(self.attention_block, "comfyui_attention_override",
                                                            token_labels=self.token_labels, max_grid=self.attention_grid)
                self.trace["cross_attention"]["max_calls"] = MAX_ATTENTION_CALLS
                self.trace["cross_attention"]["limitations"].append("Text token labels come from the node's optional clip and prompt inputs; without them tokens are numbered.")
                if self._labels_dropped:
                    self.trace["cross_attention"]["limitations"].append("Token labels were dropped because the tokenised prompt does not line up with this model's text keys; tokens are numbered instead.")
            if self.ablate_tokens:
                labels = self.token_labels or []
                self.trace["token_ablation"] = {"method": "cross_attention_key_mask", "tokens": list(self.ablate_tokens),
                                                "labels": [labels[t] if t < len(labels) else str(t) for t in self.ablate_tokens],
                                                "applied_attention_calls": 0, "limitations": list(ABLATION_LIMITATIONS)}
            self.trace["limitations"].extend(["Denoiser calls are not sampler steps; guidance may invoke the denoiser multiple times.",
                "Call budget applies to this node instance until it is re-executed. ComfyUI owns sampling and cancellation."])
        else:
            self.trace["events"].extend(trace["events"])
            self.trace["elapsed_host_seconds"] += trace["elapsed_host_seconds"]
            self.trace["status"] = trace["status"]
            self.trace["dropped_events"] += trace["dropped_events"]
        self.trace["captured_calls"] = call + 1
        if "token_ablation" in self.trace:
            self.trace["token_ablation"]["applied_attention_calls"] = self._ablation_applied
        current = self._current
        if current is not None and "cross_attention" in self.trace:
            section = self.trace["cross_attention"]
            record = current["record"]
            if record is not None:
                used = sum(len(e["maps"]) * len(e["maps"][0]) for e in section["events"] if e.get("maps"))
                if used + len(record["maps"]) * len(record["maps"][0]) > MAX_SECTION_VALUES:
                    self._skip(call, f"Cross-attention value budget of {MAX_SECTION_VALUES:,} reached; earlier calls are kept")
                else:
                    section["events"].append(record)
            else:
                seen = current.get("seen") or []
                if current["reason"]:
                    self._skip(call, current["reason"])
                elif not current.get("override_seen"):
                    self._skip(call, "This denoiser's attention calls bypass ComfyUI's attention override (no transformer_options reach the kernel), so attention maps are unavailable for this model family; module outputs are still captured")
                else:
                    hint = f"; blocks seen: {seen[0]} … {seen[-1]} ({len(seen)})" if seen else "; no labelled attention calls were seen"
                    self._skip(call, f"No attention call matched block {self.attention_block}{hint}")
            current["record"] = None
        if self.token_labels and "cross_attention" in self.trace and not self._labels_checked_family() and UNCHECKED_LABEL_LIMITATION not in self.trace["cross_attention"]["limitations"]:
            self.trace["cross_attention"]["limitations"].append(UNCHECKED_LABEL_LIMITATION)
        if self._registers and "cross_attention" in self.trace and REGISTER_LIMITATION not in self.trace["cross_attention"]["limitations"]:
            self.trace["cross_attention"]["limitations"].append(REGISTER_LIMITATION)
        if type(self.core).__name__ in SHAPE_MODELS and "cross_attention" in self.trace and SHAPE_LIMITATION not in self.trace["cross_attention"]["limitations"]:
            self.trace["cross_attention"]["limitations"].append(SHAPE_LIMITATION)
        if self._mixes_text and "cross_attention" in self.trace and TEXT_MIXING_LIMITATION not in self.trace["cross_attention"]["limitations"]:
            self.trace["cross_attention"]["limitations"].append(TEXT_MIXING_LIMITATION)
        if "cross_attention" in self.trace and self._skipped:
            self.trace["cross_attention"]["skipped"] = list(self._skipped)
        if self.publish is not None:
            self._publish_safely(call)

    def attach_denoising(self, section):
        """Sampler steps outlast the call budget, so they publish on their own."""
        if self.trace is None or self.publish is None:
            return
        self.trace["denoising"] = section
        self._publish_safely(self.calls - 1)

    def attach_images(self, images):
        """Pin decoded output to this run's trace; frames only when there is more than one image."""
        if self.trace is None or self.publish is None:
            return False
        preview, frames = encode_images(images)
        self.trace["preview"] = preview
        if frames:
            self.trace["preview_frames"] = frames
        for drop in ("", "preview_frames", "preview"):
            if drop:
                self.trace.pop(drop, None)
            try:
                self.publish(self.trace)
                return drop == ""
            except ValueError as exc:
                logging.getLogger(__name__).warning("Model Run dropped %s to fit the trace bound: %s", drop or "nothing", exc)
        return False

    def _publish_safely(self, call):
        """Publishing must never abort the user's sampling run; shrink and retry, then give up quietly."""
        try:
            self.publish(self.trace)
            return
        except ValueError as exc:
            section = self.trace.get("cross_attention")
            if section and section.get("events"):
                section["events"].pop()
                self._skip(call, f"Dropped this call's map so the trace fits: {str(exc)[:120]}")
                section["skipped"] = list(self._skipped)
                try:
                    self.publish(self.trace)
                    return
                except Exception as retry:  # noqa: BLE001 - reported below
                    exc = retry
            steps = (self.trace.get("denoising") or {}).get("steps") or []
            if any("images" in step for step in steps):
                for step in steps:
                    step.pop("images", None)
                self.trace["denoising"]["skipped"] = "The step images were dropped so that this trace could be kept; the numbers remain."
                try:
                    self.publish(self.trace)
                    return
                except Exception as retry:  # noqa: BLE001 - reported below
                    exc = retry
            dropped = 0
            for event in self.trace.get("events", []):
                dropped += event.pop("slices", None) is not None
            if dropped:
                # The layer values are worth more than losing the whole run.
                self.trace["limitations"].append("Frame and channel views were dropped from this trace so that it could be kept.")
                try:
                    self.publish(self.trace)
                    return
                except Exception as retry:  # noqa: BLE001 - reported below
                    exc = retry
            import logging
            logging.getLogger(__name__).warning("Model Run could not publish the observation trace: %s", exc)
        except Exception as exc:  # noqa: BLE001 - sampling continues regardless
            import logging
            logging.getLogger(__name__).warning("Model Run could not publish the observation trace: %s", exc)


_application = None


def application():
    global _application
    if _application is None:
        from .application import Application
        import folder_paths
        entries = [{"category": category, "path": path} for category, value in folder_paths.folder_names_and_paths.items()
                   if category != "custom_nodes" for path in value[0]]
        def install_folders():
            # Live, so a folder a custom node registers later can be installed into.
            return [{"category": category, "path": path} for category, value in folder_paths.folder_names_and_paths.items()
                    if category not in NOT_INSTALL_FOLDERS for path in value[0]]
        _application = Application(entries, host_kind="comfyui", install_folders=install_folders,
                                   comfy_dir=[folder_paths.base_path, os.path.dirname(os.path.abspath(folder_paths.__file__))])
        _application.scan()
    return _application


def encode_images(images, *, side=512, frame_side=256, max_frames=16):
    """PNG preview of the first image, plus evenly chosen smaller frames for a video batch."""
    import torch
    from PIL import Image
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] not in (1, 3, 4):
        raise ValueError("Expected an IMAGE batch shaped [frames, height, width, channels]")
    def png(tensor, limit):
        pixels = (tensor[..., :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        image = Image.fromarray(pixels.squeeze(-1) if pixels.shape[-1] == 1 else pixels)
        image.thumbnail((limit, limit))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    preview = {"mime": "image/png", "base64": png(images[0], side)}
    count = int(images.shape[0])
    if count == 1:
        return preview, None
    kept = count if count <= max_frames else max_frames
    indices = sorted({round(i * (count - 1) / (kept - 1)) for i in range(kept)}) if kept > 1 else [0]
    return preview, {"mime": "image/png", "frames": [png(images[i], frame_side) for i in indices],
                     "frame_indices": indices, "source_frames": count}


class ModelRunAttachOutput:
    CATEGORY = "Model Run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "attach"
    OUTPUT_NODE = True
    DESCRIPTION = "Pin decoded images or video frames to the Model Run trace of an observed MODEL. Connect after your VAE Decode."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"observed_model": ("MODEL", {"tooltip": "The MODEL output of Model Run · Observe."}),
                             "images": ("IMAGE",)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def attach(self, observed_model, images):
        wrapper = observed_model.model_options.get("model_function_wrapper")
        if isinstance(wrapper, DenoiserObserver):
            wrapper.attach_images(images)
        else:
            logging.getLogger(__name__).warning("Model Run · Attach Output needs the MODEL from Model Run · Observe; nothing was attached.")
        return (images,)


class ModelRunObserve:
    CATEGORY = "Model Run"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("observed_model",)
    FUNCTION = "observe"
    DESCRIPTION = "Observe bounded native denoiser outputs. Connect between your model loader and sampler. Open Model Run to inspect captures."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "modules": ("STRING", {"default": "", "multiline": True, "tooltip": "One module path per line. Leave empty to record every layer and the final output; on large models each event then keeps fewer values."}),
                "max_calls": ("INT", {"default": 8, "min": 1, "max": 32}),
                "elements": ("INT", {"default": 1024, "min": 1, "max": 4096})},
                "optional": {"output_selectors": ("STRING", {"default": "{}", "multiline": True,
                    "tooltip": "JSON mapping from module path to output tensor path, e.g. {\"transformer_blocks.0\": [1]} selects the audio tensor in an LTX block tuple."}),
                    "attention_block": ("STRING", {"default": "", "tooltip": "Measure attention to text in one block: audio:0 for the audio stream of an audio+video model, double:0 or single:3 for DiT models, input:4 / middle:0 / output:1 for UNets. Empty disables."}),
                    "attention_grid": ("INT", {"default": 32, "min": 4, "max": 64, "tooltip": "Longest side of the recorded attention grid."}),
                    "clip": ("CLIP", {"tooltip": "Optional: with prompt, labels text tokens in the attention maps."}),
                    "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "The positive prompt, only used to label tokens."}),
                    "denoising_views": ("BOOLEAN", {"default": True, "tooltip": "Record the latent, predicted noise and estimate at each sampler step."}),
                    "ablate_tokens": ("STRING", {"default": "", "tooltip": "Intervention: comma-separated text token indices whose keys are masked out of every attention call for this run. Compare the result with a run that leaves this empty."})},
                "hidden": {"workflow_graph": "PROMPT", "unique_id": "UNIQUE_ID"}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Fresh observation budget for every queued execution.

    def observe(self, model, modules, max_calls, elements, output_selectors="{}", attention_block="", attention_grid=32, clip=None, prompt="", ablate_tokens="", denoising_views=True, workflow_graph=None, unique_id=None):
        import json
        if not isinstance(output_selectors, str) or len(output_selectors) > 8192:
            raise ValueError("Output selector JSON exceeds 8 KiB")
        selectors = json.loads(output_selectors)
        block = attention_block.strip() if isinstance(attention_block, str) else ""
        ablated = parse_token_list(ablate_tokens if isinstance(ablate_tokens, str) else "")
        labels = token_labels(clip, prompt) if (block or ablated) and clip is not None and isinstance(prompt, str) and prompt.strip() else None
        clone = model.clone()
        app = application()
        last_id = [None]
        def publish(trace):
            # Replace this run's snapshot; other runs remain in bounded history.
            trace["model_name"] = type(model.model.diffusion_model).__name__
            if isinstance(workflow_graph, dict) and unique_id is not None:
                trace["native_rerun"] = True
            result = app.publish(trace, trace["model_name"], replace_id=last_id[0], note=prompt if isinstance(prompt, str) else None)
            last_id[0] = result["id"]
            if trace.get("native_rerun"):
                app.remember_workflow(result["id"], workflow_graph, unique_id)
        chosen = [s.strip() for s in modules.splitlines() if s.strip()]
        plan = {"max_calls": max_calls, "elements": elements}
        if not chosen:
            chosen = all_layer_paths(model.model.diffusion_model)
            plan = all_layer_plan(chosen, max_calls, elements)
        observer = DenoiserObserver(model, chosen, publish=publish, **plan,
                                     output_selectors=selectors,
                                    attention_block=block or None, attention_grid=attention_grid, token_labels=labels, ablate_tokens=ablated)
        clone.set_model_unet_function_wrapper(observer)
        if denoising_views:
            recorder = DenoisingRecorder(getattr(model.model, "latent_format", None), on_update=observer.attach_denoising)
            clone.set_model_sampler_post_cfg_function(recorder)
        if block or ablated:
            options = clone.model_options.setdefault("transformer_options", {})
            observer.previous_attention = options.get("optimized_attention_override")
            options["optimized_attention_override"] = observer.attention_override
        return (clone,)


def token_labels(clip, prompt):
    """Best-effort token strings from the CLIP's own tokenizer; None when unavailable."""
    try:
        tokenized = clip.tokenize(prompt[:8192])
        best = []
        for value in (tokenized.values() if isinstance(tokenized, dict) else [tokenized]):
            if not isinstance(value, list):
                continue  # tokenizers may attach flags such as keep_vision
            flat = [pair for chunk in value if isinstance(chunk, (list, tuple)) for pair in chunk]
            if len(flat) > len(best):
                best = flat
        if not best:
            return None
        tokenizer = getattr(clip, "tokenizer", None)
        if hasattr(tokenizer, "decode"):
            # Per-token decode turns byte-level BPE pieces into readable text.
            # T5 has a token that is only a space, which decodes to nothing.
            return collapse_padding([str(tokenizer.decode([p[0]], skip_special_tokens=False)) or "\u2581" for p in best][:1024])
        if hasattr(tokenizer, "untokenize"):
            words = tokenizer.untokenize([(p[0], p[1]) for p in best])
            return collapse_padding([str(word).replace("</w>", "") for _, word in words][:1024])
    except Exception:
        return None
    return None


def local_request(request):
    """Same machine, by address and by name; a forwarded or rebound request is not."""
    from urllib.parse import urlsplit
    host = urlsplit(f"//{request.host}").hostname
    return request.remote in ("127.0.0.1", "::1") and host in ("127.0.0.1", "localhost", "::1") and "X-Forwarded-For" not in request.headers


def install_routes():
    """Mount the same built app under ComfyUI, without opening another port."""
    import asyncio
    import secrets
    from urllib.parse import urlsplit
    from aiohttp import web
    from server import PromptServer
    from .server import STATIC, decode_body, dispatch
    token = secrets.token_urlsafe(32)

    async def api(request):
        origin = request.headers.get("Origin")
        if (origin and urlsplit(origin).netloc != request.host) or request.headers.get("Sec-Fetch-Site") == "cross-site":
            return web.json_response({"error": "Cross-origin requests are disabled"}, status=403)
        route = request.match_info["route"]
        # Installing writes to disk, so a ComfyUI started with --listen must not offer it to the network.
        if route.startswith("install/") and not local_request(request):
            return web.json_response({"error": "Models can only be installed from the computer running ComfyUI"}, status=403)
        if route == "session" and request.method == "GET":
            return web.json_response({"token": token}, headers={"Cache-Control": "no-store"})
        if request.headers.get("X-Model-Run-Token") != token:
            return web.json_response({"error": "Missing application session token"}, status=403)
        try:
            body = None
            if request.method == "POST":
                if request.content_type != "application/json":
                    raise ValueError("Expected application/json")
                if request.content_length is None or request.content_length > 16 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                body = decode_body(await request.read())
            result = await asyncio.to_thread(dispatch, application(), request.method, route, body)
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, LookupError, OSError) as exc:
            return web.json_response({"error": str(exc)[:2048]}, status=400)
        except RuntimeError as exc:
            return web.json_response({"error": str(exc)[:2048]}, status=409)

    async def asset(request):
        path = (STATIC / (request.match_info.get("asset") or "index.html")).resolve()
        if not path.is_relative_to(STATIC.resolve()) or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"X-Content-Type-Options": "nosniff"})

    routes = PromptServer.instance.routes
    routes.get("/model-run/api/{route:.*}")(api)
    routes.post("/model-run/api/{route:.*}")(api)
    routes.get("/model-run/")(asset)
    routes.get("/model-run/{asset:.*}")(asset)
