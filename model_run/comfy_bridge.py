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
from . import capture_limits
from .denoising import DenoisingRecorder
from .structure import inspect_module_tree
from .layer_parts import block_part_meter
from .tensor_slices import SliceRecorder
from .cross_attention import stable_audio_labels, stable_audio_3_labels, collapse_padding, UNCHECKED_LABEL_LIMITATION, joint_by_geometry, REGISTER_LIMITATION, TEXT_MIXING_LIMITATION, place_registers, measure, block_record, new_section, key_mask, text_key_range, parse_token_list, align_labels, ABLATION_LIMITATIONS

MAX_ATTENTION_CALLS = 4
MAX_SECTION_VALUES = 524288  # aggregate cross-attention map values per trace
BLOCK_CONTAINER = re.compile(r"(?:^|\.)[a-z_]*(?:blocks?|layers?)$")
TEXT_LAST_MODELS = {"AceStepConditionGenerationModel", "AsymmDiTJoint", "ErnieImageModel", "Flux3", "JoyImageTransformer3DModel", "LensTransformer2DModel", "MOSSImage", "HYVideoDiffusionTransformer"}  # the last: ComfyUI-HunyuanVideoWrapper's transformer, image then text
# Measured the same way: keys run image, text, then other conditioning, so the text is neither first nor last.
IMAGE_FIRST_MODELS = {"HiDreamImageTransformer2DModel", "StageC"}
# These call ComfyUI's attention without transformer_options, so its override never reaches them.
BYPASS_ATTENTION_MODELS = {"AceStepConditionGenerationModel", "CausalWanModel", "ErnieImageModel", "HunYuanDiT", "HunYuanDiTPlain", "PixArtMS"}
# Their latent runs along time, so a map is a strip, not an image.
AUDIO_MODELS = {"AceStepConditionGenerationModel", "ACEStepTransformer2DModel", "AudioDiffusionTransformer"}
TAIL_TEXT_MODELS = {"LensTransformer2DModel", "MageFlowTransformer2DModel", "SingleStreamDiT"}  # these drop their template openings
TEXT_LEADS = {"ACEStepTransformer2DModel": 1}  # one speaker token comes first
# Their cross-attention keys start with this prompt and go on: Hunyuan DiT's second encoder, zero padding in the others.
ZERO_PADDED_MODELS = {"Anima", "AnimateWanModel", "CameraWanModel", "CausalWanModel", "HumoWanModel", "SCAIL2WanModel", "SCAILWanModel", "VaceWanModel",
                      "WanAnimate2Model", "WanDancerModel", "WanModel", "WanModel_S2V"}
EXTRA_KEY_MODELS = {"HunYuanDiT"} | ZERO_PADDED_MODELS
SCAIL_MODELS = {"SCAIL2WanModel", "SCAILWanModel"}
# These make 3D shapes from a picture, so a map's "text" is image features and its "frames" are depth.
SECOND_TEXT_LIMITATION = ("This model reads its prompt three times over: T5 tokens, then the same Llama tokens twice, from Llama's last layer "
                          "and from the layer this block uses. A word appears once in each, and its share is split between them.")
REBUILT_PROMPT_LIMITATION = ("This model's text node adds settings such as bpm and duration to the prompt. Labels come from the prompt with "
                             "default settings, so the caption's words line up but labels after it may name the wrong setting.")
SHAPE_MODELS = {"HunYuanDiTPlain", "Hunyuan3Dv2", "LatentSeqMMFlowModel", "Trellis2"}
SHAPE_LIMITATION = ("This model makes a 3D shape from a picture. The tokens on this map are that picture's features, not words, "
                    "and a voxel model's frames are depth slices through the shape.")
QUERY_TEXT_MODELS = {"MingImage"}  # by the ComfyUI model, since its denoiser is Lumina 2's NextDiT
QUERY_TEXT_LIMITATION = ("This model's text keys are learned query tokens that its language model reads the prompt into, not the prompt's own "
                         "tokens. They are numbered, and masking one masks a query, not a word.")
LABELS_CHECKED = {"Anima", "AsymmDiTJoint", "GeneralDIT", "Kandinsky5", "CogVideoXTransformer3DModel", "LTXVModel", "MMDiT", "MiniTrainDIT", "HumoWanModel", "WanModel_S2V", "BooguTransformer2DModel", "Chroma", "ChromaRadiance", "ErnieImageModel", "Ideogram4Transformer2DModel", "JoyImageTransformer3DModel", "LensTransformer2DModel", "MageFlowTransformer2DModel", "OmniGen2Transformer2DModel", "PixDiT_T2I", "Flux", "HiDreamImageTransformer2DModel", "HunyuanVideo", "OpenAISignatureMMDITWrapper", "VaceWanModel", "MiniMaxH3Model", "MOSSImage", "NextDiT", "PixArtMS", "WanModel", "QwenImage21Transformer2DModel", "HiDreamO1Transformer", "LTXAVModel", "SingleStreamDiT"}
LABELS_CHECKED_UNET_TEXT = {768, 1024, 1280, 2048}  # SD 1.5, SD 2.x (OpenCLIP-H), the SDXL Refiner (bigG), SDXL; checked by swapping a word
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


def all_layer_paths(core, limit=None):
    """Every top-level block of a denoiser, then its final output; just the output if it has no blocks."""
    paths = layer_blocks(core)
    limit = limit or capture_limits.get("layer_paths")
    if len(paths) > limit - 1:
        paths = [paths[round(i * (len(paths) - 1) / (limit - 2))] for i in range(limit - 1)]
    return [*paths, ""]


def layer_blocks(core):
    """Every top-level block of a denoiser, in order, before any thinning."""
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
    return paths


def all_layer_plan(paths, max_calls, elements):
    """Fit a whole-model capture inside the trace bound, whatever the model's size."""
    # A larger capture size records proportionally more: Standard is the 1,000 events these were set for.
    budget = capture_limits.get("layer_events")
    prefix, slices = ALL_LAYER_PREFIX_VALUES * budget // MAX_LAYER_EVENTS, ALL_LAYER_SLICE_VALUES * budget // MAX_LAYER_EVENTS
    calls = max(1, min(max_calls, budget // len(paths)))
    events = len(paths) * calls
    return {"max_calls": calls, "elements": max(16, min(elements, prefix // events)),
            "slice_values": slices, "slice_per_event": slices // events}


# attention_block "auto": the first block in the first denoiser call whose attention reads the prompt.
AUTO_BLOCK = "auto"
AUTO_LIMITATION = ("The block was chosen automatically: the first one in the first denoiser call whose attention reads the prompt. "
                   "Name a block in Model Run · Observe to measure another.")


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


@contextmanager
def reach_replaced_attention(options, override):
    """Send attention replacement patches' kernel calls through ``override`` for one call; the options to use.

    A replacement for a block's cross-attention, as IP-Adapter's, calls ComfyUI's kernel without
    transformer_options, so ComfyUI's override never sees it. For one call, its module's reference to
    the kernel is routed through ``override`` with that patch's options, then put back.
    """
    import inspect
    replaced = options.get("patches_replace") if isinstance(options.get("patches_replace"), dict) else {}
    patches = replaced.get("attn2") if isinstance(replaced.get("attn2"), dict) else {}
    if not patches:
        yield options
        return
    from comfy.ldm.modules.attention import optimized_attention
    modules = []
    for patch in patches.values():
        module = sys.modules.get(patch.__module__ if inspect.isfunction(patch) else type(patch).__module__)
        if module is not None and getattr(module, "optimized_attention", None) is optimized_attention and module not in modules:
            modules.append(module)
    if not modules:
        yield options
        return
    current = []
    def wrap(patch):
        def run(q, k, v, extra_options):
            current.append(extra_options)
            try:
                return patch(q, k, v, extra_options)
            finally:
                current.pop()
        return run
    def routed(q, k, v, heads, *args, **kwargs):
        if not current or "transformer_options" in kwargs:
            return optimized_attention(q, k, v, heads, *args, **kwargs)
        def kernel(_q, _k, _v, _heads, *inner, **settings):
            settings.pop("transformer_options", None)  # the override already ran; do not run it twice
            return optimized_attention(q, k, v, heads, *inner, **settings)
        return override(kernel, q, k, v, heads, *args, transformer_options=current[-1], **kwargs)
    for module in modules:
        module.optimized_attention = routed
    try:
        yield {**options, "patches_replace": {**replaced, "attn2": {key: wrap(patch) for key, patch in patches.items()}}}
    finally:
        for module in modules:
            module.optimized_attention = optimized_attention


class DenoiserObserver:
    def __init__(self, patcher, module_paths, *, max_calls=8, elements=1024, publish=None, output_selectors=None,
                 attention_block=None, attention_grid=32, token_labels=None, ablate_tokens=None, slice_values=None, slice_per_event=None):
        from .application import bounded_int
        self.max_calls = bounded_int(max_calls, 1, capture_limits.CEILING["layer_calls"], "max_calls")
        self.attention_calls = capture_limits.get("attention_calls")
        self.section_values = capture_limits.get("cross_attention_values")
        self.layer_total = None  # blocks the denoiser has, when a whole-model capture thinned them
        self.elements = bounded_int(elements, 1, 4096, "elements")
        if attention_block is not None and (not isinstance(attention_block, str) or not 1 <= len(attention_block) <= 64):
            raise ValueError("attention_block must name a block such as double:0 or output:1")
        self.attention_block = attention_block or None
        self._auto_block = self.attention_block == AUTO_BLOCK
        self.attention_grid = bounded_int(attention_grid, 4, 64, "attention_grid")
        if token_labels is not None and (not isinstance(token_labels, list) or len(token_labels) > 1024):
            raise ValueError("token_labels must be a bounded list")
        self.token_labels = token_labels
        self.token_label_sets = {}
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
        most = capture_limits.get("layer_paths")
        if not module_paths or len(module_paths) > most or any(p not in names for p in module_paths):
            raise ValueError(f"Select 1 to {most} valid diffusion module paths; an empty line selects the whole denoiser")
        self.paths = list(module_paths)
        budget = 262144 * capture_limits.get("layer_events") // MAX_LAYER_EVENTS
        if len(self.paths) * self.max_calls * self.elements > budget:
            raise ValueError(f"Capture exceeds the aggregate budget of {budget:,} values. Reduce modules, calls, or elements, or choose a larger capture size.")
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
            # On the first call, how large each block's attention and MLP outputs are next to the block's own.
            meter = block_part_meter(self.core, self.paths) if call == 0 else None
            try:
                latent_shape = self._latent_shape(arguments)
                self._slices.text_tokens = self._text_token_count(arguments)
                self._slices.reference_tokens = self._reference_token_count(arguments)
                self._slices.conditional_batch = self._conditional_batch(arguments)
                self._slices.sequence = self._token_sequence(latent_shape, self._slices.text_tokens)
                try:
                    output, trace = capture_forward(Invocation(), {}, ["denoiser" + ("." + p if p else "") for p in self.paths],
                        max_tensor_elements=self.elements, max_events=len(self.paths),
                        output_selectors={"denoiser" + ("." + p if p else ""): value for p, value in self.selectors.items()},
                        slices=self._slices, latent_shape=latent_shape)
                finally:
                    if meter is not None:
                        meter[1]()
                if meter is not None:
                    self.block_parts = meter[0]()
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
            # The same class also runs other text encoders, such as Kolors', which were not checked.
            reader = next((module.attn2.to_k for module in self.core.modules() if hasattr(module, "attn2")), None)
            return getattr(reader, "in_features", None) in LABELS_CHECKED_UNET_TEXT
        return name in LABELS_CHECKED

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
        context = conditioning.get("c_crossattn", conditioning.get("context", conditioning.get("clip_text")))  # Stable Cascade C: clip_text
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
        measure_call = measure_call and self.attention_block is not None and call < self.attention_calls
        options = arguments.get("c", {}).get("transformer_options") if isinstance(arguments.get("c"), dict) else None
        order = arguments.get("cond_or_uncond")
        if not isinstance(order, list):
            order = (options or {}).get("cond_or_uncond")
        # A negative prompt run on its own has other tokens than the labels name.
        if isinstance(order, list) and order and 0 not in order:
            measure_call = False
        if not measure_call and not self.ablate_tokens:
            return None
        import torch
        conditioning = arguments.get("c") if isinstance(arguments.get("c"), dict) else {}
        context = conditioning.get("c_crossattn", conditioning.get("context", conditioning.get("clip_text")))  # Stable Cascade C: clip_text
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
        text_tokens = int(context.shape[1])
        llama = conditioning.get("encoder_hidden_states_llama3")
        if type(self.core).__name__ == "HiDreamImageTransformer2DModel" and isinstance(llama, torch.Tensor) and llama.ndim == 4:
            # Its keys after the image are T5's tokens, then Llama's twice: its last layer and this block's layer.
            text_tokens += 2 * int(llama.shape[-2])
        t5 = conditioning.get("encoder_hidden_states_t5")
        if type(self.core).__name__ == "HunYuanDiT" and isinstance(t5, torch.Tensor) and t5.ndim == 3:
            # Its keys are the BERT tokens, then mT5's.
            text_tokens += int(t5.shape[1])
        if type(self.core).__name__ in ZERO_PADDED_MODELS:
            # They pad the prompt to a fixed length with zeros, which are not tokens of the prompt.
            filled = context[batch_index].abs().sum(-1).nonzero()
            text_tokens = int(filled.max()) + 1 if len(filled) else text_tokens
        if type(self.core).__name__ == "AsymmDiTJoint" and isinstance(conditioning.get("num_tokens"), (int, float)):
            # Mochi drops its prompt's padding before attention, keeping only these tokens.
            text_tokens = min(text_tokens, max(1, int(conditioning["num_tokens"])))
        reference = conditioning.get("reference_latent")
        # SCAIL puts its reference frames before the video in the query sequence.
        scail_lead = 0
        if type(self.core).__name__ in SCAIL_MODELS and isinstance(reference, torch.Tensor) and reference.ndim == 5:
            patch = self.core.patch_size
            scail_lead = (int(reference.shape[-3]) // patch[0]) * (int(reference.shape[-2]) // patch[1]) * (int(reference.shape[-1]) // patch[2])
        text_lead = 0
        if type(self.core).__name__ == "HunyuanVideo":
            # Hunyuan Video 1.5 puts an image prompt's SigCLIP tokens, then any byT5 glyph tokens, before the prompt's.
            fea, byt5 = conditioning.get("clip_fea"), conditioning.get("txt_byt5")
            if getattr(self.core, "vision_in", None) is not None and isinstance(fea, torch.Tensor) and fea.ndim == 3:
                text_lead += int(fea.shape[1])
            if getattr(self.core, "cond_type_embedding", None) is not None and getattr(self.core, "byt5_in", None) is not None and isinstance(byt5, torch.Tensor) and byt5.ndim == 3:
                text_lead += int(byt5.shape[1])
        return {"call": call, "latent_shape": latent_shape, "audio_steps": audio_steps, "reference_tokens": self._reference_token_count(arguments), "text_tokens": text_tokens,
                "text_lead": text_lead, "scail_lead": scail_lead, "batch_index": batch_index, "batch_label": batch_label, "record": None, "reason": None,
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
            frames = int(latent_shape[-3]) // int(self.core.patch_size[0]) if len(latent_shape) == 5 else 1
            return frames * (int(latent_shape[-2]) // int(self.core.patch_size[1])) * (int(latent_shape[-1]) // int(self.core.patch_size[2])), 0
        if name in ("NextDiT", "NextDiTPixelSpace"):
            patch = int(self.core.patch_size)
            return (int(latent_shape[-2]) // patch) * (int(latent_shape[-1]) // patch), 0
        if name == "Fibo2":
            # Queries run summary tokens, the image, then any reference images.
            gist = sys.modules[type(self.core).__module__].gist_count(text_tokens, self.core.max_gist_tokens)
            return int(latent_shape[-2]) * int(latent_shape[-1]), gist
        if name in SCAIL_MODELS and len(latent_shape) == 5:
            # Queries run the reference frames, the video, then pose tokens; only the video gets a map.
            patch = self.core.patch_size
            video = (int(latent_shape[-3]) // patch[0]) * (int(latent_shape[-2]) // patch[1]) * (int(latent_shape[-1]) // patch[2])
            return video, int((self._current or {}).get("scail_lead") or 0)
        return None, 0

    def attention_override(self, func, q, k, v, heads, *args, **kwargs):
        """ComfyUI optimized_attention_override: measure the selected block, then run the real kernel."""
        current = self._current
        if current is not None:
            current["override_seen"] = True
        text_start = self._text_start if self._text_start == -1 or current is None else self._text_start + int(current.get("text_lead") or 0)
        if current is not None and self.ablate_tokens:
            try:
                options = kwargs.get("transformer_options") or {}
                key_count = int(k.shape[-2] if k.ndim == 4 else k.shape[1])
                # Joint text+image attention is only assumed where the model labels its blocks as DiT blocks;
                # an unlabelled call with equal query and key counts is self-attention and has no text keys.
                query_count = int(q.shape[-2] if q.ndim == 4 else q.shape[1])
                joint = "block_type" in options or "block" not in options and joint_by_geometry(query_count, key_count, current["text_tokens"], current["latent_shape"], self._text_lead)
                known, lead = self._image_layout(current["latent_shape"], current["text_tokens"])
                # A known layout says it is joint attention where the lengths alone cannot.
                joint = joint or bool(known and not lead and query_count == key_count and query_count > known)
                span = text_key_range(query_count, key_count, current["text_tokens"], allow_joint=joint, text_start=text_start)
                if self._image_first:
                    start = self._image_layout(current["latent_shape"], current["text_tokens"])[0] or query_count
                    span = (start, start + current["text_tokens"]) if key_count >= start + current["text_tokens"] else None
                elif type(self.core).__name__ in EXTRA_KEY_MODELS and query_count != key_count and key_count >= current["text_tokens"]:
                    span = (0, current["text_tokens"])
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
                # auto takes the first block whose attention reads the prompt, then keeps it.
                if self.attention_block == AUTO_BLOCK or self.attention_block in (label, audio_label):
                    mask = kwargs.get("mask", args[0] if args else None)
                    known, lead = self._image_layout(current["latent_shape"], current["text_tokens"])
                    if self._audio_only and is_audio:
                        known = current["audio_steps"]
                    elif type(self.core).__name__ in EXTRA_KEY_MODELS and key_count != query_count:
                        known = known or query_count
                    # Z-Image pads its text to a multiple, so the image starts where the queries' length says.
                    image_at = query_count - known if known and type(self.core).__name__ in ("NextDiT", "NextDiTPixelSpace") else None
                    measurement = measure(q, k, heads=heads, text_tokens=current["text_tokens"], latent_shape=current["latent_shape"],
                                          mask=mask, batch_index=current["batch_index"], max_grid=self.attention_grid, stream="audio" if is_audio else None, text_start=text_start, reference_tokens=current.get("reference_tokens", 0),
                                          image_first=self._image_first, known_image_tokens=known, query_lead=lead, image_at=image_at)
                    if "unavailable" in measurement:
                        current["reason"] = measurement["unavailable"]
                    elif not (self._image_first or known) and (measurement["attention_kind"] == "joint" and not allow_joint and not label.startswith("call:") or measurement["attention_kind"] == "joint" and not allow_joint and not joint_by_geometry(
                            int(q.shape[-2] if q.ndim == 4 else q.shape[1]), int(k.shape[-2] if k.ndim == 4 else k.shape[1]), current["text_tokens"], current["latent_shape"], self._text_lead)):
                        current["reason"] = "Self-attention in this block has no text keys"
                    else:
                        self._mixes_text = self._mixes_text or bool(measurement.get("text_spreads"))
                        if self.attention_block == AUTO_BLOCK:
                            self.attention_block = audio_label or label
                            section = (self.trace or {}).get("cross_attention")
                            if section is not None:
                                section["selection"] = self.attention_block
                        current["record"] = block_record(self.attention_block, denoiser_call=current["call"], batch_label=current["batch_label"],
                                                         measurement=measurement, block_label=audio_label or label)
            except Exception as exc:  # measurement must never break sampling
                current["reason"] = f"{type(exc).__name__}: {exc}"
        if self.previous_attention is not None:
            return self.previous_attention(func, q, k, v, heads, *args, **kwargs)
        return func(q, k, v, heads, *args, **kwargs)

    def _invoke(self, function, args):
        if not (self.attention_block or self.ablate_tokens):
            return self._invoke_reached(function, args)
        conditioning = args.get("c")
        options = conditioning.get("transformer_options") if isinstance(conditioning, dict) else None
        if not isinstance(options, dict):
            return self._invoke_reached(function, args)
        # A node after this one, such as Model Attention Backend, may have put its own override in place.
        found = options.get("optimized_attention_override")
        if found is not None and found != self.attention_override:
            self.previous_attention = found
        # ComfyUI's override wins over a module's own kernel, so a quantized model must keep the override out of it.
        own_kernel = self.previous_attention is None and any(type(m).__name__ == "ComfyAttention" and m.function is not None for m in self.core.modules())
        options = dict(options)
        if own_kernel:
            options.pop("optimized_attention_override", None)
        else:
            options["optimized_attention_override"] = self.attention_override
        with reach_replaced_attention(options, self.attention_override) as options:
            args = {**args, "c": {**conditioning, "transformer_options": options}}
            if own_kernel or type(self.core).__name__ in BYPASS_ATTENTION_MODELS:
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
                if getattr(self.base_model, "padding_embedding", None) is not None:
                    aligned = stable_audio_3_labels(self.token_labels, current["text_tokens"])
                else:
                    aligned = stable_audio_labels(self.token_labels, current["text_tokens"])
            if aligned is None and type(self.core).__name__ in ZERO_PADDED_MODELS | {"AsymmDiTJoint"} and len(self.token_labels) > current["text_tokens"]:
                # Mochi keeps the front of its padded prompt, as the zero-padded models do.
                aligned = list(self.token_labels[:current["text_tokens"]])
            if aligned is None:
                aligned = labels_from_encoders(type(self.core).__name__, self.token_label_sets, current["text_tokens"])
            if aligned is None:
                # With several encoders the longest list is read first; Kandinsky 5 Image's keys come from its shorter Qwen list.
                aligned = next((found for found in (align_labels(own, current["text_tokens"]) for own in self.token_label_sets.values()) if found), None)
            if aligned is None and type(self.core).__name__ in TAIL_TEXT_MODELS and len(self.token_labels) > current["text_tokens"]:
                # These encoders keep only what follows the template's opening, ending with the same tokens.
                aligned = list(self.token_labels[-current["text_tokens"]:])
            if aligned is None and type(self.core).__name__ == "JoyImageTransformer3DModel":
                aligned = vision_gap_labels(self.token_labels, current["text_tokens"], drop=34)
            if aligned is None and type(self.core).__name__ == "PixDiT_T2I" and len(self.token_labels) > current["text_tokens"] > 1:
                # PixelDiT keeps the opening token and the last 299 of its padded instruction and prompt.
                aligned = [self.token_labels[0], *self.token_labels[-(current["text_tokens"] - 1):]]
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
            if getattr(self, "block_parts", None):
                self.trace["block_parts"] = self.block_parts
            if self.attention_block is not None:
                self.trace["cross_attention"] = new_section(self.attention_block, "comfyui_attention_override",
                                                            token_labels=self.token_labels, max_grid=self.attention_grid)
                self.trace["cross_attention"]["max_calls"] = self.attention_calls
                if self._auto_block:
                    self.trace["cross_attention"]["limitations"].append(AUTO_LIMITATION)
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
                if used + len(record["maps"]) * len(record["maps"][0]) > self.section_values:
                    self._skip(call, f"Cross-attention value budget of {self.section_values:,} reached; earlier calls are kept")
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
        if type(self.core).__name__ == "AceStepConditionGenerationModel" and self.token_labels and "cross_attention" in self.trace and REBUILT_PROMPT_LIMITATION not in self.trace["cross_attention"]["limitations"]:
            self.trace["cross_attention"]["limitations"].append(REBUILT_PROMPT_LIMITATION)
        if type(self.core).__name__ == "HiDreamImageTransformer2DModel" and "cross_attention" in self.trace and SECOND_TEXT_LIMITATION not in self.trace["cross_attention"]["limitations"]:
            self.trace["cross_attention"]["limitations"].append(SECOND_TEXT_LIMITATION)
        if type(self.core).__name__ in SHAPE_MODELS and "cross_attention" in self.trace and SHAPE_LIMITATION not in self.trace["cross_attention"]["limitations"]:
            self.trace["cross_attention"]["limitations"].append(SHAPE_LIMITATION)
        if type(self.base_model).__name__ in QUERY_TEXT_MODELS and "cross_attention" in self.trace and QUERY_TEXT_LIMITATION not in self.trace["cross_attention"]["limitations"]:
            self.trace["cross_attention"]["limitations"].append(QUERY_TEXT_LIMITATION)
        if self._mixes_text and "cross_attention" in self.trace and TEXT_MIXING_LIMITATION not in self.trace["cross_attention"]["limitations"]:
            self.trace["cross_attention"]["limitations"].append(TEXT_MIXING_LIMITATION)
        if "cross_attention" in self.trace and self._skipped:
            self.trace["cross_attention"]["skipped"] = list(self._skipped)
        if self.publish is not None:
            self._publish_safely(call)

    def note_kept(self, trace):
        """Run Metrics rows: the layers, calls and maps this run held against what it kept."""
        layers = len([p for p in self.paths if p])
        if self.layer_total is not None and self.layer_total > layers:
            capture_limits.kept(trace, "Denoiser layers", f"{self.layer_total:,}", f"{layers:,} (evenly spaced)", "Raise ComfyUI layers to record more of them.")
        elif layers:
            capture_limits.kept(trace, "Denoiser layers", f"{self.layer_total or layers:,}", f"{layers:,}")
        if self.calls:
            short = self.calls > self.max_calls
            capture_limits.kept(trace, "Denoiser calls with layers", f"{self.calls:,}", f"{min(self.calls, self.max_calls):,} (the first ones)" if short else f"{self.calls:,}",
                                "Raise Denoiser calls with layers to record layers on later calls." if short else None)
        section = trace.get("cross_attention")
        if section is not None and self.calls:
            mapped = len(section.get("events") or [])
            capture_limits.kept(trace, "Cross-attention calls", f"{self.calls:,}", f"{mapped:,}",
                                "Raise Cross-attention calls to map more of them." if self.calls > self.attention_calls else None)
        steps = (trace.get("denoising") or {}).get("steps") or []
        total = (trace.get("denoising") or {}).get("total_steps")
        if steps:
            capture_limits.kept(trace, "Denoising steps", f"{total or len(steps):,}", f"{len(steps):,}",
                                "Every few steps, evenly; raise Denoising steps to keep all of them." if total and total + 1 > len(steps) else None)

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
        # Runs here packages live beside this checkout in both hosts, so one download serves both.
        packages = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "packages")
        os.makedirs(packages, exist_ok=True)
        _application = Application(entries, host_kind="comfyui", install_folders=install_folders,
                                   comfy_dir=[folder_paths.base_path, os.path.dirname(os.path.abspath(folder_paths.__file__))],
                                   packages_dir=packages)
        _application.scan()
    return _application


def source_images(graph, *, limit=4, side=384):
    """Small PNGs of the images a workflow's Load Image nodes read, so a run shows its source beside its output."""
    from PIL import Image, ImageOps
    import folder_paths
    from .comfy_instrument import media_inputs
    found = []
    for item in media_inputs(graph if isinstance(graph, dict) else {}):
        if item["kind"] != "image" or len(found) >= limit:
            continue
        try:
            with Image.open(folder_paths.get_annotated_filepath(item["file"])) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((side, side))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        except Exception:  # noqa: BLE001 - a missing or unreadable file only loses the thumbnail
            continue
        found.append({"title": str(item["title"])[:160], "file": str(item["file"])[:256], "mime": "image/png",
                      "base64": base64.b64encode(buffer.getvalue()).decode()})
    return found


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
                "max_calls": ("INT", {"default": 8, "min": 1, "max": capture_limits.CEILING["layer_calls"], "tooltip": "Denoiser calls whose layers are recorded, from the first."}),
                "elements": ("INT", {"default": 1024, "min": 1, "max": 4096})},
                "optional": {"output_selectors": ("STRING", {"default": "{}", "multiline": True,
                    "tooltip": "JSON mapping from module path to output tensor path, e.g. {\"transformer_blocks.0\": [1]} selects the audio tensor in an LTX block tuple."}),
                    "attention_block": ("STRING", {"default": "", "tooltip": "Measure attention to text in one block: audio:0 for the audio stream of an audio+video model, double:0 or single:3 for DiT models, input:4 / middle:0 / output:1 for UNets, or auto for the first block that reads the prompt. Empty disables."}),
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
        sources = []
        watching = []
        def publish(trace):
            # Replace this run's snapshot; other runs remain in bounded history.
            trace["model_name"] = type(model.model.diffusion_model).__name__
            if watching:
                watching[0].note_kept(trace)
            if not sources and isinstance(workflow_graph, dict):
                sources.append(source_images(workflow_graph))
            if sources and sources[0]:
                trace["source_images"] = sources[0]
            if isinstance(workflow_graph, dict) and unique_id is not None:
                trace["native_rerun"] = True
            result = app.publish(trace, trace["model_name"], replace_id=last_id[0], note=prompt if isinstance(prompt, str) else None)
            last_id[0] = result["id"]
            if trace.get("native_rerun"):
                app.remember_workflow(result["id"], workflow_graph, unique_id)
        chosen = [s.strip() for s in modules.splitlines() if s.strip()]
        plan = {"max_calls": max_calls, "elements": elements}
        total = None
        if not chosen:
            total = len(layer_blocks(model.model.diffusion_model))
            chosen = all_layer_paths(model.model.diffusion_model)
            plan = all_layer_plan(chosen, max_calls, elements)
        observer = DenoiserObserver(model, chosen, publish=publish, **plan,
                                     output_selectors=selectors,
                                    attention_block=block or None, attention_grid=attention_grid, token_labels=labels, ablate_tokens=ablated)
        observer.layer_total = total
        watching.append(observer)
        observer.token_label_sets = encoder_labels(clip, prompt) if labels else {}
        clone.set_model_unet_function_wrapper(observer)
        if denoising_views:
            recorder = DenoisingRecorder(getattr(model.model, "latent_format", None), on_update=observer.attach_denoising,
                                         max_steps=capture_limits.get("denoising_steps"))
            clone.set_model_sampler_post_cfg_function(recorder)
        if block or ablated:
            options = clone.model_options.setdefault("transformer_options", {})
            observer.previous_attention = options.get("optimized_attention_override")
            options["optimized_attention_override"] = observer.attention_override
        return (clone,)


def encoder_labels(clip, prompt):
    """Each text encoder's own token strings, for models whose keys come from particular encoders."""
    try:
        from .comfy_components import readable_piece
        tokenized = clip.tokenize(prompt[:8192])
        tokenizer = getattr(clip, "tokenizer", None)
        found = {}
        for key, value in (tokenized.items() if isinstance(tokenized, dict) else []):
            own = getattr(tokenizer, key, None) or getattr(tokenizer, f"clip_{key}", None)
            if isinstance(value, list) and hasattr(own, "untokenize"):
                flat = [(pair[0], pair[1]) for chunk in value if isinstance(chunk, (list, tuple)) for pair in chunk]
                found[key] = collapse_padding([readable_piece(word) for _, word in own.untokenize(flat)][:1024])
        return found
    except Exception:  # numbered tokens are still honest
        return {}


def labels_from_encoders(model, sets, text_tokens):
    """Labels for a model's text keys from the encoder that fills them, or None when the counts say otherwise."""
    if model == "OpenAISignatureMMDITWrapper":
        # SD3 keys run the CLIP tokens, then T5's.
        first, t5 = sets.get("l") or sets.get("g"), sets.get("t5xxl")
        return [*first, *t5] if first and t5 and len(first) + len(t5) == text_tokens else None
    if model in ("HunyuanVideo", "HYVideoDiffusionTransformer") and "user" in sets.get("llama", []):
        # Its encoder cuts the template's opening: the keys start after the user header and its line break.
        # With an image prompt (Hunyuan Video I2V), its LLaVA image tokens come first, so they are the shortfall.
        start = sets["llama"].index("user") + 3
        kept = sets["llama"][start:start + text_tokens]
        return [*["<image>"] * (text_tokens - len(kept)), *kept] if kept else None
    if model == "HunYuanDiT":
        # Its keys are its BERT encoder's tokens, then mT5's.
        bert, mt5 = sets.get("hydit_clip", []), sets.get("mt5xl", [])
        return [*bert, *mt5] if bert and len(bert) + len(mt5) == text_tokens else None
    if model == "HiDreamImageTransformer2DModel":
        t5, llama = sets.get("t5xxl", []), sets.get("llama", [])
        if t5 and llama and len(t5) + 2 * len(llama) == text_tokens:
            return [*t5, *llama, *llama]
    return None


def vision_gap_labels(labels, text_tokens, drop=0):
    """Labels for an edit model whose keys hold the reference image's tokens right after the user header.

    The prompt is tokenised without the image, so the image block is the difference in length, and it sits
    where the template puts it: after "user" and its line break. JoyImage also drops its system prompt.
    """
    kept = list(labels[drop:])
    gap = text_tokens - len(kept) if type(text_tokens) is int else -1
    if gap == 0:
        return kept
    at = next((i + 2 for i in range(len(kept) - 1) if kept[i].strip() == "user" and kept[i + 1] == "\n"), None)
    if gap < 0 or at is None:
        return None
    return [*kept[:at], *["<image>"] * gap, *kept[at:]]


def token_labels(clip, prompt):
    """Best-effort token strings from the CLIP's own tokenizer; None when unavailable."""
    try:
        tokenized = clip.tokenize(prompt[:8192])
        best, best_key, lists = [], None, 0
        for key, value in (tokenized.items() if isinstance(tokenized, dict) else [(None, tokenized)]):
            if not isinstance(value, list):
                continue  # tokenizers may attach flags such as keep_vision
            lists += 1
            flat = [pair for chunk in value if isinstance(chunk, (list, tuple)) for pair in chunk]
            if len(flat) > len(best):
                best, best_key = flat, key
        if not best:
            return None
        tokenizer = getattr(clip, "tokenizer", None)
        from .comfy_components import readable_piece
        # With several encoders, a shared decode() reads one encoder's ids, which turns another's into wrong words.
        own = getattr(tokenizer, best_key, None) if isinstance(best_key, str) else None
        pieces = None
        reader = own if hasattr(own, "untokenize") else tokenizer
        if hasattr(reader, "untokenize"):
            pieces = [readable_piece(word) for _, word in reader.untokenize([(p[0], p[1]) for p in best])]
        if hasattr(own, "untokenize") and lists > 1:
            return collapse_padding(pieces[:1024])
        if hasattr(tokenizer, "decode"):
            # Per-token decode turns byte-level BPE pieces into readable text. T5 decodes its space, end and padding tokens to nothing.
            return collapse_padding([str(tokenizer.decode([p[0]], skip_special_tokens=False)) or (pieces[i] if pieces else str(p[0])) for i, p in enumerate(best)][:1024])
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
    from .trace_store import size_limit, too_large
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
                if request.content_length is None:
                    raise ValueError("Invalid request size")
                if request.content_length > size_limit():
                    raise too_large(request.content_length)
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
