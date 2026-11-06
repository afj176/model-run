"""Observe custom node packs whose samplers call their transformer directly, not through ComfyUI's model wrapper.

ComfyUI-WanVideoWrapper hands out a ComfyUI ModelPatcher around its own Wan transformer, but its
sampler calls that transformer itself, so a MODEL function wrapper never runs. Its sampler does pass
the model's (deep-copied) transformer_options into every call, so the observer travels there: the
transformer is routed once, and a call that carries an observer is recorded by the same
DenoiserObserver the Observe node uses, with the call's latent and text context put in ComfyUI's form.
Its attention goes through one module-level ``attention`` function, which is routed through the
observer's attention hook for the call, block by block, as ComfyUI's own Wan labels them (``double:N``).
"""
import sys
from contextlib import contextmanager
from pathlib import Path

OBSERVER_KEY = "model_run_observer"
MASKABLE_MODES = ("sdpa",)  # the pack's kernels that take an attention mask
LIMITATIONS = [
    "Recorded through ComfyUI-WanVideoWrapper's own sampler: each call of its transformer is one denoiser call, conditional and unconditional passes separately.",
]
UNMASKED_LIMITATION = "Token ablation needs the pack's sdpa attention mode; this run used another kernel, so no keys were masked."


def route_transformer(core):
    """Make ``core`` hand a call to the observer its transformer_options carry; once per module, otherwise unchanged."""
    if getattr(core, "_model_run_routed", False):
        return
    original = core.forward
    def forward(*args, **kwargs):
        options = kwargs.get("transformer_options")
        observer = options.get(OBSERVER_KEY) if isinstance(options, dict) else None
        if observer is None:
            return original(*args, **kwargs)
        # The recorded call runs the module again without the observer, so hooks on the whole transformer fire too.
        inner = {**kwargs, "transformer_options": {k: v for k, v in options.items() if k != OBSERVER_KEY}}
        with routed_attention(core, observer):
            return observer(lambda *_unused, **_ignored: core(*args, **inner), call_arguments(args, kwargs))
    core.forward = forward
    core._model_run_routed = True


@contextmanager
def routed_attention(core, observer):
    """For one call, send the pack's attention through ``observer.attention_override``, labelled by block."""
    module = sys.modules.get(type(core).__module__)
    original = getattr(module, "attention", None)
    blocks = getattr(core, "blocks", None)
    if not (observer.attention_block or observer.ablate_tokens) or not callable(original) or blocks is None:
        yield
        return
    current = []
    def enter(index):
        return lambda _module, _args: current.append(index)
    def leave(_module, _args, _output):
        if current:
            current.pop()
    handles = [h for i, block in enumerate(blocks) for h in (block.register_forward_pre_hook(enter(i)), block.register_forward_hook(leave))]
    def routed(q, k, v, *args, **kwargs):
        if not current or q.ndim != 4 or k.ndim != 4:
            return original(q, k, v, *args, **kwargs)
        mode = kwargs.get("attention_mode", "sdpa")
        def kernel(_q, _k, _v, _heads, *inner, **settings):
            call = dict(kwargs)
            mask = settings.get("mask")
            if mask is not None:
                if mode in MASKABLE_MODES:
                    call["attn_mask"] = mask
                else:
                    observer.note(UNMASKED_LIMITATION)
            return original(q, k, v, *args, **call)
        # The pack lays q and k out as [batch, tokens, heads, dim]; the hook reads [batch, heads, tokens, dim].
        return observer.attention_override(kernel, q.transpose(1, 2), k.transpose(1, 2), v, int(q.shape[2]),
                                           transformer_options={"block_type": "double", "block_index": current[-1]})
    module.attention = routed
    try:
        yield
    finally:
        module.attention = original
        for handle in handles:
            handle.remove()


def call_arguments(args, kwargs):
    """A WanVideoWrapper transformer call in the shape DenoiserObserver reads: latent, timestep, text context."""
    import torch
    names = ("x", "t", "context")
    given = dict(zip(names, args))
    given.update({k: kwargs[k] for k in names if k in kwargs})
    latents, context = given.get("x"), given.get("context")
    latent = torch.stack(list(latents)) if isinstance(latents, (list, tuple)) else latents
    text = context[0] if isinstance(context, (list, tuple)) and context else context
    if isinstance(text, torch.Tensor) and text.ndim == 2:
        text = text.unsqueeze(0)
    conditioning = {"transformer_options": {}}
    if isinstance(text, torch.Tensor):
        conditioning["c_crossattn"] = text
    return {"input": latent, "timestep": given.get("t"), "c": conditioning, "cond_or_uncond": [1 if kwargs.get("is_uncond") else 0]}


def pack_token_labels(core, prompt):
    """Token strings from the pack's own T5 tokenizer, in the order its text encoder feeds them; None if it cannot be read."""
    try:
        from transformers import AutoTokenizer
        from .comfy_components import readable_piece
        root = Path(sys.modules[type(core).__module__].__file__).resolve().parents[2]
        tokenizer = AutoTokenizer.from_pretrained(str(root / "configs" / "T5_tokenizer"), local_files_only=True)
        ids = tokenizer(" ".join(prompt.split()), add_special_tokens=True)["input_ids"]  # the pack cleans whitespace this way
        return [readable_piece(tokenizer.convert_ids_to_tokens(i)) for i in ids][:1024]
    except Exception:  # numbered tokens are still honest
        return None


class ModelRunObserveWanVideo:
    CATEGORY = "Model Run"
    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("observed_model",)
    FUNCTION = "observe"
    DESCRIPTION = "Observe a ComfyUI-WanVideoWrapper model. Connect between WanVideo Model Loader and WanVideo Sampler. Open Model Run to inspect captures."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("WANVIDEOMODEL",),
                             "modules": ("STRING", {"default": "", "multiline": True, "tooltip": "One module path per line. Leave empty to record every block and the final output."}),
                             "max_calls": ("INT", {"default": 8, "min": 1, "max": 32}),
                             "elements": ("INT", {"default": 1024, "min": 1, "max": 4096})},
                "optional": {"attention_block": ("STRING", {"default": "", "tooltip": "Measure attention to text in one block, double:0 to double:29 on Wan 1.3B. Empty disables."}),
                             "attention_grid": ("INT", {"default": 32, "min": 4, "max": 64, "tooltip": "Longest side of the recorded attention grid."}),
                             "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "The positive prompt, only used to label tokens with the pack's own tokenizer."}),
                             "ablate_tokens": ("STRING", {"default": "", "tooltip": "Intervention: comma-separated text token indices masked out of every attention call. Needs the sdpa attention mode."})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Fresh observation budget for every queued execution.

    def observe(self, model, modules, max_calls, elements, attention_block="", attention_grid=32, prompt="", ablate_tokens=""):
        from .comfy_bridge import DenoiserObserver, all_layer_paths, all_layer_plan, application, parse_token_list
        core = model.model.diffusion_model
        app = application()
        last_id = [None]
        notes = list(LIMITATIONS)
        def publish(trace):
            trace["model_name"] = type(core).__name__
            for note in notes:
                if note not in trace["limitations"]:
                    trace["limitations"].append(note)
            last_id[0] = app.publish(trace, trace["model_name"], replace_id=last_id[0], note=prompt or None)["id"]
        chosen = [s.strip() for s in modules.splitlines() if s.strip()]
        plan = {"max_calls": max_calls, "elements": elements}
        if not chosen:
            chosen = all_layer_paths(core)
            plan = all_layer_plan(chosen, max_calls, elements)
        block = attention_block.strip() if isinstance(attention_block, str) else ""
        ablated = parse_token_list(ablate_tokens if isinstance(ablate_tokens, str) else "")
        labels = pack_token_labels(core, prompt) if (block or ablated) and isinstance(prompt, str) and prompt.strip() else None
        observer = DenoiserObserver(model, chosen, publish=publish, **plan, attention_block=block or None, attention_grid=attention_grid,
                                    token_labels=labels, ablate_tokens=ablated)
        observer.note = lambda text: notes.append(text) if text not in notes else None
        clone = model.clone()
        options = dict(clone.model_options.get("transformer_options") or {})
        options[OBSERVER_KEY] = observer
        clone.model_options["transformer_options"] = options
        route_transformer(core)
        return (clone,)
