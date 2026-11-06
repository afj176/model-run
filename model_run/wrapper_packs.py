"""Observe custom node packs whose samplers call their transformer directly, not through ComfyUI's model wrapper.

ComfyUI-WanVideoWrapper hands out a ComfyUI ModelPatcher around its own Wan transformer, but its
sampler calls that transformer itself, so a MODEL function wrapper never runs. Its sampler does pass
the model's (deep-copied) transformer_options into every call, so the observer travels there: the
transformer is routed once, and a call that carries an observer is recorded by the same
DenoiserObserver the Observe node uses, with the call's latent and text context put in ComfyUI's form.
"""
OBSERVER_KEY = "model_run_observer"
LIMITATIONS = [
    "Recorded through ComfyUI-WanVideoWrapper's own sampler: each call of its transformer is one denoiser call, conditional and unconditional passes separately.",
    "The pack uses its own attention kernels, so attention maps and token ablation are not available for it.",
]


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
        return observer(lambda *_unused, **_ignored: core(*args, **inner), call_arguments(args, kwargs))
    core.forward = forward
    core._model_run_routed = True


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
                             "elements": ("INT", {"default": 1024, "min": 1, "max": 4096})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Fresh observation budget for every queued execution.

    def observe(self, model, modules, max_calls, elements):
        from .comfy_bridge import DenoiserObserver, all_layer_paths, all_layer_plan, application
        core = model.model.diffusion_model
        app = application()
        last_id = [None]
        def publish(trace):
            trace["model_name"] = type(core).__name__
            for note in LIMITATIONS:
                if note not in trace["limitations"]:
                    trace["limitations"].append(note)
            last_id[0] = app.publish(trace, trace["model_name"], replace_id=last_id[0])["id"]
        chosen = [s.strip() for s in modules.splitlines() if s.strip()]
        plan = {"max_calls": max_calls, "elements": elements}
        if not chosen:
            chosen = all_layer_paths(core)
            plan = all_layer_plan(chosen, max_calls, elements)
        observer = DenoiserObserver(model, chosen, publish=publish, **plan)
        clone = model.clone()
        options = dict(clone.model_options.get("transformer_options") or {})
        options[OBSERVER_KEY] = observer
        clone.model_options["transformer_options"] = options
        route_transformer(core)
        return (clone,)
