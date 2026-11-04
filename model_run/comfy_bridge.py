"""Opt-in native MODEL observation. Never loads, clones weights, or changes sampling."""
from threading import Lock
import time

from .capture import capture_forward
from .structure import inspect_module_tree
from .cross_attention import measure, block_record, new_section, key_mask, text_key_range, parse_token_list, ABLATION_LIMITATIONS

MAX_ATTENTION_CALLS = 4
MAX_SECTION_VALUES = 524288  # aggregate cross-attention map values per trace


class DenoiserObserver:
    def __init__(self, patcher, module_paths, *, max_calls=8, elements=1024, publish=None, output_selectors=None,
                 attention_block=None, attention_grid=32, token_labels=None, ablate_tokens=None):
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
        self.ablate_tokens = parse_token_list(ablate_tokens)
        self._ablation_applied = 0
        self.previous_attention = None
        self._current = None
        self._skipped = []
        self.core = patcher.model.diffusion_model
        names = dict(self.core.named_modules())
        if not module_paths or len(module_paths) > 16 or any(p not in names for p in module_paths):
            raise ValueError("Select 1 to 16 valid diffusion module paths; an empty line selects the whole denoiser")
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
                output, trace = capture_forward(Invocation(), {}, ["denoiser" + ("." + p if p else "") for p in self.paths],
                    max_tensor_elements=self.elements, max_events=len(self.paths),
                    output_selectors={"denoiser" + ("." + p if p else ""): value for p, value in self.selectors.items()})
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

    def _attention_context(self, call, arguments, measure_call=True):
        """Geometry the attention override needs for this denoiser call, or None when not measuring."""
        measure_call = measure_call and self.attention_block is not None and call < MAX_ATTENTION_CALLS
        if not measure_call and not self.ablate_tokens:
            return None
        import torch
        conditioning = arguments.get("c") if isinstance(arguments.get("c"), dict) else {}
        context = conditioning.get("c_crossattn", conditioning.get("context"))
        latent = arguments.get("input")
        if not isinstance(context, torch.Tensor) or context.ndim != 3 or not isinstance(latent, torch.Tensor):
            self._skip(call, "Denoiser call has no text context tensor")
            return None
        order = arguments.get("cond_or_uncond")
        if not isinstance(order, list):
            order = (conditioning.get("transformer_options") or {}).get("cond_or_uncond")
        batch_index, batch_label = 0, "batch 0"
        if isinstance(order, list) and len(order) == int(latent.shape[0]) and 0 in order:
            batch_index, batch_label = order.index(0), "conditional (cond_or_uncond)"
        return {"call": call, "latent_shape": list(latent.shape), "text_tokens": int(context.shape[1]),
                "batch_index": batch_index, "batch_label": batch_label, "record": None, "reason": None,
                "measure": measure_call}

    def _skip(self, call, reason):
        if len(self._skipped) < 32:
            self._skipped.append({"denoiser_call": call, "reason": str(reason)[:200]})

    def attention_override(self, func, q, k, v, heads, *args, **kwargs):
        """ComfyUI optimized_attention_override: measure the selected block, then run the real kernel."""
        current = self._current
        if current is not None and self.ablate_tokens:
            try:
                options = kwargs.get("transformer_options") or {}
                span = text_key_range(int(q.shape[-2] if q.ndim == 4 else q.shape[1]), int(k.shape[-2] if k.ndim == 4 else k.shape[1]),
                                      current["text_tokens"], allow_joint="block_type" in options)
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
                    label, allow_joint = None, False
                if label is not None:
                    seen = current.setdefault("seen", [])
                    if label not in seen and len(seen) < 64:
                        seen.append(label)
                if label == self.attention_block:
                    mask = kwargs.get("mask", args[0] if args else None)
                    measurement = measure(q, k, heads=heads, text_tokens=current["text_tokens"], latent_shape=current["latent_shape"],
                                          mask=mask, batch_index=current["batch_index"], max_grid=self.attention_grid)
                    if "unavailable" in measurement:
                        current["reason"] = measurement["unavailable"]
                    elif measurement["attention_kind"] == "joint" and not allow_joint:
                        current["reason"] = "Self-attention in this block has no text keys"
                    else:
                        current["record"] = block_record(self.attention_block, denoiser_call=current["call"], batch_label=current["batch_label"],
                                                         measurement=measurement, block_label=label)
            except Exception as exc:  # measurement must never break sampling
                current["reason"] = f"{type(exc).__name__}: {exc}"
        if self.previous_attention is not None:
            return self.previous_attention(func, q, k, v, heads, *args, **kwargs)
        return func(q, k, v, heads, *args, **kwargs)

    def _invoke(self, function, args):
        if self.previous is not None:
            return self.previous(function, args)
        return function(args["input"], args["timestep"], **args["c"])

    def _publish(self, trace, call):
        for event in trace["events"]:
            event["module"] = event["module"][len("denoiser"):].lstrip(".")
            event["denoiser_call"] = call
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
                hint = f"; blocks seen: {seen[0]} … {seen[-1]} ({len(seen)})" if seen else "; no labelled attention calls were seen"
                self._skip(call, current["reason"] or f"No attention call matched block {self.attention_block}{hint}")
            current["record"] = None
        if "cross_attention" in self.trace and self._skipped:
            self.trace["cross_attention"]["skipped"] = list(self._skipped)
        if self.publish is not None:
            self._publish_safely(call)

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
        _application = Application(entries, host_kind="comfyui")
        _application.scan()
    return _application


class ModelRunObserve:
    CATEGORY = "Model Run"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("observed_model",)
    FUNCTION = "observe"
    DESCRIPTION = "Observe bounded native denoiser outputs. Connect between your model loader and sampler. Open Model Run to inspect captures."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "modules": ("STRING", {"default": "", "multiline": True}),
                "max_calls": ("INT", {"default": 8, "min": 1, "max": 32}),
                "elements": ("INT", {"default": 1024, "min": 1, "max": 4096})},
                "optional": {"output_selectors": ("STRING", {"default": "{}", "multiline": True,
                    "tooltip": "JSON mapping from module path to output tensor path, e.g. {\"transformer_blocks.0\": [1]} selects the audio tensor in an LTX block tuple."}),
                    "attention_block": ("STRING", {"default": "", "tooltip": "Measure image-to-text attention in one block: double:0 or single:3 for DiT models, input:4 / middle:0 / output:1 for UNets. Empty disables."}),
                    "attention_grid": ("INT", {"default": 32, "min": 4, "max": 64, "tooltip": "Longest side of the recorded attention grid."}),
                    "clip": ("CLIP", {"tooltip": "Optional: with prompt, labels text tokens in the attention maps."}),
                    "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "The positive prompt, only used to label tokens."}),
                    "ablate_tokens": ("STRING", {"default": "", "tooltip": "Intervention: comma-separated text token indices whose keys are masked out of every attention call for this run. Compare the result with a run that leaves this empty."})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Fresh observation budget for every queued execution.

    def observe(self, model, modules, max_calls, elements, output_selectors="{}", attention_block="", attention_grid=32, clip=None, prompt="", ablate_tokens=""):
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
            result = app.publish(trace, type(model.model.diffusion_model).__name__, replace_id=last_id[0])
            last_id[0] = result["id"]
        observer = DenoiserObserver(model, [s.strip() for s in modules.splitlines()] or [""],
                                    max_calls=max_calls, elements=elements, publish=publish, output_selectors=selectors,
                                    attention_block=block or None, attention_grid=attention_grid, token_labels=labels, ablate_tokens=ablated)
        clone.set_model_unet_function_wrapper(observer)
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
            return [str(tokenizer.decode([p[0]], skip_special_tokens=False)) for p in best][:1024]
        if hasattr(tokenizer, "untokenize"):
            words = tokenizer.untokenize([(p[0], p[1]) for p in best])
            return [str(word).replace("</w>", "") for _, word in words][:1024]
    except Exception:
        return None
    return None


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
