"""Opt-in native MODEL observation. Never loads, clones weights, or changes sampling."""
from threading import Lock
import time

from .capture import capture_forward
from .structure import inspect_module_tree


class DenoiserObserver:
    def __init__(self, patcher, module_paths, *, max_calls=8, elements=1024, publish=None, output_selectors=None):
        from .application import bounded_int
        self.max_calls = bounded_int(max_calls, 1, 32, "max_calls")
        self.elements = bounded_int(elements, 1, 4096, "elements")
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
            self._lock.release()

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
            self.trace["limitations"].extend(["Denoiser calls are not sampler steps; guidance may invoke the denoiser multiple times.",
                "Call budget applies to this node instance until it is re-executed. ComfyUI owns sampling and cancellation."])
        else:
            self.trace["events"].extend(trace["events"])
            self.trace["elapsed_host_seconds"] += trace["elapsed_host_seconds"]
            self.trace["status"] = trace["status"]
            self.trace["dropped_events"] += trace["dropped_events"]
        self.trace["captured_calls"] = call + 1
        if self.publish is not None:
            self.publish(self.trace)


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
                    "tooltip": "JSON mapping from module path to output tensor path, e.g. {\"transformer_blocks.0\": [1]} selects the audio tensor in an LTX block tuple."})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Fresh observation budget for every queued execution.

    def observe(self, model, modules, max_calls, elements, output_selectors="{}"):
        import json
        if not isinstance(output_selectors, str) or len(output_selectors) > 8192:
            raise ValueError("Output selector JSON exceeds 8 KiB")
        selectors = json.loads(output_selectors)
        clone = model.clone()
        app = application()
        last_id = [None]
        def publish(trace):
            # Replace this run's snapshot; other runs remain in bounded history.
            result = app.publish(trace, type(model.model.diffusion_model).__name__, replace_id=last_id[0])
            last_id[0] = result["id"]
        observer = DenoiserObserver(model, [s.strip() for s in modules.splitlines()] or [""],
                                    max_calls=max_calls, elements=elements, publish=publish, output_selectors=selectors)
        clone.set_model_unet_function_wrapper(observer)
        return (clone,)


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
