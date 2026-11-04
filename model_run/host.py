"""In-process ownership and serial observation scheduling for explicit runtime bindings."""
from copy import deepcopy
from threading import Lock
from uuid import uuid4

from .adapters import UnsupportedCapability
from .sessions import ObservationSession

ACTIVE = {"pending", "running", "cancelling"}


class ModelHost:
    """Own binding references and optional idempotent release callbacks.

    One observation or release operation runs at a time in this host. This does
    not reserve VRAM or coordinate other hosts/processes.
    """
    def __init__(self, *, max_models=8):
        if type(max_models) is not int or not 1 <= max_models <= 64:
            raise ValueError("max_models must be between 1 and 64")
        self._max_models = max_models
        self._lock = Lock()
        self._session = ObservationSession()
        self._models = {}
        self._closed = False
        self._maintenance = False
        self._closing = False
        self._selection = None

    def _require_idle(self):
        if self._maintenance or self._session.snapshot()["status"] in ACTIVE:
            raise RuntimeError("Host has an active observation or resource operation")

    def register(self, model_id, adapter, *, release=None):
        if not isinstance(model_id, str) or not 1 <= len(model_id) <= 128:
            raise ValueError("model_id must contain 1 to 128 characters")
        if not callable(getattr(adapter, "capabilities", None)) or not callable(getattr(adapter, "negotiate", None)):
            raise TypeError("adapter must implement the runtime binding contract")
        if release is not None and not callable(release):
            raise ValueError("release must be an idempotent callable")
        report = adapter.capabilities()
        with self._lock:
            if self._closed:
                raise RuntimeError("Host is closed")
            self._require_idle()
            if model_id in self._models:
                raise ValueError("Model ID is already registered")
            if len(self._models) >= self._max_models:
                raise RuntimeError("Host model registration limit reached")
            resource = getattr(adapter, "model", getattr(adapter, "pipeline", adapter))
            if any(item["resource"] is resource for item in self._models.values()):
                raise ValueError("The same loaded resource is already registered")
            self._models[model_id] = {"adapter": adapter, "resource": resource, "release": release,
                "binding_id": uuid4().hex, "state": "ready", "error": None, "capabilities": deepcopy(report)}

    def models(self):
        with self._lock:
            return [{"model_id": key, "binding_id": item["binding_id"], "state": item["state"], "error": item["error"],
                     "capabilities": deepcopy(item["capabilities"])} for key, item in self._models.items()]

    def submit(self, model_id, operation, *, inputs, module_paths=(), options=None, timeout_seconds=None, on_result=None):
        if on_result is not None and not callable(on_result):
            raise ValueError("on_result must be callable")
        if not isinstance(inputs, dict) or options is not None and not isinstance(options, dict):
            raise ValueError("inputs and options must be dictionaries")
        options = dict(options or {})
        if set(options) & {"control", "inputs", "input_ids", "module_paths"}:
            raise ValueError("Options cannot override host-owned request fields")
        if not isinstance(module_paths, (list, tuple)):
            raise ValueError("module_paths must be a list or tuple")
        paths = list(module_paths)
        methods = {"module_outputs": "capture", "next_token": "next_token",
                   "probe_attention": "probe_attention", "ablate_heads": "ablate_heads", "patch_layers": "patch_layers", "denoising_steps": "capture_steps"}
        if operation not in methods:
            raise UnsupportedCapability("Unknown observation operation")
        with self._lock:
            if self._closed:
                raise RuntimeError("Host is closed")
            self._require_idle()
            item = self._models[model_id]
            if item["state"] != "ready":
                raise RuntimeError("Model resource is not ready")
            adapter = item["adapter"]
            feature = "module_outputs" if operation in ("probe_attention", "ablate_heads", "patch_layers") else operation
            negotiation = adapter.negotiate([feature])
            if not negotiation["accepted"] or not callable(getattr(adapter, methods[operation], None)):
                raise UnsupportedCapability(f"Operation unavailable: {operation}")
            if operation in ("next_token", "probe_attention", "ablate_heads") and set(inputs) != {"input_ids"}:
                raise ValueError("Language-model input requires exactly input_ids")
            if operation == "patch_layers" and set(inputs) != {"input_ids", "source_ids"}:
                raise ValueError("Activation patching requires input_ids and source_ids")
            def observe(control):
                if operation == "module_outputs":
                    output, trace = adapter.capture(inputs, paths, control=control, **options)
                elif operation == "next_token":
                    output, trace = adapter.next_token(inputs["input_ids"], paths, control=control, **options)
                elif operation == "probe_attention":
                    trace = adapter.probe_attention(inputs["input_ids"], control=control, **options)
                    output = None
                elif operation == "ablate_heads":
                    trace = adapter.ablate_heads(inputs["input_ids"], control=control, **options)
                    output = None
                elif operation == "patch_layers":
                    trace = adapter.patch_layers(inputs["input_ids"], inputs["source_ids"], control=control, **options)
                    output = None
                else:
                    output, trace = adapter.capture_steps(inputs, control=control, **options)
                if on_result is not None:
                    on_result(output, trace)
                trace["host_model_id"] = model_id
                trace["host_binding_id"] = item["binding_id"]
                return trace
            run_id = self._session.start(observe, timeout_seconds=timeout_seconds)
            self._selection = {"model_id": model_id, "binding_id": item["binding_id"], "operation": operation}
            return run_id

    def snapshot(self):
        with self._lock:
            return {"closed": self._closed, "resource_operation": self._maintenance,
                    "selection": deepcopy(self._selection), "run": self._session.snapshot()}

    def cancel(self):
        return self._session.cancel()

    def wait(self, timeout=None):
        self._session.wait(timeout)
        return self.snapshot()

    def take_trace(self):
        return self._session.take_trace()

    def unregister(self, model_id):
        return self._unregister(model_id)

    def _unregister(self, model_id, *, closing=False):
        with self._lock:
            if self._closing and not closing:
                raise RuntimeError("Host shutdown is in progress")
            self._require_idle()
            item = self._models[model_id]
            item["state"] = "releasing"
            self._maintenance = True
        try:
            if item["release"] is not None:
                item["release"]()
        except Exception as exc:
            with self._lock:
                item.update(state="release_failed", error=f"{type(exc).__name__}: {exc}"[:2048])
            raise
        else:
            with self._lock:
                del self._models[model_id]
        finally:
            with self._lock:
                self._maintenance = False

    def close(self):
        # Shutdown waits for cooperative cleanup before releasing any model resources.
        with self._lock:
            if self._maintenance or self._closing:
                raise RuntimeError("A resource operation is still active")
            self._closed = True
            self._closing = True
        try:
            self._session.close(wait=True)
            errors = []
            for model_id in [item["model_id"] for item in self.models()]:
                try:
                    self._unregister(model_id, closing=True)
                except Exception as exc:
                    errors.append({"model_id": model_id, "error": str(exc)[:2048]})
            return errors
        finally:
            with self._lock:
                self._closing = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        errors = self.close()
        if errors:
            if exc is not None:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"Model resource release also failed: {errors}")
                return False
            raise RuntimeError(f"Model resource release failed: {errors}")
