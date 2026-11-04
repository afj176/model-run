"""Host-independent local application. All resource mutations share one work queue."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import RLock
import gc
import json
import time
from uuid import uuid4

from .adapters import CausalLMAdapter, DiffusersPipelineAdapter
from .catalog import discover_models
from .host import ModelHost
from .inventory import inspect_file, inspect_index
from .trace_store import encode_trace


def bounded_int(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


class Application:
    def __init__(self, entries=(), *, host_kind="standalone"):
        self.entries = list(entries)
        self.host_kind = host_kind
        self.host = ModelHost(max_models=1)
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-run-app")
        self._state = {"status": "idle", "action": None, "error": None, "job_id": None}
        self._closed = False
        self._cancelled = False
        self._catalog = {"artifacts": [], "errors": []}
        self._paths = {}
        self._loaded = None
        self._tree = None
        self._traces = []
        self._trace_bytes = 0

    def _start(self, action, function):
        with self._lock:
            if self._closed:
                raise RuntimeError("Application is closed")
            if self._state["status"] == "running":
                raise RuntimeError("Wait for the current operation or cancel it")
            self._cancelled = False
            job_id = uuid4().hex
            self._state = {"status": "running", "action": action, "error": None, "job_id": job_id}
            def execute():
                started = time.perf_counter()
                try:
                    function()
                    with self._lock:
                        self._state["status"] = "cancelled" if self._cancelled else "completed"
                except BaseException as exc:
                    # Keep the worker lifecycle recoverable even if a provider exits.
                    with self._lock:
                        self._state.update(status="cancelled" if self._cancelled else "failed", error=f"{type(exc).__name__}: {exc}"[:2048])
                finally:
                    with self._lock:
                        self._state["elapsed_seconds"] = time.perf_counter() - started
            self._future = self._executor.submit(execute)
            return {"job_id": job_id}

    def snapshot(self):
        with self._lock:
            loaded = None if self._loaded is None else {k: self._loaded[k] for k in ("id", "name", "task", "device")}
            return {"host": self.host_kind, "job": deepcopy(self._state), "loaded": loaded,
                    "observation": self.host.snapshot(), "models": self.host.models(),
                    "traces": [{k: t[k] for k in ("id", "name", "kind", "size_bytes", "revision")} for t in self._traces]}

    def catalog(self):
        with self._lock:
            return deepcopy(self._catalog)

    def scan(self):
        def discover():
            result = discover_models(self.entries)
            paths = {}
            for artifact in result["artifacts"]:
                path = Path(artifact.pop("path")).resolve()
                identifier = sha256(str(path).encode()).hexdigest()[:24]
                paths[identifier] = path
                artifact.update(id=identifier, name=path.name, relative_path=self._relative(path), task="inventory")
                # Root paths are intentionally not exposed by the browser API.
                artifact["origins"] = [{"category": o["category"]} for o in artifact["origins"]]
                artifact.pop("shard_paths", None)
            # Configs identify local package candidates, not proven executable architectures.
            dirs = sorted({p.parent for p in paths.values()}, key=str)
            for directory in dirs:
                if any((parent / "model_index.json").is_file() for parent in directory.parents if parent in dirs):
                    continue  # Pipeline components are not standalone causal models.
                if (directory / "model_index.json").is_file():
                    task = "diffusion"
                elif (directory / "config.json").is_file() and any(directory.glob("*.safetensors")):
                    config_path = directory / "config.json"
                    if config_path.stat().st_size > 1024 * 1024:
                        continue
                    try:
                        config = json.loads(config_path.read_text(encoding="utf-8"))
                    except (ValueError, UnicodeError):
                        continue
                    architectures = config.get("architectures", []) if isinstance(config, dict) else []
                    task = "text" if isinstance(architectures, list) and any(isinstance(a, str) and a.endswith("ForCausalLM") for a in architectures) else "inventory"
                else:
                    continue
                identifier = sha256(str(directory).encode()).hexdigest()[:24]
                paths[identifier] = directory
                result["artifacts"].insert(0, {"id": identifier, "name": directory.name,
                    "relative_path": self._relative(directory), "format": "local_package", "status": "candidate",
                    "task": task, "configured_categories": [], "capabilities": {},
                    "limitations": ["Local configuration candidate; load verifies runtime compatibility."]})
            with self._lock:
                self._paths, self._catalog = paths, result
        return self._start("scan", discover)

    def _relative(self, path):
        for entry in self.entries:
            root = Path(entry["path"]).resolve()
            if path.is_relative_to(root):
                return f'{entry["category"]}/{path.relative_to(root).as_posix()}'
        return path.name

    def _path(self, identifier):
        with self._lock:
            path = self._paths.get(identifier)
        if path is None:
            raise ValueError("Unknown catalog item; refresh the catalog")
        resolved = path.resolve(strict=True)
        if not any(resolved.is_relative_to(Path(e["path"]).resolve()) for e in self.entries):
            raise ValueError("Artifact is outside configured model roots")
        return resolved

    def inspect(self, identifier):
        path = self._path(identifier)
        if path.is_dir():
            config = path / ("model_index.json" if (path / "model_index.json").is_file() else "config.json")
            if config.stat().st_size > 1024 * 1024:
                raise ValueError("Configuration exceeds 1 MiB")
            return {"kind": "model_configuration", "config": json.loads(config.read_text(encoding="utf-8")),
                    "limitations": ["Configuration is declared metadata, not measured runtime structure."]}
        if path.suffix == ".safetensors":
            result = inspect_file(path)
        elif path.name.endswith(".safetensors.index.json"):
            result = inspect_index(path)
        elif path.suffix.lower() == ".onnx":
            from .onnx_graph import inspect_onnx
            result = inspect_onnx(path)
        else:
            return {"kind": "unsupported_artifact", "limitations": ["This format has no installed metadata reader. It has not been deserialized."]}
        result.pop("path", None)
        return result

    def structure(self):
        with self._lock:
            return deepcopy(self._tree)

    def artifact_structure(self, identifier):
        path = self._path(identifier)
        if path.is_dir():
            raise ValueError("Select a safetensors artifact or index to explore stored tensor structure")
        report = self.inspect(identifier)
        if report.get("format") not in ("safetensors", "safetensors_index"):
            raise ValueError("Spatial artifact browsing currently supports safetensors and shard indexes")
        from .artifact_structure import tensor_namespace
        return tensor_namespace(report, name=path.name)

    def _release(self):
        if self._loaded is not None:
            self.host.unregister(self._loaded["id"])
            with self._lock:
                self._loaded, self._tree = None, None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, identifier, device="cpu"):
        if self.host_kind != "standalone":
            raise ValueError("ComfyUI owns model loading. Connect the Model Run Observe node to a loaded MODEL.")
        if device not in ("cpu", "cuda"):
            raise ValueError("Device must be cpu or cuda")
        path = self._path(identifier)
        if not path.is_dir():
            raise ValueError("Standalone loading requires a complete local Transformers or Diffusers directory")
        def load():
            self._release()
            if (path / "model_index.json").is_file():
                import torch
                from diffusers import DiffusionPipeline
                pipeline = DiffusionPipeline.from_pretrained(str(path), local_files_only=True, use_safetensors=True,
                    trust_remote_code=False, torch_dtype=torch.float32)
                pipeline.to(device)
                adapter, tokenizer, task = DiffusersPipelineAdapter(pipeline), None, "diffusion"
                from .structure import inspect_module_tree
                core = getattr(pipeline, "transformer", None) or getattr(pipeline, "unet", None)
                tree = inspect_module_tree(core) if core is not None else {"nodes": [], "limitations": ["Pipeline has no exposed torch denoiser"]}
                release = lambda: pipeline.to("cpu")
            else:
                from .local_text import load_local_text_model
                model, tokenizer = load_local_text_model(path, device=device, attention_implementation="eager")
                adapter, task = CausalLMAdapter(model), "text"
                tree, release = adapter.structure(), lambda: model.to("cpu")
            with self._lock:
                cancelled = self._cancelled
            if cancelled:
                release()
                return
            self.host.register(identifier, adapter, release=release)
            with self._lock:
                self._loaded = {"id": identifier, "name": path.name, "device": device, "task": task,
                                "adapter": adapter, "tokenizer": tokenizer}
                self._tree = tree
        return self._start("load", load)

    def unload(self):
        return self._start("unload", self._release)

    def run(self, request):
        if not isinstance(request, dict):
            raise ValueError("Expected a run request")
        prompt = request.get("prompt", "")
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= 8192:
            raise ValueError("Prompt must contain 1 to 8192 characters")
        paths = request.get("modules", [])
        if not isinstance(paths, list) or len(paths) > 32 or any(not isinstance(p, str) or len(p) > 512 for p in paths):
            raise ValueError("Select at most 32 module paths")
        paths = list(paths)
        elements = bounded_int(request.get("elements", 1024), 1, 4096, "elements")
        ablate = request.get("ablate")
        if ablate not in (None, "head", "layer"):
            raise ValueError("ablate must be head or layer")
        attach_to = request.get("attach_to")
        if ablate and not isinstance(attach_to, str):
            raise ValueError("Capture attention for this prompt before ablating heads")
        def observe():
            item = self._loaded
            if item is None:
                raise ValueError("Load a model first")
            import torch
            if item["task"] == "text":
                tokenizer = item["tokenizer"]
                mode = request.get("input_mode", "raw")
                if mode not in ("raw", "chat"):
                    raise ValueError("Input mode must be raw or chat")
                if mode == "chat":
                    prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                else:
                    prompt_text = prompt
                ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=mode == "raw")["input_ids"]
                if not 1 <= ids.shape[1] <= 128:
                    raise ValueError("Use a prompt of 1 to 128 tokens for bounded observation")
                inputs = {"input_ids": ids.to(item["device"])}
                attention = request.get("attention", False)
                if type(attention) is not bool:
                    raise ValueError("attention must be boolean")
                if ablate:
                    operation = "ablate_heads"
                    options = {"layer": bounded_int(request.get("layer", 0), 0, 1024, "layer"),
                               "heads": [bounded_int(request.get("head", 0), 0, 1024, "head")] if ablate == "head" else None}
                else:
                    operation = "probe_attention" if attention else "next_token"
                    options = {"layer": bounded_int(request.get("layer", 0), 0, 1024, "layer"),
                               "head": bounded_int(request.get("head", 0), 0, 1024, "head")} if attention else {"max_tensor_elements": elements}
                if not paths and not attention and not ablate:
                    paths.extend([n["path"] for n in self._tree["nodes"] if n["path"].split(".")[-1].isdigit()][:32] or [""])
                def decorate(output, trace):
                    trace["input_mode"] = mode
                    trace["tokens"] = [{"id": i, "text": tokenizer.decode([i])} for i in ids[0].tolist()]
                    for token in trace.get("output", {}).get("top_tokens", []):
                        token["text"] = tokenizer.decode([token["token_id"]])
                    for entry in trace.get("logit_lens", {}).get("layers", []):
                        for token in entry["top_tokens"]:
                            token["text"] = tokenizer.decode([token["token_id"]])
                    if trace.get("kind") == "head_ablation":
                        for entry in [trace["baseline"], *trace["heads"]]:
                            for token in entry["top_tokens"]:
                                token["text"] = tokenizer.decode([token["token_id"]])
                    trace["limitations"].append("This application trace includes decoded input tokens; exported traces may contain private text.")
            else:
                import inspect
                parameters = inspect.signature(type(item["adapter"].pipeline).__call__).parameters
                if "prompt" not in parameters or "output_type" not in parameters:
                    raise ValueError("This pipeline needs a task-specific input adapter; use ComfyUI to observe it")
                steps = bounded_int(request.get("steps", 8), 1, 32, "steps")
                size = bounded_int(request.get("size", 256), 64, 512, "size")
                if size % 8:
                    raise ValueError("Image size must be divisible by 8")
                seed = bounded_int(request.get("seed", 0), 0, 2147483647, "seed")
                inputs = {"prompt": prompt, "num_inference_steps": steps, "output_type": "pil",
                          "generator": torch.Generator(device=item["device"]).manual_seed(seed)}
                for key in ("height", "width"):
                    if key in parameters:
                        inputs[key] = size
                operation, options = "denoising_steps", {"max_tensor_elements": elements, "max_events": 32}
                cross = request.get("cross_attention")
                if cross is not None:
                    if not isinstance(cross, str) or not 1 <= len(cross) <= 512:
                        raise ValueError("cross_attention must name one attention module")
                    options["cross_attention"] = cross
                    options["attention_grid"] = bounded_int(request.get("attention_grid", 32), 4, 64, "attention_grid")
                    tokenizer = getattr(item["adapter"].pipeline, "tokenizer", None)
                    if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
                        try:
                            encoded = tokenizer(prompt, padding="max_length", truncation=True, max_length=getattr(tokenizer, "model_max_length", 77))
                            options["token_labels"] = [str(t).replace("</w>", "") for t in tokenizer.convert_ids_to_tokens(encoded["input_ids"])][:1024]
                        except Exception:
                            options["token_labels"] = None
                def decorate(output, trace):
                    import base64
                    import io
                    images = getattr(output, "images", None)
                    if images and hasattr(images[0], "save"):
                        buffer = io.BytesIO()
                        images[0].save(buffer, format="PNG")
                        if buffer.tell() <= 4 * 1024 * 1024:
                            trace["preview"] = {"mime": "image/png", "base64": base64.b64encode(buffer.getvalue()).decode()}
                    trace["sampling"] = {"seed": seed, "steps": steps, "size": size}
            with self._lock:
                if self._cancelled:
                    return
                self.host.submit(item["id"], operation, inputs=inputs, module_paths=paths, options=options,
                                 timeout_seconds=300, on_result=decorate)
            state = self.host.wait()["run"]
            trace = self.host.take_trace()
            if trace is not None and trace.get("kind") == "head_ablation":
                self._attach_ablation(attach_to, trace)
            elif trace is not None:
                trace["model_name"] = item["name"]
                trace["structure"] = self._tree
                self.publish(trace, item["name"])
            if state["status"] == "failed":
                raise RuntimeError(state["error"])
        return self._start("observe", observe)

    def _attach_ablation(self, attach_to, result):
        """Merge an ablation result into the attention trace it was measured against."""
        target = self.trace(attach_to)
        if target.get("kind") != "attention_capture" or target.get("layer") != result["layer"]:
            raise ValueError("Ablation must target the attention capture of the same layer")
        if target.get("input", {}).get("token_count") != result["position"] + 1:
            raise ValueError("The prompt differs from the attention capture; capture attention again first")
        if result["head_count"] != target.get("head_count", result["head_count"]):
            raise ValueError("The attention capture reports a different head count")
        existing = target.get("ablation")
        heads = {entry["head"]: entry for entry in (existing or {}).get("heads", [])}
        for entry in result["heads"]:
            heads[entry["head"]] = entry
        target["ablation"] = {key: result[key] for key in ("method", "layer", "head_count", "head_dim", "position", "baseline", "limitations")}
        target["ablation"]["heads"] = [heads[key] for key in sorted(heads)]
        with self._lock:
            record = next((t for t in self._traces if t["id"] == attach_to), None)
            name = record["name"] if record else "Attention capture"
        self.publish(target, name, replace_id=attach_to)

    def publish(self, trace, name="Imported trace", *, replace_id=None):
        data = encode_trace(trace)
        value = json.loads(data)
        record = {"id": uuid4().hex, "name": str(name)[:160], "kind": trace["kind"], "size_bytes": len(data), "trace": value, "revision": 1}
        with self._lock:
            previous = next((t for t in self._traces if t["id"] == replace_id), None)
            if previous is not None:
                self._traces.remove(previous)
                self._trace_bytes -= previous["size_bytes"]
                record["id"] = previous["id"]
                record["revision"] = previous["revision"] + 1
            while self._traces and (len(self._traces) >= 8 or self._trace_bytes + len(data) > 32 * 1024 * 1024):
                self._trace_bytes -= self._traces.pop(0)["size_bytes"]
            self._traces.append(record)
            self._trace_bytes += len(data)
        return {"id": record["id"]}

    def trace(self, identifier):
        with self._lock:
            for record in self._traces:
                if record["id"] == identifier:
                    return deepcopy(record["trace"])
        raise ValueError("Trace is no longer retained")

    def delete_trace(self, identifier):
        with self._lock:
            for record in list(self._traces):
                if record["id"] == identifier:
                    self._traces.remove(record)
                    self._trace_bytes -= record["size_bytes"]
                    return {"deleted": True}
        raise ValueError("Unknown trace")

    def cancel(self):
        with self._lock:
            self._cancelled = True
            self.host.cancel()
        return {"requested": True}

    def close(self):
        with self._lock:
            self._closed = True
        self.cancel()
        self._executor.shutdown(wait=True)
        return self.host.close()
