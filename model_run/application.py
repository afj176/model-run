"""Host-independent local application. All resource mutations share one work queue."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import RLock
import re
import gc
import json
import time
from uuid import uuid4

from .adapters import CausalLMAdapter, DiffusersPipelineAdapter
from .catalog import discover_models
from .host import ModelHost
from .embedding_models import EmbeddingAdapter, is_embedding_package, load_local_embedding_model
from .inventory import inspect_file, inspect_index
from .model_install import ModelInstaller
from .media_models import (ImageClassifierAdapter, SpeechAdapter, declared_name, decode_audio, decode_image, encode_wav,
                           load_local_media_model, read_config, task_for_config)
from .trace_store import encode_trace

MAX_RUNS = 16
MAX_TRACES = 64
MAX_RETAINED_BYTES = 256 * 1024 * 1024
MAX_PINNED_RUNS = 8
MAX_PINNED_BYTES = 128 * 1024 * 1024


def bounded_int(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


class Application:
    def __init__(self, entries=(), *, host_kind="standalone", install_folders=None, comfy_dir=None):
        self.entries = list(entries)
        folders = install_folders if callable(install_folders) else (lambda: list(install_folders if install_folders is not None else self.entries))
        self.installer = ModelInstaller(folders, comfy_dir=comfy_dir)
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
        self._native_workflows = {}  # in memory only; workflows never enter a trace or an export
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
        if self.installer.take_rescan():
            try:
                self.scan()
            except RuntimeError:
                self.installer.request_rescan()  # another job is running; pick it up on a later poll
        with self._lock:
            loaded = None if self._loaded is None else {k: self._loaded.get(k) for k in ("id", "name", "task", "device", "dtype")}
            if loaded is not None:
                loaded["chat_template"] = bool(getattr(self._loaded.get("tokenizer"), "chat_template", None))
            return {"host": self.host_kind, "job": deepcopy(self._state), "loaded": loaded,
                    "observation": self.host.snapshot(), "models": self.host.models(), "install": self.installer.status(),
                    "traces": [{k: t[k] for k in ("id", "name", "kind", "size_bytes", "revision", "run", "step", "pinned", "note")} for t in self._traces],
                    "retention": {"runs": len({t["run"] for t in self._traces}), "max_runs": MAX_RUNS,
                                  "bytes": self._trace_bytes, "max_bytes": MAX_RETAINED_BYTES,
                                  "pinned_runs": len({t["run"] for t in self._traces if t["pinned"]}), "max_pinned_runs": MAX_PINNED_RUNS}}

    def catalog(self):
        with self._lock:
            return deepcopy(self._catalog)

    def installed_names(self):
        with self._lock:
            return {(origin["category"], artifact["name"]) for artifact in self._catalog["artifacts"] for origin in artifact.get("origins", [])
                    if isinstance(origin, dict) and "category" in origin}

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
                if any(is_embedding_package(parent) for parent in directory.parents if parent in dirs):
                    continue  # An embedding package's dense layers are steps of that package.
                display = directory.name
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
                    task = task_for_config(config)
                    if task == "inventory" and is_embedding_package(directory):
                        task = "embedding"
                    display = declared_name(config, directory.name) if task in ("image", "audio") else directory.name
                else:
                    continue
                identifier = sha256(str(directory).encode()).hexdigest()[:24]
                paths[identifier] = directory
                result["artifacts"].insert(0, {"id": identifier, "name": display,
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
            display = path.name
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
                release = lambda: discard(pipeline)
            elif is_embedding_package(path) and task_for_config(read_config(path)) == "inventory":
                model, tokenizer, head = load_local_embedding_model(path, device=device)
                adapter, task = EmbeddingAdapter(model, head), "embedding"
                tree, release = adapter.structure(), lambda: discard(model)
            elif task_for_config(read_config(path)) in ("image", "audio"):
                config = read_config(path)
                task = task_for_config(config)
                model, processor = load_local_media_model(path, task, device=device)
                adapter = ImageClassifierAdapter(model) if task == "image" else SpeechAdapter(model)
                tokenizer = processor
                tree, release = adapter.structure(), lambda: discard(model)
                display = declared_name(config, path.name)
            else:
                from .local_text import load_local_text_model
                model, tokenizer = load_local_text_model(path, device=device, attention_implementation="eager")
                adapter, task = CausalLMAdapter(model), "text"
                tree, release = adapter.structure(), lambda: discard(model)
                if decoder_prefix(model):
                    tree["primary_prefix"] = decoder_prefix(model)  # the stack a text prompt runs, ahead of image or audio towers
            with self._lock:
                cancelled = self._cancelled
            if cancelled:
                release()
                return
            self.host.register(identifier, adapter, release=release)
            with self._lock:
                self._loaded = {"id": identifier, "name": display, "device": device, "task": task, "dtype": loaded_dtype(adapter),
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
        run_group, run_step = request.get("run_group"), request.get("run_step")
        if run_group is not None and (not isinstance(run_group, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_group)):
            raise ValueError("Invalid run group")
        if run_step not in (None, "module", "generation", "attention"):
            raise ValueError("Invalid run step")
        attach_to = request.get("attach_to")
        if ablate and not isinstance(attach_to, str):
            raise ValueError("Capture attention for this prompt before ablating heads")
        generate = request.get("generate")
        if generate is not None:
            generate = bounded_int(generate, 1, 256, "generate")
        patching = request.get("patch")
        if patching not in (None, "last", "all"):
            raise ValueError("patch must be last or all")
        source_prompt = request.get("source_prompt")
        if patching and (not isinstance(source_prompt, str) or not 1 <= len(source_prompt) <= 8192 or not isinstance(attach_to, str)):
            raise ValueError("Activation patching needs a source prompt and a target trace to attach to")
        if patching and ablate:
            raise ValueError("Choose either ablation or patching")
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
                    if not getattr(tokenizer, "chat_template", None):
                        raise ValueError("This tokenizer has no chat template; use the raw prompt mode")
                    prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                else:
                    prompt_text = prompt
                ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=mode == "raw")["input_ids"]
                if mode == "raw":
                    ids = with_start_token(tokenizer, ids)
                if not 1 <= ids.shape[1] <= 128:
                    raise ValueError("Use a prompt of 1 to 128 tokens for bounded observation")
                inputs = {"input_ids": ids.to(item["device"])}
                if request.get("ablate_tokens") is not None or request.get("cross_attention") is not None:
                    raise ValueError("Token ablation and cross-attention maps apply to diffusion pipelines; load one first")
                if patching:
                    source_text = tokenizer.apply_chat_template([{"role": "user", "content": source_prompt}], tokenize=False, add_generation_prompt=True) if mode == "chat" else source_prompt
                    source_ids = tokenizer(source_text, return_tensors="pt", add_special_tokens=mode == "raw")["input_ids"]
                    if mode == "raw":
                        source_ids = with_start_token(tokenizer, source_ids)
                    if not 1 <= source_ids.shape[1] <= 128:
                        raise ValueError("Use a source prompt of 1 to 128 tokens")
                    inputs["source_ids"] = source_ids.to(item["device"])
                attention = request.get("attention", False)
                if type(attention) is not bool:
                    raise ValueError("attention must be boolean")
                if generate is not None:
                    operation = "generate"
                    options = {"max_new_tokens": generate, "top_k": 5}
                elif patching:
                    operation = "patch_layers"
                    options = {"positions": patching}
                elif ablate:
                    operation = "ablate_heads"
                    options = {"layer": bounded_int(request.get("layer", 0), 0, 1024, "layer"),
                               "heads": [bounded_int(request.get("head", 0), 0, 1024, "head")] if ablate == "head" else None}
                else:
                    operation = "probe_attention" if attention else "next_token"
                    options = {"layer": bounded_int(request.get("layer", 0), 0, 1024, "layer"),
                               "head": bounded_int(request.get("head", 0), 0, 1024, "head")} if attention else {"max_tensor_elements": elements}
                if not paths and not attention and not ablate and not patching and generate is None:
                    paths.extend(text_layer_paths(item["adapter"].model, self._tree) or [""])
                def decorate(output, trace):
                    trace["input_mode"] = mode
                    trace["tokens"] = [{"id": i, "text": tokenizer.decode([i])} for i in ids[0].tolist()]
                    for token in trace.get("output", {}).get("top_tokens", []):
                        token["text"] = tokenizer.decode([token["token_id"]])
                    for token in trace.get("logits_view", {}).get("top", []):
                        token["text"] = tokenizer.decode([token["token_id"]])
                    for entry in trace.get("logit_lens", {}).get("layers", []):
                        for token in entry["top_tokens"]:
                            token["text"] = tokenizer.decode([token["token_id"]])
                    if trace.get("kind") == "head_ablation":
                        for entry in [trace["baseline"], *trace["heads"]]:
                            for token in entry["top_tokens"]:
                                token["text"] = tokenizer.decode([token["token_id"]])
                    if trace.get("kind") == "generation_capture":
                        for step in trace["steps"]:
                            step["text"] = tokenizer.decode([step["token_id"]])
                            for token in step["top_tokens"]:
                                token["text"] = tokenizer.decode([token["token_id"]])
                        trace["response"] = tokenizer.decode([s["token_id"] for s in trace["steps"]], skip_special_tokens=False)
                    if trace.get("kind") == "activation_patching":
                        for entry in [trace["source_run"], trace["target_run"], *trace["layers"]]:
                            for token in entry["top_tokens"]:
                                token["text"] = tokenizer.decode([token["token_id"]])
                        trace["source_run"]["tokens"] = [{"id": i, "text": tokenizer.decode([i])} for i in inputs["source_ids"][0].tolist()]
                    trace["limitations"].append("This application trace includes decoded input tokens; exported traces may contain private text.")
                    if item.get("dtype") not in (None, "float32"):
                        trace["limitations"].append(f"The model ran in {item['dtype']}, its checkpoint's own precision, because it would not fit the GPU in float32; values differ slightly from a float32 run.")
            elif item["task"] == "embedding":
                if ablate or patching or generate is not None or request.get("attention") or request.get("ablate_tokens") is not None or request.get("cross_attention") is not None:
                    raise ValueError("An embedding model has no next token, response or attention probe; running it records its layers and the sentence vector")
                tokenizer = item["tokenizer"]
                encoded = tokenizer(prompt, return_tensors="pt")
                if not 1 <= encoded["input_ids"].shape[1] <= 128:
                    raise ValueError("Use a prompt of 1 to 128 tokens for bounded observation")
                if not paths:
                    paths.extend([n["path"] for n in self._tree["nodes"] if n["path"].split(".")[-1].isdigit()][:32] or [""])
                inputs = {"input_ids": encoded["input_ids"].to(item["device"]), "attention_mask": encoded["attention_mask"].to(item["device"])}
                operation, options = "embed", {"max_tensor_elements": elements}
                def decorate(output, trace):
                    trace["tokens"] = [{"id": i, "text": tokenizer.decode([i])} for i in encoded["input_ids"][0].tolist()]
                    trace["limitations"].append("This application trace includes decoded input tokens; exported traces may contain private text.")
                    if item.get("dtype") not in (None, "float32"):
                        trace["limitations"].append(f"The model ran in {item['dtype']}, its checkpoint's own precision, because it would not fit the GPU in float32; values differ slightly from a float32 run.")
            elif item["task"] in ("image", "audio"):
                if ablate or patching or generate is not None or request.get("attention") or request.get("ablate_tokens") is not None or request.get("cross_attention") is not None:
                    raise ValueError("Ablation, patching and attention probes apply to text models; running this model records its layers and output")
                processor = item["tokenizer"]
                if not paths:
                    paths.extend([n["path"] for n in self._tree["nodes"] if n["path"].split(".")[-1].isdigit()][:32] or [""])
                if item["task"] == "image":
                    image = decode_image(request.get("media"))
                    pixels = processor(images=image, return_tensors="pt")["pixel_values"]
                    inputs = {"pixel_values": pixels.to(device=item["device"], dtype=torch.float32)}
                    operation, options = "classify", {"max_tensor_elements": elements, "top_k": 5}
                    def decorate(output, trace):
                        import base64
                        import io
                        # What the model received: its processor's size, not the upload's.
                        buffer = io.BytesIO()
                        image.resize((int(pixels.shape[-1]), int(pixels.shape[-2]))).save(buffer, format="PNG")
                        trace["image_classification"]["input_image"] = {"mime": "image/png", "base64": base64.b64encode(buffer.getvalue()).decode()}
                        trace["input_media"] = {"kind": "image", "name": prompt[:160], "source_width": image.width, "source_height": image.height}
                        trace["limitations"].append("This application trace includes the input image; exported traces may contain private images.")
                else:
                    samples, audio = decode_audio(request.get("media"))
                    features = processor(samples, sampling_rate=audio["rate"], return_tensors="pt")["input_features"]
                    inputs = {"input_features": features.to(device=item["device"], dtype=torch.float32)}
                    extractor = processor.feature_extractor
                    operation, options = "transcribe", {"max_tensor_elements": elements, "top_k": 5,
                        "audio_info": {"used_seconds": audio["used_seconds"], "hop_length": int(extractor.hop_length), "rate": audio["rate"]}}
                    def decorate(output, trace):
                        text = processor.tokenizer
                        transcript = trace["transcript_generation"]
                        transcript["tokens"] = [{"id": i, "text": text.decode([i])} for i in transcript.pop("prompt_token_ids")]
                        for step in transcript["steps"]:
                            step["text"] = text.decode([step["token_id"]])
                            for token in step["top_tokens"]:
                                token["text"] = text.decode([token["token_id"]])
                        transcript["response"] = text.decode([s["token_id"] for s in transcript["steps"]], skip_special_tokens=True)
                        trace["input_media"] = {**audio, "name": prompt[:160]}
                        view = trace["audio_view"]
                        for token in view["tokens"]:
                            token["text"] = text.decode([token["token_id"]])
                        view["audio"] = encode_wav(samples, audio["rate"])
                        trace["limitations"].append("This application trace includes the input audio; exported traces may contain private speech.")
            else:
                if ablate:
                    raise ValueError("Head ablation needs a loaded text model")
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
                ablate_tokens = request.get("ablate_tokens")
                if ablate_tokens is not None:
                    from .cross_attention import parse_token_list
                    options["ablate_tokens"] = parse_token_list(ablate_tokens)
                    if not options["ablate_tokens"]:
                        raise ValueError("Select at least one token to ablate")
                cross = request.get("cross_attention")
                if cross is not None or ablate_tokens is not None:
                    tokenizer = getattr(item["adapter"].pipeline, "tokenizer", None)
                    if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
                        try:
                            encoded = tokenizer(prompt, padding="max_length", truncation=True, max_length=getattr(tokenizer, "model_max_length", 77))
                            options["token_labels"] = [str(t).replace("</w>", "") for t in tokenizer.convert_ids_to_tokens(encoded["input_ids"])][:1024]
                        except Exception:
                            options["token_labels"] = None
                if cross is not None:
                    if not isinstance(cross, str) or not 1 <= len(cross) <= 512:
                        raise ValueError("cross_attention must name one attention module")
                    options["cross_attention"] = cross
                    options["attention_grid"] = bounded_int(request.get("attention_grid", 32), 4, 64, "attention_grid")
                def decorate(output, trace):
                    import base64
                    import io
                    images = getattr(output, "images", None)
                    if images and hasattr(images[0], "save"):
                        buffer = io.BytesIO()
                        images[0].save(buffer, format="PNG")
                        if buffer.tell() <= 4 * 1024 * 1024:
                            trace["preview"] = {"mime": "image/png", "base64": base64.b64encode(buffer.getvalue()).decode()}
                    trace["sampling"] = {"seed": seed, "steps": steps, "size": size, "prompt": prompt}
                    trace["limitations"].append("This application trace records the prompt text so a run can be repeated; exported traces may contain private text.")
            with self._lock:
                if self._cancelled:
                    return
                self.host.submit(item["id"], operation, inputs=inputs, module_paths=paths, options=options,
                                 timeout_seconds=300, on_result=decorate)
            state = self.host.wait()["run"]
            trace = self.host.take_trace()
            if trace is not None and trace.get("kind") == "head_ablation":
                self._attach_ablation(attach_to, trace)
            elif trace is not None and trace.get("kind") == "activation_patching":
                self._attach_patching(attach_to, trace)
            elif trace is not None:
                trace["model_name"] = item["name"]
                trace["structure"] = self._tree
                transcript = trace.pop("transcript_generation", None)
                if transcript is not None:
                    # A transcript and its layer outputs are one run: published as its two parts.
                    group = run_group or uuid4().hex
                    transcript["model_name"] = item["name"]
                    transcript["input_media"] = trace.get("input_media")
                    self.publish(transcript, item["name"], run=group, step="generation", note=prompt)
                    self.publish(trace, item["name"], run=group, step="module", note=prompt)
                else:
                    self.publish(trace, item["name"], run=run_group, step=run_step, note=prompt)
            if state["status"] == "failed":
                raise RuntimeError(state["error"])
        return self._start("observe", observe)

    def remember_workflow(self, trace_id, graph, node):
        with self._lock:
            live = {t["id"] for t in self._traces}
            self._native_workflows = {k: v for k, v in self._native_workflows.items() if k in live}
            # Only what ComfyUI needs to queue the graph; the executor adds bookkeeping such as is_changed: NaN.
            clean = {key: {"class_type": value["class_type"], "inputs": deepcopy(value["inputs"])}
                     for key, value in graph.items() if isinstance(value, dict) and "class_type" in value and "inputs" in value}
            self._native_workflows[trace_id] = {"prompt": clean, "node": str(node)}

    def native_workflow(self, identifier):
        """The ComfyUI graph that produced a retained trace, so it can be queued again."""
        with self._lock:
            workflow = self._native_workflows.get(identifier)
            retained = any(t["id"] == identifier for t in self._traces)
        if workflow is None or not retained:
            raise ValueError("This run's ComfyUI workflow is no longer available; queue it from ComfyUI")
        return deepcopy(workflow)

    def _attach_patching(self, attach_to, result):
        """Store an activation-patching result on the target trace it was measured against."""
        target = self.trace(attach_to)
        if target.get("kind") not in ("attention_capture", "pytorch_module_capture"):
            raise ValueError("Patching attaches to a text capture")
        if target.get("input", {}).get("token_count") != result["target_run"]["token_count"]:
            raise ValueError("The target prompt differs from trace B; rerun B first")
        target["patching"] = {key: result[key] for key in ("method", "positions", "layer_count", "source_run", "target_run", "layers", "limitations")}
        with self._lock:
            record = next((t for t in self._traces if t["id"] == attach_to), None)
            name = record["name"] if record else "Capture"
        self.publish(target, name, replace_id=attach_to)

    def _attach_ablation(self, attach_to, result):
        """Merge an ablation result into the attention trace it was measured against, per layer."""
        target = self.trace(attach_to)
        captured = {target.get("layer")} | {s["index"] for s in target.get("layer_stacks", []) if isinstance(s, dict)}
        if target.get("kind") != "attention_capture" or result["layer"] not in captured:
            raise ValueError("Ablation must target a layer of this attention capture")
        if target.get("input", {}).get("token_count") != result["position"] + 1:
            raise ValueError("The prompt differs from the attention capture; capture attention again first")
        if result["head_count"] != target.get("head_count", result["head_count"]):
            raise ValueError("The attention capture reports a different head count")
        blocks = [b for b in target.get("ablations", []) if isinstance(b, dict)]
        existing = next((b for b in blocks if b.get("layer") == result["layer"]), None)
        if existing is None and result["layer"] == target.get("layer") and target.get("ablation"):
            existing = target["ablation"]
        heads = {entry["head"]: entry for entry in (existing or {}).get("heads", [])}
        for entry in result["heads"]:
            heads[entry["head"]] = entry
        block = {key: result[key] for key in ("method", "layer", "head_count", "head_dim", "position", "baseline", "limitations")}
        block["heads"] = [heads[key] for key in sorted(heads)]
        target["ablations"] = [b for b in blocks if b.get("layer") != result["layer"]] + [block]
        target["ablations"].sort(key=lambda b: b["layer"])
        if result["layer"] == target.get("layer"):
            target["ablation"] = block
        with self._lock:
            record = next((t for t in self._traces if t["id"] == attach_to), None)
            name = record["name"] if record else "Attention capture"
        self.publish(target, name, replace_id=attach_to)

    def publish(self, trace, name="Imported trace", *, replace_id=None, run=None, step=None, note=None):
        """Keep a trace in memory as encoded bytes. Traces of one run are kept and evicted together."""
        data = encode_trace(trace)
        record = {"id": uuid4().hex, "name": str(name)[:160], "kind": trace["kind"], "size_bytes": len(data), "data": data,
                  "revision": 1, "run": run, "step": step, "pinned": False, "note": str(note)[:48] if note else None}
        with self._lock:
            previous = next((t for t in self._traces if t["id"] == replace_id), None)
            if previous is not None:
                self._traces.remove(previous)
                self._trace_bytes -= previous["size_bytes"]
                record.update(id=previous["id"], revision=previous["revision"] + 1, run=previous["run"],
                              step=previous["step"], pinned=previous["pinned"], note=record["note"] or previous["note"])
            record["run"] = record["run"] or record["id"]
            record["pinned"] = record["pinned"] or any(t["run"] == record["run"] and t["pinned"] for t in self._traces)
            self._traces.append(record)
            self._trace_bytes += len(data)
            self._evict(record)
        return {"id": record["id"]}

    def _over_limits(self):
        return (len({t["run"] for t in self._traces}) > MAX_RUNS or len(self._traces) > MAX_TRACES
                or self._trace_bytes > MAX_RETAINED_BYTES)

    def _drop(self, record):
        self._traces.remove(record)
        self._trace_bytes -= record["size_bytes"]
        self._native_workflows.pop(record["id"], None)

    def _evict(self, newest):
        """Oldest unpinned run first, then older traces of the newest run, then pinned runs as a last resort."""
        while self._over_limits():
            order = list(dict.fromkeys(t["run"] for t in reversed(self._traces)))[::-1]
            others = [run for run in order if run != newest["run"]]
            pinned = {t["run"] for t in self._traces if t["pinned"]}
            unpinned = [run for run in others if run not in pinned]
            same_run = [t for t in self._traces if t["run"] == newest["run"] and t is not newest]
            if unpinned or not same_run and others:
                victim = (unpinned or others)[0]
                for record in [t for t in self._traces if t["run"] == victim]:
                    self._drop(record)
            elif same_run:
                self._drop(same_run[0])
            else:
                break

    def trace(self, identifier):
        with self._lock:
            for record in self._traces:
                if record["id"] == identifier:
                    data = record["data"]
                    break
            else:
                raise ValueError("Trace is no longer retained")
        return json.loads(data)

    def pin_trace(self, identifier, pinned):
        """Keep a whole run out of ordinary eviction, up to a cap that always leaves room for new captures."""
        if type(pinned) is not bool:
            raise ValueError("pinned must be true or false")
        with self._lock:
            record = next((t for t in self._traces if t["id"] == identifier), None)
            if record is None:
                raise ValueError("Trace is no longer retained")
            members = [t for t in self._traces if t["run"] == record["run"]]
            if pinned and not record["pinned"]:
                held = {t["run"] for t in self._traces if t["pinned"]}
                held_bytes = sum(t["size_bytes"] for t in self._traces if t["pinned"])
                if len(held) >= MAX_PINNED_RUNS:
                    raise ValueError(f"At most {MAX_PINNED_RUNS} runs can be pinned; unpin one or export it first")
                if held_bytes + sum(t["size_bytes"] for t in members) > MAX_PINNED_BYTES:
                    raise ValueError(f"Pinned runs can hold at most {MAX_PINNED_BYTES // (1024 * 1024)} MiB; unpin one or export it first")
            for member in members:
                member["pinned"] = pinned
        return {"pinned": pinned}

    def delete_trace(self, identifier):
        """Delete the whole run this trace belongs to, so no partial run is left behind."""
        with self._lock:
            record = next((t for t in self._traces if t["id"] == identifier), None)
            if record is None:
                raise ValueError("Unknown trace")
            for member in [t for t in self._traces if t["run"] == record["run"]]:
                self._drop(member)
        return {"deleted": True}

    def cancel(self):
        with self._lock:
            self._cancelled = True
            self.host.cancel()
        return {"requested": True}

    def close(self):
        with self._lock:
            self._closed = True
        self.installer.close()
        self.cancel()
        self._executor.shutdown(wait=True)
        return self.host.close()


def discard(target):
    """Free a model's weights where they are, without copying them to the CPU first."""
    modules = [target] if hasattr(target, "parameters") else [c for c in getattr(target, "components", {}).values() if hasattr(c, "parameters")]
    for module in modules:
        try:
            module.to("meta")
        except Exception:  # noqa: BLE001 - dropping the references still frees it
            pass


def loaded_dtype(adapter):
    model = getattr(adapter, "model", None) or getattr(adapter, "pipeline", None)
    try:
        return str(next(model.parameters()).dtype).replace("torch.", "")
    except Exception:  # noqa: BLE001 - pipelines expose no single dtype
        return None


def with_start_token(tokenizer, ids):
    """Prepend the start-of-text token when the tokenizer defines one but left it off, as Gemma 4's does.

    A start token that is also the end token (GPT-2) is not one the model expects, so it is left alone.
    """
    import torch
    start = getattr(tokenizer, "bos_token_id", None)
    ends = getattr(tokenizer, "eos_token_id", None)
    ends = set(ends) if isinstance(ends, (list, tuple)) else {ends}
    if not isinstance(start, int) or start in ends or (ids.shape[1] and int(ids[0, 0]) == start):
        return ids
    return torch.cat([torch.tensor([[start]], dtype=ids.dtype), ids], dim=1)


def text_layer_paths(model, tree, limit=32):
    """The language model's numbered layers first, since image and audio towers do not run on a text prompt."""
    numbered = [n["path"] for n in tree["nodes"] if n["path"].split(".")[-1].isdigit()]
    prefix = decoder_prefix(model)
    inside = [path for path in numbered if not prefix or path.startswith(prefix + ".")]
    return (inside or numbered)[:limit]


def decoder_prefix(model):
    """Where the language model's decoder sits in the module tree, or "" when it is the root or unknown."""
    try:
        decoder = model.get_decoder()
        return next(name for name, module in model.named_modules() if module is decoder)
    except Exception:  # noqa: BLE001 - not every model names its decoder
        return ""
