# Runtime adapter contract

The shipped application now owns local loading and bounded jobs through `model_run.application.Application`; `model_run.server` serves the standalone UI. The native `ModelRunObserve` node in `model_run.comfy_bridge` observes denoiser calls while ComfyUI retains loading, conditioning, sampling and device ownership. See [release validation](reports/application-release-2026-11-04.md) for actual host tests. The lower-level binding contract below remains available independently.

Runtime bindings are separate from file discovery and visualization. The caller explicitly binds an already-loaded object through `model_run.adapters.default_registry()`. Core imports do not load optional model libraries. Model construction, tokenizer/media preparation, placement, precision, trust and hardware budgets remain host responsibilities.

| Binding ID | Accepted object | Operations |
| --- | --- | --- |
| pytorch.module | Loaded PyTorch module | Module containment and bounded selected module outputs; suitable for native ComfyUI modules as well as other PyTorch model families |
| transformers.causal_lm | Loaded Transformers generation-capable decoder-only model | Generic module operations, raw next-token probabilities, explicit attention probe |
| diffusers.pipeline | Loaded Diffusers pipeline | Bounded denoising latent callbacks when the pipeline explicitly exposes that interface |

`capabilities()` describes readiness. `negotiate([feature, ...])` accepts only available operations and returns a reason for every denied request. An architecture's checkpoint compatibility is determined by separate conformance evidence, not by a matching Python class. Generic compiled/scripted module observation is conservatively unverified. Attention remains a probe-required capability because an eager setting alone does not guarantee the selected tensor will be returned.

Examples, using caller-owned loaded objects:

```python
from model_run.adapters import default_registry

registry = default_registry()
adapter = registry.bind("pytorch.module", loaded_module)
report = adapter.negotiate(["module_tree", "module_outputs"])
tree = adapter.structure()
output, trace = adapter.capture(
    compatible_inputs,
    ["selected.module"],
    max_tensor_elements=1024,
    output_selectors={"selected.module": ["audio", 0]},
)
```

The selector is optional and must match the actual output container. A bad selector raises an explicit error and hooks are removed. Observed tensors are bounded float32 prefixes, not automatically representative summaries. Capturing may incur device synchronization and reshape-copy overhead.

Bindings serialize their own observations and release locks after failure. They cannot prevent callers from running inference directly on the underlying object or constructing another binding for that object; the host must provide exclusive ownership during capture. Models are not silently moved or switched to evaluation mode. Generic capture blocks when any module is training. Binding a model does not itself run inference.

Pipeline bindings require explicit `num_inference_steps` within the current 1–128 recording limit and reject callback overrides. A fresh recorder is created for each invocation. The trace records latent tensors, not decoded image/audio/video previews, and output decoding remains pipeline-specific. Callback availability alone does not prove support for a checkpoint or workload.

Third-party providers can register a named binding factory in a registry instance. Duplicate names and unknown providers fail explicitly. The registry does not execute arbitrary entry points or install provider dependencies automatically. Host model/device lifecycle and resource scheduling remain host responsibilities.

## Observation sessions and cooperative cancellation

`model_run.sessions.ObservationSession` provides one active observation and one retained trace. A submitted operation receives a `RunControl` and returns only a trace; session results do not retain the model's raw tensor outputs.

```python
from model_run.sessions import ObservationSession

with ObservationSession() as session:
    run_id = session.start(
        lambda control: adapter.capture(
            compatible_inputs, ["selected.module"], control=control
        )[1],
        timeout_seconds=30,
    )
    status = session.wait(timeout=60)
    trace = session.take_trace()
```

The lifecycle is pending → running → completed/failed/cancelled. A cancellation request changes an active run to cancelling until the model reaches a checkpoint and cleanup finishes. The host can inspect `snapshot()`, request `cancel()`, and consume the completed or partial trace with `take_trace()`. Starting another run requires consuming or discarding the previous trace. Closing discards retained traces, signals cancellation and refuses new work.

Pass the same control to `capture`, `next_token`, `probe_attention` or `capture_steps`. PyTorch checks before/after the forward and at selected module hooks; attention checks before/after its forward; Diffusers checks step callbacks. Timeouts are checked at those boundaries. Neither deadlines nor cancellation interrupt an executing GPU kernel, a model loader or a blocked native call. `wait(timeout=...)` limits the caller's wait only; it does not cancel the observation. `close(wait=False)` signals cancellation without waiting for the worker, while context-manager exit waits for cleanup and can therefore block until a native operation returns.

Cancelled and failed captures may expose a validated partial trace with an explicit status. Hooks installed by Model Run are removed; pre-existing framework hooks are preserved. Adapter locks are released on failure and cancellation. Sessions retain at most one validated JSON trace within the 16 MiB serialized limit, return status/progress separately and never automatically write traces to disk. GPU scheduling, unloading caller-owned weights, cancelling model loading, and cross-process ownership require separate host support.

## In-process model host

`model_run.host.ModelHost` registers explicitly loaded bindings under model IDs and runs one observation at a time across all registrations. It rejects duplicate IDs and duplicate registrations of the exact same resource object. The configurable registration limit bounds the number of held model references, not their VRAM footprint. Capabilities listed by `models()` are registration-time snapshots; every submission negotiates current readiness again.

```python
from model_run.host import ModelHost

with ModelHost(max_models=2) as host:
    host.register("selected-model", adapter, release=release_model_resources)
    run_id = host.submit(
        "selected-model",
        "module_outputs",
        inputs=compatible_inputs,
        module_paths=["selected.module"],
        options={"max_tensor_elements": 1024},
    )
    status = host.wait(timeout=60)
    trace = host.take_trace()
```

Supported operations are `module_outputs`, `next_token`, `probe_attention` and `denoising_steps`. Requests supply compatible in-process inputs; this is not a remote tensor decoder, model loader or HTTP API. Unknown or unavailable operations are rejected. Traces include both a host model ID and a unique binding ID so reusing a display ID does not make old results ambiguous.

The host blocks model unregistration and resource callbacks while an observation is pending, running or cancelling. Optional release callbacks must be idempotent and explicitly implement any device movement, patcher cleanup or other resource work. Removing a binding without a callback drops the host reference; it does not force unload externally referenced weights. A failed callback leaves the entry in `release_failed` state and permits retry; it cannot receive new observations in that state.

`close()` refuses new work, cancels and waits for active observations, discards retained traces, then attempts every registered release callback. It returns per-model cleanup errors and permits another close call to retry failed releases. Context-manager exit raises cleanup failures when there is no original exception; it preserves an original body exception and adds cleanup information where the Python version supports exception notes.

All coordination is within one host. Other bindings, direct model calls and other processes remain outside its ownership. Resident-model byte budgets, automatic loading/offloading, interruption of blocked native calls and multi-device admission remain separate work. Long native calls can delay cancellation and shutdown as described above.
