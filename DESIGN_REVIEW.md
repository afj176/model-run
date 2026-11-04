# Model Run — design review

Status: design only; no implementation or runtime validation. The selected visual concept is the warm, sans-serif interface with a navigable 3D model and contextual inspector. Model Run is a working name.

## Product contract

A local observatory for model structure and, when an instrumented runtime supports it, actual inference. The same frontend and core services should work as a standalone web application and through a ComfyUI integration. A model is selected first; capabilities come from its architecture and runtime. Text, image, video and audio are model capabilities, not buttons that transform one model into another.

## Screen review

- **Header and model selector:** show model identity, load state, source, architecture and capabilities. Provide explicit loading, download, error and unload states. Avoid claiming universal model compatibility.
- **Input:** change controls by supported task. Text models accept text; other adapters may accept prompts, media or settings. Display actual tokenizer output including special tokens and distinguish raw text from chat-template input.
- **3D model:** generate geometry from adapter-provided component metadata. Orbit, pan, zoom, reset, focus and selection are achievable with browser 3D graphics. Selected components expand into operations only when the adapter describes them. Keep a 2D/list navigation and keyboard-accessible inspector for usability and fallback. Geometry represents structure, not a literal physical placement of weights.
- **Layer stack:** one item per actual component, indexed consistently; collapse or aggregate large models and clearly indicate what is hidden. Color only captured data, with named statistic, normalization, scale and capture context; otherwise use neutral styling. A model may have branches or stages rather than a linear stack.
- **Inspector:** report component type, shape, metadata and explicitly captured tensors. Attention is conditional on runtime support; select layer, query head and tokens. Identify query/key axes, causal masking and aggregation. Do not present attention as a causal explanation. Neuron-level MLP values are optional and must identify precisely which intermediate tensor was captured.
- **Controls:** show only capabilities available for the selected model and trace. Display toggles affect the scene, not computation; capturing additional data requires a new run when it was not recorded. Compare and export are later features, not assumed first-release functions.
- **Output:** next-token probabilities belong to language models and the actual generation position. Distinguish raw softmax probabilities from distributions after sampling filters and record sampling settings. Other modalities use appropriate output views.
- **Timeline:** distinguish captured execution events from animated playback of a completed trace. Do not claim layer-by-layer real-time timing without measured events. Generation token steps, diffusion steps and video frame time are different axes. Playback seeking inspects retained data, not reversible model execution.
- **Status and failure:** show loading, idle, recording, completed, cancelled, out-of-memory, unsupported and partial capture states. Allow cancellation, show capture cost, and preserve the last successful trace.

## Qwen3-0.6B corrections

The mockup is illustrative and must not be used as architecture data. The official configuration has 28 decoder layers, hidden size 1,024, MLP intermediate size 3,072, 16 query heads, 8 key/value heads, head dimension 128 and SiLU activation. It uses grouped-query attention and a gated MLP; the mockup's 2,048 hidden size and GELU diagram are incorrect. Use model configuration and an architecture-specific adapter; validate shapes against captured output. Index layers 0–27. Do not equate the 3,072 MLP units with every hidden dimension or with all model parameters.

Sources: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json and https://huggingface.co/Qwen/Qwen3-0.6B

## Proposed technical boundaries

1. **Model adapter:** identifies supported architecture/runtime, discovers component tree and capabilities, manages loading and instruments a forward pass. Start with a pinned, tested Qwen3 Transformers/PyTorch adapter and local safetensors; do not promise GGUF or arbitrary checkpoints. Treat unrecognized architectures as unsupported or metadata-only where verified.
2. **Capture:** bounded token count, selected layers/heads/tensors and configurable retention. Use inference mode and scoped hooks with cleanup; avoid uncontrolled GPU-to-CPU copies. Attention implementations may not expose attention matrices; explicitly select and verify a compatible implementation or report unavailable. Instrumentation can increase memory, change speed and introduce small numerical differences.
3. **Trace contract:** versioned records for model revision, runtime, device, dtype, tokenizer and input mode, generation parameters, capture configuration, component IDs, event ordering, tensor dimensions and provenance. Distinguish exact data, aggregated data and conceptual geometry. Restrict trace size and avoid silently saving prompts.
4. **Renderer:** WebGL/WebGPU-capable 3D renderer with scene layout generated from component metadata; instancing, level of detail, bounded labels, picking, clipping and camera controls. Maintain synchronized DOM inspector and keyboard navigation. Fall back to a 2D component tree if 3D is unavailable. Visual animation is replay of trace events unless explicitly measured live.
5. **Hosts:** shared UI and core APIs; standalone Python service for local use and a thin ComfyUI custom-node/web-extension integration. Verify actual lifecycle, routing, dependencies and GPU coexistence against the installed ComfyUI version before implementation. Never assume the existing ComfyUI image workflow provides a Qwen LLM runtime.
6. **Local trust:** bind standalone service to loopback by default; constrain model file access to configured locations and avoid executing remote model code by default. Model downloads are explicit; validate artifacts and surface license and size. Do not expose arbitrary filesystem paths through the UI.

## Modality roadmap

First validate one text model end-to-end. Image, video and audio require independent adapter work, not just a new UI tab. For a supported diffusion pipeline, capture selected denoising steps and decode previews with an appropriate decoder; video additionally distinguishes frame time from denoising step; audio views depend on whether the model provides audio tokens, latent tensors, waveforms or other representations. A spectrogram derived from output audio is not automatically an internal activation. Each adapter declares capabilities and limitations.

## Acceptance before expanding scope

- For a fixed prompt, selected model and revision, verify token IDs, output logits/probabilities, shapes and captured attention/activations against direct reference inference with suitable numerical tolerances.
- Verify selected layer/head/token maps to the captured tensor; masked attention and normalization are represented correctly; absence of capture never appears as zero.
- Demonstrate orbit, selection, focus, readable keyboard-accessible inspector, 2D fallback, bounded memory, cancellation and useful errors.
- Verify one trace replay and export preserves provenance and declares reductions; do not advertise export until implemented.
- Verify standalone and ComfyUI host flows separately, including lifecycle cleanup and resource contention.
- Measure startup, inference, capture overhead and browser performance on the target machine before setting performance targets.

## Open design decisions

Whether initial playback is captured-then-replayed or incremental streaming; exact model acquisition and hardware settings; ComfyUI launch surface and whether an output node is useful; capture retention and export format; which image adapter follows the first text release. These require implementation spikes and user review, not visual mockup assumptions.
