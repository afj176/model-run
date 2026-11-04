# Model Run — compatibility architecture and execution plan

Status: research and architecture proposal. No compatibility or performance claims are validated until backed by conformance results. Visualization implementation is deliberately downstream of the backend contracts.

## Objective and boundary

Maximize useful inspection across local models and modalities without falsely promising that arbitrary artifacts expose execution or internal values. Compatibility is per artifact, architecture, task, runtime, version, device and capture mode—not a single model-family label. Opaque hosted APIs can expose only their available inputs, outputs and metadata. Proprietary, encrypted or unsupported formats may not even expose structure.

## Support levels (independent capabilities, not a strict ladder)

- Inventory: files, format, sizes, tensor names/shapes/dtypes and metadata where readable. No graph or execution inferred from weights alone.
- Structure: explicitly represented graph, verified model configuration, or runtime module hierarchy; attach provenance and confidence to each node/edge. Module nesting is not automatically execution order.
- Execution: a tested runtime, supported task, preprocessing and model-specific artifacts. Explicitly identify device, runtime, versions, precision and model revision.
- Observation: selected measured events, intermediate values and/or output; availability varies by runtime and optimized/fused kernels. No absent data represented as zero.
- Intervention: modifying activations/weights requires separate runtime support, isolation and controls; excluded from initial foundation.

## Architecture

1. Artifact discovery: local file or directory; safely classify safetensors shards/index, Hugging Face config, ONNX, GGUF and ComfyUI checkpoints using verified readers, not extensions alone. Preserve unknown files. Report evidence and unresolved components. For sharded files validate indices, missing shards, duplicate keys and bounds; inventory lazily, no full weight load. Metadata-only does not execute model code. Consider formats separately: a checkpoint containing tensors is not an executable model.
2. Catalog and identity: content fingerprints, source, revision, file list, format, tensor index, architecture candidates, modality/task candidates, licenses where known. Distinguish inferred from declared and measured. Never identify model solely from a filename; do not silently guess shapes or tokenizer.
3. Format providers: independent reader plugins returning typed artifact metadata and errors. Safetensors header inventory first; ONNX graph parser and GGUF metadata readers in later validated increments. No untrusted pickle deserialization during inventory. Resource limits, path containment, malformed inputs and symlinks require tests.
4. Architecture providers: map trustworthy configurations/runtime structures to a canonical model graph with stable IDs, typed nodes, ports, edges, tensor shapes/dtypes and provenance. A generic graph is allowed to be incomplete; operation order must not be fabricated. Provide Qwen3 as a reference implementation, then test another independent architecture to avoid coupling abstraction to Qwen.
5. Runtime providers: independent from artifact and architecture providers. A provider negotiates execution, tracing and individual capture capabilities, versions, dtype, device and task; it supplies explicit reasons for unsupported options. First PyTorch/Transformers eager; subsequently a ComfyUI-native adapter and other runtimes only after proving instrumentation. Do not presume a PyTorch hook works on fused, compiled, quantized or remote engines.
6. Task and modality providers: preprocessing, inputs, generation, output decoding and trace axes. Keep text autoregression, image diffusion, video diffusion and audio as separate contracts; multimodal models may combine several. A ComfyUI workflow can contain multiple distinct models, conditioning, sampler and VAE, so represent the workflow and its component models separately.
7. Trace transport/storage: versioned manifest and typed events plus bounded tensor payloads; stable IDs, exact versus summary, reduction method, axis names, timestamps, truncation and provenance. Support incomplete/failed captures. Explicit bounded recording, cancellation, redaction and retention. Streaming events and replay share a schema but report dropped events and capture overhead.
8. Consumer interface: graph, capability report and trace available without rendering; a future 3D client consumes these contracts. Standalone and ComfyUI host adapters use the same engine where practical, with explicit process and GPU lifecycle ownership.

## Milestones and validation gates

A. Contract and fixture suite: schemas, capability matrix, errors, provenance, small synthetic fixtures; deterministic contract tests. No dependency on visualization.
B. Artifact inventory: safetensors and sharded manifests first. Test malformed metadata, missing/duplicate shards, partial and empty tensors, large headers, limits and no execution. Report exact supported scope. Add ONNX and GGUF separately with real and generated fixtures.
C. Generic structure: ONNX graph and PyTorch module hierarchy with accurate source labels; represent unknowns, branching, dynamic shapes and partial graph. Compare against source graphs; do not claim module hierarchy is a computation trace.
D. Reference execution: Qwen3-0.6B, pinned dependencies and revision, local/download consent, CPU plus available GPU. Capture bounded tokenization, logits and selected layer/attention where supported. Compare directly against independent reference forward passes and ensure hooks are cleaned up. Measure overhead and resource budgets. Add a structurally different small model as the abstraction test.
E. ComfyUI: inspect installed version and existing model management and server integration; prototype a dedicated process/service boundary and a native-node boundary. Test real host lifecycle, no duplicate model assumptions, workflow identity, cancellation and VRAM contention. No claim of universal hooks without validation.
F. Image, video, audio: select actual representative small models and workflow for each, verify distinct preprocessing, runtime, sampling and decode requirements; contract tests for relevant timelines. Add adapters incrementally. Capture supported intermediate data without assuming a shared architecture.
G. Trace/replay API: compare recorded payloads and ordering to reference, storage limits and corruption handling, clean cancellation and sensitive input retention. Only then build the 3D frontend on the proven contracts.

## Compatibility matrix and test evidence

Each row identifies exact artifact revision, format, architecture, task, runtime version, device, precision, capabilities, instrumentation method, test date and limitations. Tests classify supported/partial/unsupported with evidence, rather than inferring compatibility from a model family. Include golden small fixtures, negative/corrupt cases and integration tests; performance and numerical comparisons are measured on actual hardware.

## Research references

- Safetensors describes tensor storage and metadata, including shapes and dtypes, not model execution: https://huggingface.co/docs/safetensors/metadata_parsing
- ONNX provides a serialized computation graph, with extensible operators and runtime-dependent execution: https://onnx.ai/onnx/repo-docs/IR.html
- PyTorch export captures graphs subject to limitations including control flow and unsupported features: https://docs.pytorch.org/docs/stable/export.html
- Diffusers provides step callbacks with pipeline-specific tensor availability; latent previews are model-dependent: https://huggingface.co/docs/diffusers/en/using-diffusers/callback
- ComfyUI extension discovery and frontend integration: https://github.com/Comfy-Org/docs/blob/main/custom-nodes/walkthrough.mdx
