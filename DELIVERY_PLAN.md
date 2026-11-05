# Model Run v0.1 delivery

The initial release is a local model observatory, available standalone and inside
ComfyUI. Completion means a usable application, not compatibility with every
possible architecture. Existing checkpoint verification remains the evidence for
the tested families.

## Release acceptance

Status: implemented and checked for the v0.1 scope below. Evidence and remaining compatibility boundaries are in [the application release report](reports/application-release-2026-11-04.md) and [visual QA](design-qa.md).

- Browse ComfyUI model roots and explicit extra-path configurations. No personal
  Hugging Face, Ollama or Unsloth paths in product defaults or automated tests.
- Inspect bounded artifact metadata and runtime structure with clear provenance.
- Load local safetensors Transformers causal models and local Diffusers pipelines,
  one resident application-owned model at a time; explicit device and unloading.
- Run bounded text capture, optional attention, and diffusion latent capture.
- Observe native ComfyUI denoiser calls through an opt-in MODEL node, preserving
  existing wrappers, model management, conditioning and sampler behavior.
- Share one frontend between hosts: interactive 3D structure, tensor values,
  component list/inspector, attention, probabilities, replay, import/export,
  cancellation, progress and actionable errors. No fabricated measured data.
- Package assets locally, with no CDN required. Keyboard/list fallback and reduced
  motion. Test API boundaries, lifecycle, trace fidelity and browser interactions.

## Explicit release boundaries

Remote code, downloading without an explicit request, intervention, model training, generic media
preprocessing, operation-graph inference, and universal architecture support are
outside this release. ComfyUI owns image/video/audio generation; Model Run observes
its actual denoiser calls. Standalone diffusion reports latent steps and supported
decoded image output. Tensor spatial layouts are display layouts with explicit
axis/selection labels, never literal neuron positions.

## Work sequence

1. Application task contracts and bounded API.
2. Native ComfyUI bridge and integration tests.
3. Shared frontend and data-driven 3D renderer.
4. End-to-end real checkpoint checks, visual QA, packaging and operating docs.
