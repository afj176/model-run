# Validation evidence

## Application release — 2026-11-04

The standalone application, native ComfyUI extension, browser acceptance and packaging are recorded in [the release report](reports/application-release-2026-11-04.md). The current automated suite passes 58 tests. Real application runs cover Qwen3-0.6B and the tiny safetensors diffusion pipeline. A native two-step FLUX.2 Klein workflow produced identical baseline and observed latents (maximum absolute difference 0). The ComfyUI frontend exposes Model Run → Open Model Run. Earlier entries below retain their original test scope; their historical host/UI omissions are superseded only by these specific checks.

## Model host lifecycle — 2026-11-04

Generated fixtures verify bounded registration, duplicate-resource rejection, cross-model observation exclusion, cancellation, trace consumption, resource-release exclusion, retry after cleanup failure, shutdown refusal of new work and preservation of original errors. The Qwen CUDA smoke completed next-token capture through ModelHost and compared probabilities with the prior direct run, then verified that the explicit release callback moved the model to CPU. The tiny safetensors Stable Diffusion CUDA smoke completed hosted denoising capture, compared the final latent prefix with its reference and verified CPU placement of UNet weights after host shutdown. These tests validate only the registered host's own scheduling and provided cleanup callbacks, not system-wide GPU allocation.

## Observation lifecycle — 2026-11-04

Generated fixtures verify session exclusivity, completion/trace consumption, cancellation, checkpoint deadlines, invalid/oversized trace rejection, partial trace retention, closing active work, failure recovery, hook removal and lock release. The real Qwen3-0.6B CUDA smoke cancelled after the first selected decoder layer, retained one module event, verified that pre-existing Transformers hooks were preserved, and then successfully repeated next-token capture against the baseline logits. The tiny safetensors Stable Diffusion CUDA smoke cancelled at the second denoising callback, retained the first step, and successfully repeated a complete same-seed run matching the baseline latent. These are cooperative boundary checks, not evidence of preempting kernels, cancelling loading or reclaiming host-owned model allocations.

## Runtime binding layer — 2026-11-04

The explicit registry and binding API passed fixture-based tests for capability negotiation, evaluation-mode gating, unknown/duplicate provider rejection, observation lock recovery after errors, and Diffusers callback readiness. The real local Qwen3-0.6B CUDA smoke was rerun through the causal-LM binding, including reference logits and attention comparison. The tiny safetensors Stable Diffusion CUDA smoke was rerun through the pipeline binding and matched the unobserved latent output. The native FLUX.2 Klein forward/capture comparison was rerun through the generic PyTorch binding and passed. These checks validate the new dispatch path for those selected workloads; they do not extend earlier checkpoint guarantees to untested versions or workloads.

A pure-Python wheel was built through the setuptools PEP 517 backend and its core/registry imported directly from the wheel with Python isolated mode and site packages disabled. Package contents were restricted to source and wheel metadata. The pip wheel wrapper stalled in this environment and was cancelled; pip installation itself has not been validated by this check.

## Qwen3-0.6B local smoke

On 2026-11-03, `Qwen/Qwen3-0.6B` snapshot `c1899de289a04d12100db370d81485cdf75e47ca` was downloaded into the project-local, ignored `.models` directory. `tests/real_qwen_smoke.py` loaded its local safetensors and tokenizer with remote code disabled and ran a five-token prompt on CPU and CUDA. PyTorch reported `2.12.0.dev20260312+cu128`; Transformers reported `5.13.0`. The run captured the first and last transformer layer outputs, verified the 28-layer count and 1,024 hidden width, and compared captured forward-pass logits with a separate uninstrumented forward pass. Both selected ` Paris` as the top next token. The safetensors metadata reader reported 311 tensors.

Re-run with `python tests/real_qwen_smoke.py .models/Qwen3-0.6B cpu` or `cuda`. This smoke tests one checkpoint, prompt, precision (float32), environment and two modules. It does not validate attention matrices, quantized or compiled models, long contexts, model downloads as a product feature, concurrent GPU workloads, ComfyUI or other model families. Runtime duration includes model loading and capture and is not a benchmark.

The automated test suite additionally exercises randomly initialized small Qwen2 and GPT-2 implementations and synthetic ONNX and safetensors fixtures. These are implementation tests, not evidence of trained checkpoint support for those model families.

On 2026-11-04 the Qwen CUDA smoke was extended with eager-attention capture of layer 0, head 0. The returned matrix matched a second reference forward pass; row sums and causal masking were checked. This validation is limited to the short prompt and selected head; eager attention may use substantially more memory than optimized attention.

On 2026-11-04 the `model_run.run_text` CLI was run with the same local checkpoint on CUDA. It exported a module capture and a 6-by-6 attention trace and both were successfully reloaded. The PowerShell pipe included a newline, accounting for the additional input token relative to the smoke test. The recorded config hash is not a checkpoint identity verification.

On 2026-11-04 an initial CUDA Diffusers smoke used `hf-internal-testing/tiny-stable-diffusion-pipe`, a small test checkpoint, for a three-step run. Diffusers 0.37.0 loaded its original UNet and VAE from pickle-based `.bin` files, emitting an unsafe-serialization warning; this happened during the test and a one-time local conversion, not through the Model Run loader. Those components were then saved as safetensors in `.models/tiny-sd-safetensors`. The smoke script now requires safetensors. The callback run is compared with an uninstrumented run at the same seed and captures the final latent prefix. This test is specific to this small pipeline, not a general image, video or audio support claim.
