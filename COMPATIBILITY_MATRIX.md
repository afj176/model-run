# Compatibility evidence

The [application release report](reports/application-release-2026-11-04.md) adds standalone HTTP/UI checks and an actual native ComfyUI sampler workflow. The extension reads the live host's model-path registry; standalone mode uses explicitly configured paths. Historical limitations below describe the earlier campaigns, not the current host integration. Full-workflow evidence currently covers FLUX.2 Klein; other families retain their separately documented forward/capture scope.

The broader 2026-11-04 native-checkpoint campaign is documented in `reports/local-model-campaign-2026-11-04.md`. It adds native loading of all six local diffusion models and all six encoders, decode/capture comparisons for all six VAEs, a latent-upscaler forward comparison, and native LoRA registration for both local adapters. The exact remaining inference gaps are listed there.

Subsequent gates are recorded in `reports/runtime-gates-2026-11-04.md`: all six diffusion models and all six encoders passed short forward/capture comparisons; both LoRAs passed numerical patch-application checks. Those later results supersede the earlier report's corresponding pending gates, within the stated test scope.

This is a record of tested configurations, not a claim of general support for every variant. Capture means selected module outputs are compared against an uninstrumented forward pass; metadata inventory does not execute a model. All observed inference tests below use float32 on CUDA, PyTorch 2.12.0.dev20260312+cu128 and Transformers 5.13.0 unless otherwise stated. Synthetic inputs are explicitly identified.

| Artifact | Revision | Test | Result and scope |
| --- | --- | --- | --- |
| Qwen/Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca | Local safetensors, real tokenizer, short text, first/last layer; eager attention head | Pass on CPU and CUDA; captured logits match baseline; selected attention matches reference. See VALIDATION.md. |
| HuggingFaceTB/SmolLM2-135M | 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 | Synthetic token IDs; first transformer layer | Pass on CUDA; captured logits match baseline. Tokenization and text generation not covered. |
| google/vit-base-patch16-224 | 3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3 | Synthetic image tensor; first transformer layer | Pass on CUDA; classification logits match baseline. Image preprocessing and real images not covered. |
| openai/whisper-tiny | 169d4a4341b33bc18d8881c4b69c2e104e1cc0af | Synthetic audio feature tensor; first encoder layer | Pass on CUDA; logits match baseline. Audio preprocessing and transcription not covered. |
| hf-internal-testing/tiny-stable-diffusion-pipe | Local converted safetensors copy; original revision not recorded | Three denoising steps, same seed | Pass on CUDA with Diffusers 0.37.0; latent output matches baseline. Original checkpoint conversion used pickle-based files; see VALIDATION.md. |
| Local ComfyUI safetensors, including LTX 2.5 video/audio VAE and transformer, FLUX, Qwen Image, LoRAs and text encoders | Local files; content hashes not recorded | Header and tensor index inspection only | 21 files inventoried successfully. No loading, inference or runtime capture is implied. |

Hugging Face downloads and Ollama are test sources, not the planned primary discovery surface. An initial explicit extra-path YAML reader has been checked against the Open Cinema runtime configuration: it identified 10 configured model directories, all present, while excluding custom-node paths. Built-in paths, runtime-added paths, automatic configuration discovery and host integration are not yet covered. Do not modify the user's models.

The bounded catalog scanner was manually checked against those 10 directories and inventoried 21 safetensors files without errors or truncation. Automated discovery tests use temporary generated fixtures and do not depend on personal Hugging Face, Ollama or ComfyUI paths. Real-checkpoint smoke scripts take their model directory as an explicit argument and are opt-in.

The catalog CLI was subsequently checked with the shared ComfyUI models root and the Open Cinema extra-path configuration together: it deduplicated their overlap and inventoried 21 safetensors files without scan errors. Standard categories are a static mapping based on the locally inspected ComfyUI version, not live introspection; extensions may add or alter paths.
