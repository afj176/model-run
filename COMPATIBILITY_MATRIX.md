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

## ComfyUI model families

Checked by `tools/check_comfy_families.py`, which builds each family's real ComfyUI class at a tiny size with random weights and runs the real Observe code on it. It tests structure and code paths, not results: random weights say nothing about what a real model learned, and tiny configs can differ from released ones.

| Family | Layers found | Output unchanged | Attention map | Text keys sit |
|---|---|---|---|---|
| SD1.5 / SD2 UNet | 9 | yes | yes | text only (cross-attention) |
| SDXL UNet | 9 | yes | yes | text only (cross-attention) |
| Stable Diffusion 3 | 2 | yes | yes | first |
| AuraFlow | 3 | yes | yes | after 8 other tokens |
| Hunyuan DiT | 2 | yes | yes | first |
| PixArt | 3 | yes | yes | text only (cross-attention) |
| Flux 1 | 4 | yes | yes | first |
| Flux 2 | 4 | yes | yes | first |
| Chroma | 6 | yes | yes | first |
| Hunyuan Video | 6 | yes | yes | first |
| Wan 2.x | 2 | yes | yes | text only (cross-attention) |
| LTX Video | 2 | yes | yes | text only (cross-attention) |
| Mochi | 2 | yes | yes | after the image (from its source; every key changes with the text here) |
| Cosmos Predict2 | 2 | yes | yes | text only (cross-attention) |
| Lumina 2 / Z-Image | 2 | yes | yes | first |
| Qwen-Image | 2 | yes | yes | first |
| Qwen-Image 2.1 | 2 | yes | yes | first |
| HiDream I1 | 2 | yes | yes | after 64 other tokens |
| Flux 3 | 2 | yes | yes | after the image |
| Lens | 2 | yes | yes | after the image |
| OmniGen 2 | 2 | yes | yes | first |
| Boogu | 3 | yes | yes | first |
| Stable Audio | 2 | yes | yes | text only (cross-attention) |
| ACE-Step | 4 | yes | yes | after 1 other tokens |
| Mage Flow | 2 | yes | yes | first |
| JoyImage | 2 | yes | yes | after the image |
| Ideogram 4 | 2 | yes | yes | first |
| Krea 2 | 6 | yes | yes | first |
| MOSS Image | 4 | yes | yes | after the image |
| Fibo 2 | 3 | yes | yes | first |
| Kandinsky 5 | 3 | yes | yes | text only (cross-attention) |
| Ernie Image | 2 | yes | yes | after the image |
| CogVideoX | 2 | yes | yes | first |
| Hunyuan Image 2.1 | 6 | yes | yes | first |
| PixelDiT | 3 | yes | yes | first |
| Wan causal AR | 2 | yes | yes | text only (cross-attention) |
| Wan VACE | 3 | yes | yes | after 96 other tokens |
| Wan Camera | 3 | yes | yes | text only (cross-attention) |
| Wan HuMo | 2 | yes | yes | text only (cross-attention) |
| Wan Animate | 2 | yes | yes | text only (cross-attention) |
| Wan Animate 2 | 2 | yes | yes | text only (cross-attention) |
| Wan S2V | 3 | yes | yes | text only (cross-attention) |
| Wan SCAIL | 2 | yes | yes | text only (cross-attention) |
| Wan SCAIL 2 | 2 | yes | yes | text only (cross-attention) |
| Wan Dancer | 2 | yes | yes | not reached |
| Cosmos Video | 6 | yes | yes | text only (cross-attention) |
| Z-Image pixel space | 3 | yes | yes | first |
| PiD | 4 | yes | yes | first |
| Chroma Radiance | 7 | yes | yes | first |
| Stable Cascade C | 8 | yes | yes | after 256 other tokens |
| Stable Cascade B | 4 | yes | no: conditions on a pooled summary, not words | after the image |
| MiniMax H3 | 3 | yes | yes | first |
| ACE-Step 1.5 | 8 | yes | yes | after the image |
| Trellis 2 | 6 | yes | yes | text only (cross-attention) |
| Hunyuan3D 2 | 2 | yes | no: shape tokens are not an image | first |
| Hunyuan3D 2.1 | 3 | yes | no: shape tokens are not an image | text only (cross-attention) |
| TripoSplat | 6 | yes | no: shape tokens are not an image | after 16 other tokens |
| YuE 2 | 2 | yes | no: its text arrives as a precomputed key cache | not reached |
| MiniMax Music 3 | 36 | yes | no: its condition is mixed into the audio, with no attention to text | not reached |
| SenseNova U1.5 | 1 | yes | no: text is token IDs inside the model, not a text input | not reached |
| Anima | 8 | yes | yes | text only (cross-attention) |

Where a map exists, the check also changes only the text and confirms the map reads text where the keys really changed; a mismatch prints `WRONG`. Word labels are a separate check that needs real weights, and the app says when a family's have not been checked.

Not in this table:

- LTX 2.5 with audio, HiDream-O1 and SeedVR2 are checked on real weights below instead.
- SVD, Stable Zero123, SV3D, Lotus, SD x4 upscaler, instruct-pix2pix, SSD-1B, LongCat Image, Ming Image, Hunyuan Video 1.5 and the other subclasses use a network class listed above with a different config. They have not been run themselves.
- SAM 3, Depth Anything 3, RT-DETR, MoGe and SAM 3D Body do not sample, so `Observe`, which records sampler calls, does not apply to them.
- YuE 2, MiniMax Music 3 and SenseNova U1.5 were built from their fixed sizes with fewer layers or smaller settings, and SenseNova's text path was not exercised.

## Real weights through ComfyUI

| Model and workflow | Result |
|---|---|
| Flux2 Klein: square, wide 512x256 with guidance, batch of 2 | All 25 layers, maps, labels checked |
| Qwen-Image 2.1: with and without Turbo LoRA, 768x512 | All 32 layers, maps, labels checked |
| Qwen-Image Edit 2511: reference image, Lightning LoRA | All 60 layers on the image stream, maps with numbered tokens; labels are dropped because image tokens are in the text |
| HiDream-O1 | Maps, labels checked |
| LTX 2.5: 9 and 33 frames, with and without guidance, audio stream | All 48 layers, video and audio maps, labels checked |
| LTX 2.3 dev: Gemma 3, distilled LoRA | All 48 layers, maps, labels place real tokens first |
| SD 1.5: 512, guidance 7 | 26 layers, all with views; maps on every block with attention; labels checked by swapping words and by masks |
| SDXL base: 1024, guidance 7 | 20 layers, all with views; maps on input 4 to output 5, where its attention blocks are; labels checked the same way |
| Text Encode, VAE Decode and Observe's sampler steps: SD 1.5, SDXL, Flux2 Klein, LTX 2.5 | All recorded, and output pixel-identical to plain nodes. VAE Encode checked on SD 1.5 |
| VAE Decode and sampler steps: Qwen-Image 2.1 | Recorded, output pixel-identical. It uses its own text node, so Text Encode does not apply |
| Sampler steps: HiDream-O1 | Recorded, output pixel-identical. Its VAE and text encoder have no layers to record |
| SeedVR2 | Layers recorded; no map, because it bypasses the hook and conditions on fixed embeddings |
