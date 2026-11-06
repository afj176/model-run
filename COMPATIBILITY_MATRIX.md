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
| Wan VACE | 3 | yes | yes | text only (cross-attention) |
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

How much is still unchecked, including which models people use most, is in [the coverage baseline](reports/model-family-baseline-2026-11-05.md).

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
| PixArt Sigma: 512, guidance 4.5, 10 steps | Layers and sampler steps recorded, output pixel-identical. Maps through the attention slot. Masking only "red" removed the red and masking only "wooden" kept it, so labels are marked checked on this one prompt and seed |
| Stable Audio Open: 6 seconds, guidance 6 | Layers and sampler steps recorded, output files byte-identical. Time-strip maps with labels; the last two tokens are its seconds numbers, placed from its source code. Masking tokens changed the output, but word positions were not tested by listening, so labels stay marked unchecked |
| Anima: 768, guidance 4, 12 steps | All layers, sampler steps, text encoder and VAE recorded, output pixel-identical. Maps run on the real prompt tokens, leaving out its 501 zero padding keys. Masking "red" removed the red and masking "wooden" removed the wood |
| MOSS Image `int8`: 768, guidance 2.5, 12 steps | Same, pixel-identical with attention measured. This model exposed that ComfyUI's override skips a module's own int8 kernel, which changed the image, so such models now measure through the attention slot. Masking removed the red and the wood |
| Z-Image `int8`: 768, guidance 4, 12 steps | Same, pixel-identical. Its text is padded before the image, and only maps on joint blocks count. Masking "red" gave a bare wooden cube, and masking "wooden" kept it red |
| Krea 2 `fp8`: 768, 8 steps | Same, pixel-identical. Its encoder drops the template's opening, so labels are the last 12 tokens. Masking "red" gave a wooden cube, and masking "wooden" kept it red |
| MiniMax H3 `int8`, NVFP4 encoder, Attention Backend node: 768 image and 22-frame video | Same, pixel-identical. Masking "red" gave a white cube and masking "wooden" removed the wood in the image test. Video maps cover the video tokens; its audio tokens are in the same sequence and get no strip of their own |
| Attention measured on Flux2 Klein, Qwen-Image 2.1, LTX 2.5 | Still pixel-identical with the new attention routing. A HiDream-O1 rerun also matched, but its block name matched nothing, so it measured no attention |
| Flux 1 Dev `fp8`: 768, 12 steps | Layers, sampler steps, text encoders and VAE recorded, pixel-identical. It exposed that a shared decode turned T5 ids into wrong words, now fixed for any model with several encoders. Masking "red" gave a wooden cube and masking "wooden" removed the wood |
| Wan 2.2 5B: 640x352, 9 frames, guidance 5, 14 steps | Layers, sampler steps, text encoder and VAE recorded, frame pixel-identical. Maps run on the real tokens, leaving out zero padding. Masking removed the red and the wood, though images at this short test size are rough. Wan's other networks have not run on real weights |
| SD3.5 Large `fp8`: 768, guidance 4, 14 steps | Layers, sampler steps, text encoders and VAE recorded, pixel-identical. Keys run CLIP tokens then T5's, and labels now say so. Masking "red" in both gave a wooden cube, and masking "wooden" gave a white table |
| HiDream I1 Dev `fp8`: 768, 14 steps | Pixel-identical. Masking T5's "red" alone barely changed the image. Its keys also hold Llama's tokens twice, and with all three mapped, masking "red" in each gave a teal cube. Masking "wooden" kept a red cube on reddish wood, so that half is inconclusive |
| Chroma1 HD `fp8`: 768, guidance 3.5, 14 steps | Pixel-identical. Masking "red" gave a dark cube, and masking "wooden" kept it red without the wood |
| Hunyuan Video: 512x320, 9 frames, 12 steps | Frame pixel-identical. Labels are the tokens after the template's user header. Masking "red" gave an orange-brown cube. Masking "wooden" kept it red with a table still looking like wood, so that half is inconclusive |
| Wan VACE 1.3B (same network as 14B): 640x352, 9 frames, 14 steps | Frame pixel-identical. Masking "red" gave a white cube, and masking "wooden" removed the wood |
| Wan Animate 2 distilled `int8`: 480x320, 9 frames, reference image and pose | Frame pixel-identical, with maps and labels. Its motion comes from the reference image and pose, and masking words barely changed the result, so labels stay unchecked |
| ACE-Step 1.5 all-in-one: 8 seconds | Audio files byte-identical, with time-strip maps. Labels come from the prompt with default settings, so words after the caption may be off, and the app says so. Not checked by listening |
| Illustrious XL v0.1 (SDXL finetune): 832, 20 steps | Pixel-identical with maps and labels. This early anime model did not draw the test prompt, so the word test says nothing; SDXL's labels are already checked |
| Lens Turbo, GPT-OSS NVFP4 encoder: 768, 4 steps | Pixel-identical. Its encoder keeps only what follows the template's opening, so labels are the last tokens. Masking "red" gave a white cube; masking "wooden" kept the wood |
| Boogu Turbo `fp8`: 768, 4 steps | Pixel-identical; early blocks report that their image tokens do not fit the latent grid, later blocks map. Masking "red" turned the cube mostly white with red spots; masking "wooden" kept the wood |
| Ideogram 4 `int8` with its unconditional model, as in ComfyUI's template: 768, 12 steps | Pixel-identical. It needs a structured caption: a short one gives a gray "blocked" placeholder even without these nodes. Masking "red" gave a white cube |
| OmniGen 2: 768, guidance 5, 16 steps | Pixel-identical; early blocks report that their image tokens do not fit the latent grid. Masking "red" gave a black cube; masking "wooden" kept the wood |
| Ernie Image Turbo, Ministral 3B: 768, 8 steps | Pixel-identical. Masking "red" gave a blue cube; masking "wooden" kept the wood |
| PixelDiT 1.3B: 768, guidance 4, 16 steps | Pixel-identical. Its keys are the opening token and the last 299 of its padded instruction and prompt, and labels now follow that. Masking "red" gave a wooden cube, and masking "wooden" gave a red table |
| Chroma Radiance x0 (pixel space): 768, 16 steps | Pixel-identical. Masking "red" gave a striped wooden block; masking "wooden" kept a wood-colored table |
| Hunyuan DiT 1.2: 768, guidance 6, 20 steps | Pixel-identical. Its keys are BERT's 77 tokens, then mT5's 256, and maps and labels now cover both. Masking "red" in both changed the cube's shape but kept it red, so labels stay unchecked |
| Mage Flow Turbo `int8`: 768, 4 steps | Pixel-identical. Labels are the tokens after the template's opening. Masking "red" gave a white cube, and masking "wooden" gave a stone table |
| JoyImage Edit `int8`: edit a 768 image, guidance 4, 20 steps | Pixel-identical, and it did the edit. The reference image's tokens sit in the text keys after the user header, and labels mark them as image. Masking "blue" kept the cube red, and masking "marble" kept the wooden table |
| Fibo 2 Edit Turbo `int8`: edit a 768 image, 4 steps | Pixel-identical, and it did the edit. Maps on its single blocks; its text keys hold the image inside a structured caption, so tokens are numbered |
| PiD (Flux 1 latents to pixels): 512 to 2048, 4 steps | Pixel-identical, with maps. It runs on an empty prompt, so there are no words to label |
| Wan 2.2 S2V `fp8`: 480x320, 9 frames, reference image and a test beat through wav2vec2 | Frame pixel-identical, with maps and labels. Masking "red" gave a dark green cube, and masking "wooden" gave a white table |
| HuMo 17B `fp8`: 480x320, 9 frames, reference image and a test beat through Whisper | Frame pixel-identical. Masking "red" changed the cube's color; masking "wooden" barely changed the frame |
| Wan 2.1 Fun Camera 1.3B: start image, zoom-in camera path | Frame pixel-identical, with maps and labels. The start image sets the frame, so the word test says nothing |
| Causal Forcing (Wan 2.1, frame by frame): 4 steps | Frame pixel-identical. Its frame-by-frame path passes no options to attention, so maps come through each module's attention slot, named `call:N` |
| SCAIL-Preview (Kijai's ComfyUI conversion) and SCAIL-2 `int8`: reference image and pose | Frames pixel-identical. Queries run the reference frame, the video, then pose tokens, and maps now cover the video alone. The reference image sets the frame, so the word test says nothing |
| Wan 2.2 Animate 14B: reference image and pose, Move mode | Frame pixel-identical, with maps and labels. The reference sets the frame, so the word test says nothing |
| Wan Dancer global 14B `fp8`: reference image and a test beat | Frame pixel-identical, with maps and labels. Without its LoRA at this short test the frames are noisy, so the word test says nothing |
| AuraFlow 0.3 (fal): 768, guidance 3.5, 20 steps | Pixel-identical. Its blocks are named `call:N`. Masking "wooden" gave a white table; masking "red" broke the image into streaks |
| Cosmos Predict 2 2B text to image: 768, guidance 4, 20 steps | Pixel-identical; maps on its cross-attention calls. Masking "red" gave a black cube, and masking "wooden" gave a white table |
| Flux 3 Klein `int8`, Qwen3-VL 4B `int8`: 768, 16 steps | Pixel-identical, maps on its single blocks with labels. Masking "red" or "wooden" barely changed the image, so labels stay unchecked |
| Stable Cascade stages C and B: 768, 20 and 10 steps | Pixel-identical with the nodes on either stage. Stage C receives its prompt as `clip_text`, which is now read, and maps work; masking words barely changed the image, so labels stay unchecked. Stage B conditions on a pooled summary, so it has no word map |
| SenseNova U1.5 (t8star's ComfyUI conversion): 512, 16 steps | Pixel-identical, layers recorded. Its text is token IDs inside the model, so there is no word map |
| LTX Video 2B 0.9.5: 480x320, 9 frames, 20 steps | Frame pixel-identical, maps on its cross-attention calls. Masking "red" gave purple cubes on the wooden table; masking "wooden" kept the red cube on a table without wood |
| Mochi preview `fp8`: 480x320, 7 frames, 20 steps | Frame pixel-identical. It drops its prompt's padding before attention, so the bridge counts the tokens it keeps (`num_tokens`). Masking "red" gave cream cubes; masking "wooden" changed little |
| Cosmos 1 7B Text2World: 1280x704, 33 frames, 20 steps | Frame pixel-identical, maps on its cross-attention calls (odd `call:N`). At 704x448 and 9 frames it draws noise even without the nodes, so it was run at its native size. Masking "wooden" gave a grey table with the red cubes kept; masking "red" broke up the scene |
| Kandinsky 5 Lite text to video: 480x320, 9 frames, 20 steps | Frame pixel-identical. Its encoder drops the template up to the user header, now recognised for labels. Masking "red" turned one face of the cube orange; masking "wooden" changed little |
| CogVideoX through VOID pass 1 (Netflix, Comfy-Org template): 480x320, 9 frames, a mask over a figure | Frame pixel-identical, maps on its joint attention. The prompt describes the scene after removal; masking "table" removed the table it painted in, while masking "wooden" changed little |
| ACE-Step 1 3.5B: 8 seconds, instrumental tags, 20 steps | The denoiser's output latent is bit-identical with the nodes on. In one node run of three, the decoded audio differed from the plain runs in 4 of 704,512 samples (at most 64 of 32,768); the latent did not, so that comes from ComfyUI's audio decode. Maps with labels on the tags. Masking "guitar" or "piano" changes almost every sample, but whether the right instrument went has not been checked by ear, so labels stay unchecked |
| MiniMax Music 3 `fp16`, its pruned `int8` encoder: 8 seconds, 20 steps | Decoded audio identical with the nodes on, layers recorded. Its attention bypasses ComfyUI's override, so there is no map |
| YuE 2 3B `int8`: 8 seconds without a score, 32 steps | Decoded audio identical with the nodes on, layers recorded. Its attention bypasses ComfyUI's override, so there is no map |
| Hunyuan3D 2 `fp16` and 2.1: one reference image, 3,072 and 4,096 shape tokens, 20 steps | The sampler's latent is bit-identical with the nodes on, layers recorded. Shape tokens are not an image, so there is no map |
| TripoSplat `fp16` (VAST-AI), DINOv3-H: one masked image, 20 steps | Its latent is a nested tensor that cannot be saved, so the decoded splat (`.ply`) was compared: byte-identical with the nodes on, layers recorded. No map: its condition has more tokens than a map allows |
| Trellis 2 `int8`, DINOv3-L: one image, structure stage, 12 steps | The 16x16x16 structure latent is bit-identical with the nodes on. Maps on its cross-attention calls (odd `call:N`) show each voxel slice reading the 1,029 image-feature tokens; there are no words to label |
| SeedVR2 | Layers recorded; no map, because it bypasses the hook and conditions on fixed embeddings |

## Language and embedding models in the standalone app

Transformers folders from Hugging Face, put in ComfyUI's `text_encoders` folder and loaded with Load on CUDA (RTX 5090 laptop, 24 GB). Each ran a layer capture, an attention probe and a 12-token response.

| Model | What ran | Result |
|---|---|---|
| Qwen3-8B | bfloat16, since float32 would not fit the GPU | Next token " Paris", 32 layers recorded, attention probe, response in 2 s (57 s in float32, spilling into shared memory) |
| Qwen3-14B | bfloat16; 27.5 GB still spills past the GPU | Next token " Paris", response in 56 s. Unloading used to copy the model to the CPU first and could exhaust RAM; it now frees the weights in place |
| Gemma 3 4B it | float32, as `Gemma3ForConditionalGeneration` loaded as a causal language model | Next token " Paris", the first 32 of its 34 language layers recorded; the image tower is not run by a text prompt and is no longer picked by default |
| Gemma 4 E4B it | bfloat16, as `Gemma4ForConditionalGeneration` | Its tokenizer leaves off `<bos>`, without which the model answers " France" and repeats itself; raw prompts now get it, giving " Paris". The model bar and Layer list show its 42 language layers, not the audio and vision towers |
| EmbeddingGemma 300M | Sentence Transformers package, new embedding task | 24 layers and a 768-value vector from its own mean pooling, two dense layers and normalisation. The vector matches the Sentence Transformers library's to within 5e-7; a paraphrase scores 0.73 cosine against 0.23 for an unrelated sentence |

Recursive's Flint was not run: its Transformers modelling code is not published, and the GGUF builds need a GGUF runtime this app does not have.
