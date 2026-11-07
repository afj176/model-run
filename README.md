# Model Run

Model Run shows what a model actually does when it runs. Load a model, run it, and explore
the tensors it produced: layer outputs, attention, the next-token distribution, denoising
steps, where an image reads its prompt, and what changes when you remove a head or a word.

It runs two ways, with the same interface:

- **Standalone**, as a local web app that loads language, embedding, image-classifier,
  speech and Diffusers models itself.
- **Inside ComfyUI**, as custom nodes that observe ComfyUI's own runs. ComfyUI keeps
  loading, sampling and memory; Model Run records what passes through.

Everything stays on your computer: no CDN, no remote inference, no remote model code, and
nothing downloads without a click.

Every number in the interface is measured from the run. Nothing is inferred from
architecture names, missing data is never drawn as zero, and every trace lists its own
limitations. [Tested models](#tested-models) lists what has been checked, on which machine,
and how.

## Contents

- [Run standalone](#run-standalone)
- [Run inside ComfyUI](#run-inside-comfyui)
- [Get models](#get-models)
- [Explore a capture](#explore-a-capture)
- [Tested models](#tested-models)
- [Privacy and safety](#privacy-and-safety)
- [Development](#development)
- [Python API](#python-api)

## Run standalone

Model Run runs on Windows, macOS and Linux with Python 3.10 or later (use `python3` where
`python` is Python 2 or missing, as on many Macs and Linux systems). Install it into the
environment that will load your models:

```bash
python -m pip install -e ".[transformers,diffusers,media,comfy-paths]"
```

Then start the server, pointing it at a ComfyUI folder if you have one:

```bash
python -m model_run.server --comfy-dir /path/to/ComfyUI
```

Open **http://127.0.0.1:8766/**. The server listens on this computer only.

| Option | What it does |
| --- | --- |
| `--comfy-dir DIR` | A ComfyUI folder: its `models` folder, its `extra_model_paths.yaml`, Comfy Desktop's shared folders, and the model lists ComfyUI ships. |
| `--models-dir DIR` | A ComfyUI `models` folder on its own. |
| `--extra-paths FILE` | An `extra_model_paths.yaml`; may be repeated. |
| `--port N` | Default 8766. |

With no folders the app still imports and replays traces. It only looks where you point
it; Hugging Face, Ollama and other caches are not searched.

### Start it in the background

Two scripts start the server in the background, wait until it answers and print its process
ID. Both refuse a port that is already in use and never stop another process; stop the
printed PID to end Model Run. Logs go to `.tmp/app-server.stdout.log` and
`.tmp/app-server.stderr.log`.

Windows (PowerShell 5.1 or 7):

```powershell
.\Start-ModelRun.ps1 -ComfyDir C:\path\ComfyUI
```

`-ModelsDir`, `-ExtraPaths`, `-Port` and `-Python` are optional.

macOS and Linux:

```bash
./start-model-run.sh --comfy-dir /path/to/ComfyUI
```

`--models-dir`, `--extra-paths` (repeatable), `--port` and `--python` (default `python3`) are
optional. Stop it with `kill PID`.

Both scripts and the app were run on Windows 11 and on Ubuntu 24.04 under WSL2. macOS has not
been tested yet.

### Devices

The standalone app runs models on the CPU or an NVIDIA GPU (CUDA). On a Mac it runs them
on the CPU; Apple GPUs (MPS) are not used yet. Inside ComfyUI, models run on whatever device
ComfyUI uses, because ComfyUI does the loading and Model Run only records.

### What it loads

Put a model folder in one of ComfyUI's model folders (a Transformers folder goes in
`text_encoders` or `clip`) and it appears in the library.

| Kind | Notes |
| --- | --- |
| Causal language models (Transformers, safetensors) | One forward pass of up to 128 tokens, a greedy response of 32 to 256 tokens, attention probes. Raw or chat-template input. Image-and-text models that Transformers also loads as language models, such as Gemma 3 and 4, load as text. |
| Embedding models (Sentence Transformers packages) | Layers and the sentence vector from the package's own pooling, dense and normalise steps. |
| Image classifiers | Top labels, per-layer outputs on the patch grid, and the class token's attention over the image. |
| Whisper | Transcript with per-token confidence, the spectrogram it heard, playback, and each token's attention over audio time. Uses the first 30 seconds. |
| Diffusers text-to-image pipelines | Denoising steps, cross-attention maps and a decoded preview. |

Models load in float32. On CUDA, a 16-bit checkpoint that would not fit in float32 loads in
its own precision instead, and the model bar and the trace say so. One model is loaded at a
time; loading another unloads the first. Image and speech files you choose stay on this
computer.

Image models need Pillow. Speech needs scipy, plus soundfile or PyAV (ComfyUI ships PyAV).

## Run inside ComfyUI

Clone or copy this repository into ComfyUI's `custom_nodes` folder and restart ComfyUI. The
nodes need nothing beyond what ComfyUI installs. Open **Model Run → Open Model Run** from the
menu, or go to `/model-run/` on your ComfyUI address. The app there reads ComfyUI's live
model folders, including ones added by custom nodes.

| Node | Where it goes | What it records |
| --- | --- | --- |
| **Model Run · Observe** | Between a MODEL output and its sampler | The denoiser's blocks and output for each call, denoising previews, attention maps and token masking. |
| **Model Run · Attach Output** | After VAE Decode | Pins the decoded images or frames to the observed run. |
| **Model Run · Text Encode** | In place of CLIP Text Encode | The text encoder's layers and tokens. |
| **Model Run · Generate Text** | In place of Generate Text | The prompt through the language model's layers, and for each written token its probability and the tokens it nearly wrote. Same text as Generate Text. |
| **Model Run · VAE Decode** / **VAE Encode** | In place of VAE Decode / Encode | What each VAE block produced. |
| **Model Run · Observe (WanVideoWrapper)** | Between WanVideo Model Loader and WanVideo Sampler | The same as Observe, for Kijai's ComfyUI-WanVideoWrapper. |
| **Model Run · Observe (HunyuanVideoWrapper)** | Between HunyuanVideo Model Loader and HunyuanVideo Sampler | The same, for Kijai's ComfyUI-HunyuanVideoWrapper. It records only the queued prompt it runs in. |

The nodes are built to leave the output unchanged, and every model under
[Tested models](#tested-models) was checked against the same graph without them.

### Observe settings

- **modules**: leave empty to record every block and the final output, or list module paths
  one per line (for example `double_blocks.0`). Paths differ by model; record everything
  once to see them.
- **max_calls** and **elements** bound the capture. A call is one denoiser invocation, which
  is not always one sampling step (guidance can call twice).
- **attention_block** maps where the image reads its prompt in one block: `double:N` or
  `single:N` for DiT models, `input:N` / `middle:0` / `output:N` for UNets, `audio:N` for
  an audio stream, and `call:N` for models that do not name their blocks. Wire **clip** and
  **prompt** to label the tokens.
- **ablate_tokens** masks token positions out of every attention call (for example `4, 5`).
  Compare with a run that leaves it empty to see what those tokens did.
- **output_selectors** picks a nested output, such as `{"transformer_blocks.0": [1]}` for
  LTX's audio stream.

Observe has been checked before and after ControlNets (classic, DiT and model-patch) and
IP-Adapter, and with GGUF models loaded through ComfyUI-GGUF.

The ComfyUI app is served on ComfyUI's own address. Keep that host local, or put it behind
the same access controls you use for ComfyUI.

## Get models

**Get models** searches the model lists ComfyUI ships (its workflow templates and ComfyUI
Manager's list) and, when you ask, Hugging Face, then installs into ComfyUI's folders.
**By workflow** installs everything a template is missing.

- Nothing downloads without a click. The confirmation shows the size, source, destination
  and free space.
- Only `https` files from Hugging Face in `.safetensors`, `.sft` or `.gguf` form install.
  Pickle formats such as `.ckpt`, `.pt` and `.bin` can run code when loaded, so they are
  listed but refused. GGUF files load through the ComfyUI-GGUF custom node.
- Each file is checked against its published SHA-256 where there is one, never replaces an
  existing file, and resumes if stopped.
- For a gated model, accept its licence on Hugging Face and log in with its CLI. The app uses
  that login, never stores it, and sends it only to huggingface.co.
- Inside ComfyUI, installing works only from the computer running ComfyUI.

## Explore a capture

- **Run** captures a whole session for a text model: layer outputs, the greedy response, and
  an attention probe at the chosen layer and head that records every layer. Diffusion runs
  record denoising steps.
- **Run View** shows the model as a stack of slabs, one per block or step, tinted by measured
  activity, with the selected event's values as a field of bars.
- **Tensor Exploration Table** shows attention and activations as a navigable 3D matrix with
  a linked heatmap, 32 × 32 values per window. **Tensor Volume** stacks a tensor's recorded
  maps along one axis. Frames and channels can be stepped through and played.
- **Layer Stack** and the timeline select a layer; the inspector shows paths, shapes, dtypes
  and values.
- **What it predicted** shows the next-token distribution and a logit lens: the top token
  after every layer, and where the answer first took the lead.
- **Response** shows each written token shaded by its probability, its runners-up, its logit
  lens and the attention from that position to every earlier token. A reasoning model's
  `<think>` block is shown separately.
- **Head profile and gallery** label each attention head from its weights (attention sink,
  previous-token, self-attending, focused, diffuse) and browse every head of a layer.
- **Ablate** and **Sweep** zero one head, or every head in turn, for a full forward pass and
  record how the next token moves.
- **Compare** puts two traces side by side. Edit one word and **Run B** reruns the same
  capture; Model Run reports which tokens changed, whether the answer held, and where the
  layers diverged. **Patch A → B** copies A's residual stream into B one layer at a time to
  find where the answer is carried.
- **Where the prompt is read** (diffusion) ranks prompt tokens by how much the image reads
  them, with a map per token over the preview image, per frame for video. **Run without a
  token** reruns with tokens masked and shows the pixel difference.
- **Report** downloads a self-contained HTML report of the open trace, built in the browser,
  with every limitation recorded in the trace.

Attention shows where a layer read, not what caused an output. Ablation and patching are
interventions, not natural behaviour, and their effects do not add up across heads.

**History.** The last 16 runs (up to 64 traces or 256 MiB) stay in memory; pin up to 8 runs
to keep them longer. Nothing is written to disk unless you export it, so restarting clears
the history. Export and import use validated JSON up to 16 MiB per trace.

## Tested models

Every model below ran on real weights with Model Run's nodes in the graph, and the output
matched the same graph without them: pixel for pixel for images and video frames, byte for
byte for audio, bit for bit for 3D latents. Two plain runs were compared first, since
ComfyUI's first run after a model load can differ on its own. Where a model reads a prompt,
its attention maps were checked by masking words such as "red" and "wooden" and looking at
the result. A model that is not listed should still load and record its layers: anything
Model Run does not understand is reported, never guessed.

### Test machine

| | |
| --- | --- |
| OS | Windows 11 Pro |
| CPU | Intel Core Ultra 9 275HX, 24 cores |
| RAM | 63 GB |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, 24 GB VRAM |
| Software | ComfyUI 0.42.1 and 0.43.2 (it updated during testing), PyTorch 2.12 with CUDA 13.0 in ComfyUI and CUDA 12.8 in the standalone app |

Many of these models do not fit in 24 GB of VRAM alongside their text encoders and VAEs.
ComfyUI loaded every model with its dynamic VRAM loading, which stages weights in system
RAM and moves them to the GPU as they are needed, and offloaded each model to the CPU when
the next one loaded. In the standalone app, Qwen3 14B (27.5 GB) and Qwen3 8B in float32
did not fit and spilled into shared GPU memory, which is system RAM: a 12-token response
took 56 and 57 seconds, against 2 seconds for Qwen3 8B in bfloat16. That is why the app
loads a 16-bit checkpoint in its own precision when float32 would not fit. None of these
times are benchmarks, and a machine with less memory will run the larger models slower or
not at all.

### Standalone app

| Model | Attention map |
| --- | --- |
| Qwen3 0.6B, 8B and 14B | Yes (attention probe) |
| SmolLM2 135M | Not checked (layer capture only) |
| Gemma 3 4B it | Yes (attention probe) |
| Gemma 4 E4B it | Yes (attention probe) |
| EmbeddingGemma 300M | No probe for embedding models |
| ViT base (google/vit-base-patch16-224) | Yes, over the image |
| Whisper tiny | Yes, over audio time |
| Tiny Stable Diffusion test pipeline (Diffusers) | Not checked (denoising steps only) |

### ComfyUI: image

| Model | Attention map |
| --- | --- |
| SD 1.5 | Yes, labelled |
| SD 2.1 768 | Yes, labelled |
| SD 2.1 unCLIP-H | Yes, labelled |
| SDXL 1.0 | Yes, labelled |
| SDXL Refiner | Yes, labelled |
| Illustrious XL v0.1 (SDXL finetune) | Yes, labelled |
| Segmind Vega | Yes, labelled |
| SSD-1B | Yes, labelled |
| KOALA 700M and 1B | Yes, labelled |
| InstructPix2Pix (SD 1.5 and SDXL) | Yes, labelled |
| SD x4 upscaler | Yes, labelled |
| Lotus depth | Yes; it reads one fixed embedding, no words |
| SD3.5 Large | Yes, labelled |
| Flux 1 Dev | Yes, labelled |
| Flux 1 Schnell | Yes, labelled |
| Flux 1 Fill | Yes, labelled |
| Flux2 Klein | Yes, labelled |
| Flux 3 Klein | Yes, labelled |
| Chroma1 HD | Yes, labelled |
| Chroma Radiance | Yes, labelled |
| PixArt Alpha | Yes, labelled |
| PixArt Sigma | Yes, labelled |
| AuraFlow 0.3 | Yes, labelled |
| Hunyuan DiT 1.2 | Yes, labelled |
| Hunyuan Image 2.1 and its refiner | Yes, labelled |
| HiDream I1 Dev | Yes, labelled |
| HiDream-O1 | Yes, labelled |
| Qwen-Image 2.1 | Yes, labelled |
| Qwen-Image Edit 2511 | Yes, numbered |
| Z-Image | Yes, labelled |
| Anima | Yes, labelled |
| MOSS Image | Yes, labelled |
| Krea 2 | Yes, labelled |
| MiniMax H3 (image and video) | Yes, labelled |
| Lens Turbo | Yes, labelled |
| Boogu Turbo | Yes, labelled |
| Ideogram 4 | Yes, labelled |
| OmniGen 2 | Yes, labelled |
| Ernie Image Turbo | Yes, labelled |
| PixelDiT 1.3B | Yes, labelled |
| Mage Flow Turbo | Yes, labelled |
| JoyImage Edit | Yes, labelled |
| Fibo 2 Edit Turbo | Yes, numbered |
| LongCat Image | Yes, labelled |
| Kandinsky 5 Image | Yes, labelled |
| Ming Image 0.1 Design | Yes, numbered (its text keys are learned queries) |
| Cosmos Predict 2 2B (text to image) | Yes, labelled |
| Stable Cascade (stages C and B) | Stage C labelled; stage B has no prompt to map |
| PiD (Flux latents to pixels) | Yes; empty prompt, no words |
| SenseNova U1.5 | No; its text is token IDs inside the model |

### ComfyUI: video

| Model | Attention map |
| --- | --- |
| LTX 2.5 (video and audio) | Yes, labelled |
| LTX 2.3 dev | Yes, labelled |
| LTX Video 2B 0.9.5 | Yes, labelled |
| Wan 2.1 T2V 1.3B | Yes, labelled |
| Wan 2.1 I2V 14B | Yes, labelled |
| Wan 2.1 Fun Control 1.3B | Yes, labelled |
| Wan 2.1 Fun Camera 1.3B | Yes, labelled |
| Wan VACE 1.3B | Yes, labelled |
| Wan 2.2 TI2V 5B | Yes, labelled |
| Wan 2.2 Fun Camera 14B | Yes, labelled |
| Wan 2.2 S2V | Yes, labelled |
| Wan 2.2 Animate 14B | Yes, labelled |
| Wan Animate 2 | Yes, labelled |
| Wan Dancer 14B | Yes, labelled |
| HuMo 17B | Yes, labelled |
| Causal Forcing (Wan 2.1) | Yes |
| SCAIL-Preview and SCAIL-2 | Yes, labelled |
| Hunyuan Video | Yes, labelled |
| Hunyuan Video I2V | Yes, labelled |
| SkyReels V1 Hunyuan I2V | Yes, labelled |
| Hunyuan Video 1.5 (T2V and I2V) | Yes, labelled |
| Hunyuan Video 1.5 1080p super-resolution | Yes, labelled |
| Mochi preview | Yes, labelled |
| Cosmos 1 7B (Text2World and Video2World) | Yes, labelled |
| Cosmos Predict 2 2B Video2World | Yes, labelled |
| Kandinsky 5 Lite (text to video) | Yes, labelled |
| CogVideoX (VOID pass 1) | Yes, labelled |
| SVD XT | Yes; it reads one image token, no words |
| SeedVR2 (video restoration) | No; its attention bypasses ComfyUI's hook |

### ComfyUI: audio

| Model | Attention map |
| --- | --- |
| Stable Audio Open | Yes, labelled |
| Stable Audio 3 Medium | Yes, labelled |
| ACE-Step 1 3.5B | Yes, labelled |
| ACE-Step 1.5 | Yes, labelled |
| MiniMax Music 3 | No; its attention bypasses ComfyUI's hook |
| YuE 2 3B | No; its attention bypasses ComfyUI's hook |

### ComfyUI: 3D

| Model | Attention map |
| --- | --- |
| Hunyuan3D 2, 2 mini and 2.1 | No; shape tokens are not an image |
| TripoSplat | No; its condition is too long for a map |
| Trellis 2 | Yes, over image features (no words) |

### ComfyUI: language models through Model Run · Generate Text

Gemma 4 E4B, Gemma 3 4B, Qwen3 4B, Qwen3.5 4B, Qwen3.5 9B and Qwen3.8 27B. Each wrote the same text as ComfyUI's Generate Text, and
every written token was the top token in Model Run's record.

### ComfyUI: add-ons and custom node packs

| Model | Attention map |
| --- | --- |
| ControlNet v1.1 canny (SD 1.5) | Yes, labelled |
| SD3.5 Large canny ControlNet | Yes, labelled |
| Z-Image Turbo Fun Union ControlNet | Yes, labelled |
| IP-Adapter and IP-Adapter Plus (SD 1.5, ComfyUI_IPAdapter_plus) | Yes, labelled |
| GGUF: Flux 1 Dev Q4_K_S with a T5 Q5_K_M encoder (ComfyUI-GGUF) | Yes, labelled |
| GGUF: Qwen-Image Q4_K_S | Yes, labelled |
| GGUF: Wan 2.2 TI2V 5B Q8_0 | Yes, labelled |
| Kijai's ComfyUI-WanVideoWrapper (Wan 2.1 T2V 1.3B) | Yes, labelled |
| Kijai's ComfyUI-HunyuanVideoWrapper (HunyuanVideo 720 cfg-distilled) | Yes, labelled |

Not tested, because no file that ComfyUI can load was found: Stable Zero123, SD 2.1
unCLIP-L, SV3D, CogVideoX T2V and I2V (outside the VOID pipeline) and Wan 2.1 FlowRVS.

## Privacy and safety

- The standalone server binds to loopback, rejects cross-origin requests and requires a
  session token.
- Models load from local safetensors with remote code disabled. Pickle files are never
  deserialized, even for inspection.
- Traces contain the decoded prompt, and speech traces contain the audio. Exports can
  contain private input.
- Importing a trace never runs model code.

## Development

```powershell
python -m unittest discover -s tests -p "test_*.py"
cd frontend
npm ci
npm test
npm run build
npm run package
```

The built front end lives in `model_run/web`, so rebuild and package after changing
`frontend/src`. Tests use generated fixtures and never need personal model paths; tests
for optional packages skip when those packages are missing. Run the backend tests under
ComfyUI's Python as well when changing the nodes.

| Tool | Purpose |
| --- | --- |
| `tools/check_comfy_families.py` | Run from a ComfyUI folder: builds every model family ComfyUI knows at tiny size and checks that Observe leaves the output unchanged, finds layers and maps attention. Re-run after ComfyUI updates. |
| `tools/verify_comfy_workflow.py` | Compares a real Flux2 workflow's latents with and without Observe. |
| `tools/verify_app_http.py` | Drives the standalone app over HTTP with a real checkpoint. |
| `tools/package_release.py` | Builds the custom-node ZIP from an allowlist; never bundles models or caches. |
| `tests/real_*_smoke.py` | Opt-in checks against a real Qwen3 or Diffusers checkpoint, given its folder. |

Command-line tools: `python -m model_run ARTIFACT` prints a file's metadata,
`python -m model_run.catalog_cli --models-dir ROOT` lists a model folder, and
`python -m model_run.run_text MODEL_DIR --module MODULE` captures one module on text
from stdin.

`frontend/AGENTS.md` holds implementation notes for coding agents.

## Python API

The app is built on bindings you can use directly on a model you already loaded.

```python
from model_run.adapters import default_registry

adapter = default_registry().bind("pytorch.module", loaded_module)
report = adapter.negotiate(["module_tree", "module_outputs"])  # every refusal comes with a reason
output, trace = adapter.capture(inputs, ["selected.module"], max_tensor_elements=1024)
```

| Binding | Accepts | Operations |
| --- | --- | --- |
| `pytorch.module` | Any loaded PyTorch module | Module tree and bounded module outputs |
| `transformers.causal_lm` | A Transformers decoder-only model | Also next-token probabilities and attention probes |
| `diffusers.pipeline` | A Diffusers pipeline with step callbacks | Bounded denoising latents per step |

- Captured tensors are bounded float32 prefixes, not summaries. A bad output selector raises
  an error, and Model Run's hooks are always removed; existing hooks are kept.
- Bindings never move a model or switch it to evaluation mode, and refuse to capture a
  module in training mode.
- `model_run.sessions.ObservationSession` runs one observation at a time with cooperative
  cancellation and a timeout, checked at forward boundaries, module hooks and step
  callbacks. Neither interrupts a running GPU kernel or a model load.
- `model_run.host.ModelHost` registers several bindings and runs one observation across all
  of them, with release callbacks for unloading.

Model Run does not train models, convert formats or load GGUF itself; GGUF models are
observed when ComfyUI-GGUF loads them in ComfyUI.
