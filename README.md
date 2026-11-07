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
limitations. [COMPATIBILITY_MATRIX.md](COMPATIBILITY_MATRIX.md) records what has been
checked, on which models, and how.

## Contents

- [Run standalone](#run-standalone)
- [Run inside ComfyUI](#run-inside-comfyui)
- [Get models](#get-models)
- [Explore a capture](#explore-a-capture)
- [Privacy and safety](#privacy-and-safety)
- [Development](#development)
- [Python API](#python-api)

## Run standalone

Python 3.10 or later. Install into the environment that will load your models:

```powershell
python -m pip install -e ".[transformers,diffusers,media,comfy-paths]"
python -m model_run.server --comfy-dir C:\path\ComfyUI
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

On Windows, `Start-ModelRun.ps1 -ModelsDir C:\path\ComfyUI\models` starts the server in the
background, waits until it answers and prints its process ID (`-ExtraPaths`, `-Python` and
`-Port` are optional). It refuses a port that is in use and never stops another process.

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

The nodes are built to leave the output unchanged, and every run in the compatibility
matrix was checked against the same graph without them.

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
