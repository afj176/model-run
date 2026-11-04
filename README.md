# Model Run

A local model observatory: inspect model artifacts, explore runtime structure in
3D, and replay measured internal tensors. The same interface runs as a standalone
application and inside ComfyUI. No CDN, remote inference, automatic downloads or
remote model code is required.

## Run standalone

Python 3.10+ is required. Install into the environment that will load your models:

```powershell
python -m pip install -e ".[transformers,diffusers,comfy-paths]"
python -m model_run.server --models-dir C:\path\ComfyUI\models --extra-paths C:\path\extra_model_paths.yaml
```

Open **http://127.0.0.1:8766/**. Either path option can be omitted; without paths,
trace import/replay remains available. The app discovers only the paths supplied
at startup. Hugging Face, Ollama and Unsloth caches are not product defaults.

On Windows, `Start-ModelRun.ps1 -ModelsDir C:\path\ComfyUI\models` starts a hidden
background process, checks HTTP readiness and prints its PID. `-ExtraPaths`,
`-Python`, and `-Port` are optional. It refuses an occupied port and never stops an
existing process. Stop the printed PID to end that instance; traces in memory are
discarded. The HTTP server binds exclusively to loopback.

### Model library

- **Inspect** reads bounded metadata. Unsupported formats remain listed without
  deserialization. Safetensors/index and optional ONNX readers are available.
- **Explore** builds a spatial tensor-name namespace from safetensors headers.
  This works without loading weights, including larger image/video/audio models.
  Namespace groups are explicitly distinct from runtime modules and execution.
- **Load** is available for declared local causal-LM packages and local Diffusers
  packages. Choose CPU or CUDA. Only one application-owned model is resident;
  loading another first unloads the previous one. This is not a VRAM reservation.
- Causal models use local safetensors and remote code disabled. Text captures are
  one forward pass, at most 128 input tokens, with raw next-token softmax results.
  Raw prompt and tokenizer chat-template input modes are explicit.
- Diffusers capture requires a pipeline declaring the latent callback interface
  and prompt/output inputs. The app supports bounded text-to-image inputs and a
  decoded PNG preview where returned. Other tasks use native ComfyUI execution or
  the Python adapter APIs; a model's modality is not changed by a UI toggle.

## ComfyUI extension

Use the source checkout as a custom-node directory (or copy the release ZIP's
`model-run` directory into ComfyUI's `custom_nodes`). Restart ComfyUI. Dependencies
for native observation are supplied by ComfyUI; standalone Transformers/Diffusers
extras are not needed by the native observation node.

Open **Model Run → Open Model Run** from ComfyUI's menu, or visit
`http://127.0.0.1:8188/model-run/` using your actual ComfyUI address/port.
The extension reads the live `folder_paths` registry, including extra paths.

Connect **Model Run · Observe** between a MODEL-producing node and its sampler.

- An empty `modules` field observes the whole denoiser. Otherwise enter one
  loaded diffusion-module path per line, such as `double_blocks.0` for the tested
  Flux2 model. Paths vary by architecture; inspect the captured hierarchy first.
- Set `max_calls` and `elements` to bound capture. Calls are denoiser invocations,
  not necessarily sampling steps. The aggregate budget is 262,144 values.
- `output_selectors` optionally selects nested output tensors. For an LTX block,
  `{"transformer_blocks.0":[1]}` selects the audio tensor in its output tuple.
  Omission selects the first tensor, and the actual output path is recorded.
- The node clones the patcher, preserves an existing function wrapper, and uses
  scoped hooks. Downstream patcher clones share the observer's bounded budget.
  A new queued execution resets the observation budget.
- ComfyUI retains ownership of loading, conditioning, sampling, cancellation,
  outputs and memory management. There is no second model load in the extension.

The extension exposes captures on the ComfyUI server's existing network boundary.
Keep that host local or apply your normal ComfyUI authentication/access controls.
The standalone app rejects cross-origin requests and requires a session token.

## Explore a capture

The **Tensor Exploration Table** presents recorded attention weights and bounded
activation prefixes as a navigable 3D matrix, with a linked keyboard-accessible
heatmap and pink selection. Attention uses a 0–1 scale; signed activations use
a zero plane and a window-local symmetric linear scale. The table displays at
most 32 × 32 values per window; use the row/column controls to navigate. Unknown
axis semantics are named `dim N`. Leading tensor indices currently remain zero.
Values outside the retained prefix and non-finite values are not plotted; the
inspector distinguishes them. An attention mask is not inferred from zeros.
The selected text-model attention probe captures a complete single head, subject
to runtime support and the 128-token input bound. It may materialize other heads.
This is windowing, not aggregation or arbitrary full-tensor capture.

Use **3D Model** to orbit/pan/zoom and select component plates.
The selected layer expands two real child modules, prioritized by child count,
with up to four clickable child-module blocks each. These are containment views,
not inferred operation wiring or activation measurements. Four component shortcuts
also support keyboard selection in the inspector; the full tree remains available.
The Orbit/Pan/Zoom toolbar selects the primary drag behavior; Reset restores the camera.

**Layer Stack**
provides a keyboard-accessible list and text-model capture selection. **Tensor
Volume** changes the retained prefix's display layout; it is not a physical map of
neurons. Reset/fullscreen and optional auto-orbit are available. Reduced motion
disables auto-orbit. If WebGL fails, the list and inspector remain usable.

The inspector reports actual paths, shapes, dtypes, selected output paths through
trace metadata, values and prefix statistics. **Attention** runs a separate,
bounded attention probe for compatible text models. Attention is not causality.

### Insights

Every insight is computed from recorded values; nothing is inferred from
architecture names. The same panels render in the standalone app and inside
ComfyUI, lighting up wherever the trace carries the data.

- **Token labels.** Attention axes, the heatmap, the callout and the inspector
  use the decoded prompt tokens when the trace includes them.
- **What this head does.** Each attention matrix is profiled from its weights
  (first-token mass, previous-token mass, diagonal mass, mean row entropy) and
  labelled attention sink / previous-token / self-attending / focused / diffuse
  / mixed, with the strongest keys for the selected query.
- **What happens without this head.** From an attention capture, "Ablate"
  zeroes the selected head's output (its slice of the layer's output
  projection) for a full forward pass and records how the next-token
  distribution moves: top token before and after, probability of the original
  answer, entropy, and the KL shift in bits. "Sweep" does this for every head
  of the layer, one forward pass per head, and ranks them; heads whose removal
  flips the answer are marked. Results attach to the attention capture as a new
  revision, so they export with it. This is a zero ablation at every position,
  an intervention rather than natural behaviour, and effects of several heads
  are not additive. The request is refused when the prompt no longer matches
  the capture.
- **Where the prompt is read (diffusion).** Image-to-text attention maps
  measured from the exact query and key tensors the attention kernel receives,
  head-averaged and area-pooled to a grid of at most 32×32 (64 with the node
  setting). In the standalone app pick a `attn2` module under "Cross-attention
  map" before running a Diffusers UNet pipeline; the prompt is tokenised with
  the pipeline's tokenizer for labels. In ComfyUI set `attention_block` on
  Model Run · Observe (`double:0` or `single:3` for Flux, Qwen-Image and other
  joint-attention DiTs, `output:1` style for UNets) and optionally wire `clip`
  plus the prompt to label tokens. ComfyUI maps come from the attention
  override hook, so they include rotary embeddings and masks; up to four
  denoiser calls are recorded. The panel ranks tokens by how much the image
  reads them, shows each token's map (over the preview image when one exists)
  and, for joint attention, how much of each region's attention goes to text
  at all. Attention shows where text is read, not what caused a pixel.
- **Report.** The toolbar's Report button downloads a self-contained HTML
  insight report for the open trace: facts, prompt tokens, the head profile
  with heatmap and strongest keys, last-token attention, ablation results, the
  head gallery, per-layer series, the prediction story with logit lens,
  cross-attention maps for the most-read tokens, activation statistics with
  per-call deltas, and every limitation recorded with the trace. It is built
  in the browser from the trace alone, so it works identically in ComfyUI,
  contains no scripts, and prints cleanly.
- **Head gallery.** An attention probe now returns every head of the selected
  layer (rounded to 5 decimals, within a 262,144-value bound) so heads can be
  browsed without another forward pass, plus per-layer head-averaged summaries
  that draw the sparkline over the Layer / Ordinal Events timeline.
- **What the last token looked at.** The measured attention from the final
  query position to each key, averaged over every layer and head. An attention
  rollout estimate is recorded alongside it. Neither is a causal attribution.
- **What it predicted next.** The next-token distribution with its entropy and a
  confidence label, and a **logit lens**: the last-position residual stream
  after every layer, passed through the final norm and output head, showing the
  top token per layer and where the final answer first took the lead.
- **Activation insight.** RMS, peak-to-RMS ratio, near-zero share and sign
  balance for the captured prefix, with the largest values; traces with
  denoiser calls or diffusion steps also show the RMS change between
  consecutive calls of the same module.
- **Compare.** Any two retained traces: attention weights are subtracted cell by
  cell when shapes match, per-layer activation RMS is tabulated, and next-token
  predictions are set side by side.

Color/height use a symmetric linear or labeled signed-log1p scale. The renderer
shows at most 64 component groups and 2,048 expanded retained values. It also shows
the first 64 captured values as sign-colored context dots on each measured layer
plate (from its first retained event). The expanded tensor uses sign color,
normalized intensity and depth; its row/column grid is a display layout.
The inspector lists
the first 256 values and first 64 local parameters. Null represents a non-finite
sample, not zero. Missing captures are never colored as observed zero values.

The timeline replays recorded events; it does not claim real-time kernel timings
or reverse model execution. Capture history retains at most eight traces / 32 MiB.
Export/import uses validated JSON up to 16 MiB per trace. Nothing is saved
automatically. Text traces include decoded input tokens; exports can contain
private input-derived data. Importing a trace does not execute model code.

## Build and validate

Built assets are included under `model_run/web`. To rebuild:

```powershell
cd frontend
npm ci
npm run build
node scripts/package-app.mjs
```

Run fixture tests with `python -m unittest discover -s tests -q`. Optional runtime
dependencies must be installed for their tests. Tests use generated fixtures and
never require personal model paths. `tools/verify_app_http.py` and
`tools/verify_comfy_workflow.py` are opt-in real-checkpoint checks with explicit
arguments. The latter uses a bounded Flux2 workflow and compares observed versus
unobserved output latents.

See [VALIDATION.md](VALIDATION.md), [COMPATIBILITY_MATRIX.md](COMPATIBILITY_MATRIX.md)
and [reports/application-release-2026-11-04.md](reports/application-release-2026-11-04.md)
for exact evidence. A verified checkpoint/workload does not certify every variant.

## Python APIs and boundaries

[RUNTIME_ADAPTERS.md](RUNTIME_ADAPTERS.md) documents PyTorch, Transformers and
Diffusers bindings, `ObservationSession`, and `ModelHost`. The application uses
those contracts with one resource queue and explicit release callbacks. Capture
cancellation is cooperative; model loading and running accelerator kernels cannot
be forcibly interrupted. Noncontiguous prefix capture may allocate a reshape copy.

Metadata CLI: `python -m model_run ARTIFACT`. Catalog CLI:
`python -m model_run.catalog_cli --models-dir ROOT --extra-paths CONFIG.yaml`.
Local text CLI: `python -m model_run.run_text MODEL_DIRECTORY --module MODULE` reads
raw stdin. Fingerprinting is explicit and reads complete selected files.

This release does not execute GGUF/ONNX, infer operation graphs from weights,
train models, intervene on activations, or implement generic media preprocessing.
Image/video/audio native models are observed through their ComfyUI execution.
The renderer deliberately avoids the mockup's invented operation wiring and
illustrative activations.
