/** What Model Run adds inside ComfyUI, shown from the standalone app. */
const INSIDE = [
  [
    "Your ComfyUI model files",
    "Single-file checkpoints and diffusion models, fp8 and int8 text encoders and GGUF: everything ComfyUI loads, run from Model Run.",
  ],
  [
    "ComfyUI's workflows, observed",
    "Run a ready workflow and see each denoising step, where every prompt word lands in attention, the VAE stages and the image it made.",
  ],
  [
    "Image, video and audio models",
    "Flux, Qwen Image, HiDream, LTX, Wan, Stable Audio and the rest, through ComfyUI's own loaders and memory management.",
  ],
  [
    "Back and forth with the canvas",
    "Observe this workflow sends whatever is open in ComfyUI to Model Run; the workflow Model Run ran stays on the canvas to change.",
  ],
  [
    "Get models by workflow",
    "Every ComfyUI template, nearest to ready first, with its missing files in one step.",
  ],
];

export default function Unlock() {
  return (
    <div className="unlock-panel">
      <p>
        This is Model Run on its own. It runs complete Hugging Face packages:
        language, embedding, image-classifier, speech and Diffusers models.
        Inside ComfyUI it runs everything ComfyUI can, too.
      </p>
      <h3>Unlocked inside ComfyUI</h3>
      <ul>
        {INSIDE.map(([title, text]) => (
          <li key={title}>
            <b>{title}</b>
            <span>{text}</span>
          </li>
        ))}
      </ul>
      <p className="muted">
        Captures, Compare, reports, import and export, and the Runs here
        packages work the same in both.
      </p>
      <h3>Start Model Run inside ComfyUI</h3>
      <p>
        From this folder, point the start script at your ComfyUI folder (the one
        holding <code>main.py</code>). It starts ComfyUI with Model Run loaded
        and opens both; your ComfyUI folder is not changed.
      </p>
      <pre>
        <code>.\Start-ModelRun.ps1 -ComfyDir "C:\path\to\ComfyUI"</code>
      </pre>
      <pre>
        <code>./start-model-run.sh --comfy-dir /path/to/ComfyUI</code>
      </pre>
      <p className="muted">
        Already run ComfyUI yourself? Put this repository in its{" "}
        <code>custom_nodes</code> folder and open Model Run from ComfyUI’s Model
        Run sidebar.
      </p>
    </div>
  );
}
