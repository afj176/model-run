import { useEffect, useState } from "react";
import { FolderOpen, Play, UploadSimple } from "@phosphor-icons/react";
import { startStage } from "./libraryData";

/** Seconds since `running` became true, ticking once a second; 0 when not running. */
function useElapsed(running) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) {
      setSeconds(0);
      return undefined;
    }
    const started = Date.now();
    const timer = setInterval(
      () => setSeconds(Math.floor((Date.now() - started) / 1000)),
      1000,
    );
    return () => clearInterval(timer);
  }, [running]);
  return seconds;
}

/** What to do next before there is a capture: choose a model, wait for it, or press Run. */
export default function StartPanel({
  state,
  loadingName,
  onOpenLibrary,
  onRun,
  onImport,
}) {
  const stage = startStage(state);
  const seconds = useElapsed(stage === "loading");
  const loaded = state?.loaded;
  if (stage === "comfyui")
    return (
      <div className="start-panel">
        <h3>Run a model through ComfyUI</h3>
        <ol className="start-steps">
          <li>
            <b>A text model</b>
            <span>
              Open the library, choose one marked Writes text or Reads prompts,
              then write a prompt and press Run.
            </span>
            <button className="primary" onClick={onOpenLibrary}>
              <FolderOpen aria-hidden="true" /> Open the model library
            </button>
          </li>
          <li>
            <b>An image, video or audio workflow</b>
            <span>
              Get models › By workflow: press Run on one that is ready with your
              models. Or, on the ComfyUI canvas, use Observe this workflow in
              the Model Run sidebar.
            </span>
          </li>
        </ol>
        <button className="text-button" onClick={onImport}>
          <UploadSimple aria-hidden="true" /> Or import a capture you exported
        </button>
      </div>
    );
  if (stage === "loading")
    return (
      <div className="start-panel" role="status" aria-live="polite">
        <span className="start-spinner" aria-hidden="true" />
        <h3>Loading {loadingName || "the model"}…</h3>
        <p>
          {seconds} s · Reading weights into memory. Large models take longer,
          and Cancel in the toolbar stops it.
        </p>
      </div>
    );
  if (stage === "ready")
    return (
      <div className="start-panel">
        <h3>{loaded.name} is ready</h3>
        <p>
          {["image", "audio"].includes(loaded.task)
            ? "Choose a file above, then press Run to see what the model makes of it."
            : "Write a prompt above and press Run. It captures the layers, the response and attention in one go."}
        </p>
        <button className="primary" onClick={onRun}>
          <Play aria-hidden="true" /> Run
        </button>
      </div>
    );
  return (
    <div className="start-panel">
      <h3>Look inside a model as it runs</h3>
      <ol className="start-steps">
        <li>
          <b>Choose a model</b>
          <span>
            One marked Runs here, or a starter from Get models › Runs here.
          </span>
          <button className="primary" onClick={onOpenLibrary}>
            <FolderOpen aria-hidden="true" /> Open the model library
          </button>
        </li>
        <li>
          <b>Write a prompt and press Run</b>
          <span>
            Model Run records each layer, what the model wrote and where it
            looked.
          </span>
        </li>
      </ol>
      <button className="text-button" onClick={onImport}>
        <UploadSimple aria-hidden="true" /> Or import a capture you exported
      </button>
    </div>
  );
}
