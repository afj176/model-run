import { useState } from "react";
import { ArrowClockwise } from "@phosphor-icons/react";
import { api } from "./api";
import { formatBytes } from "./installData";
import {
  formatCount,
  groupArtifacts,
  groupOf,
  kindOf,
  withoutPackageFiles,
} from "./libraryData";

function InspectSummary({ report }) {
  const s = report.summary;
  const stacks = Object.entries(s?.block_stacks || {});
  return (
    <div className="inspect-summary">
      {s && (
        <dl>
          <dt>Parameters</dt>
          <dd>{formatCount(s.parameters)}</dd>
          <dt>Stored as</dt>
          <dd>
            {Object.entries(s.bytes_by_dtype)
              .map(([dtype, bytes]) => `${dtype} ${formatBytes(bytes)}`)
              .join(" · ")}
          </dd>
          {stacks.length > 0 && (
            <>
              <dt>Repeated blocks</dt>
              <dd>
                {stacks
                  .map(([name, count]) => `${name} × ${count}`)
                  .join(" · ")}
              </dd>
            </>
          )}
          <dt>Used by</dt>
          <dd>
            {s.workflows?.length
              ? s.workflows.join(" · ")
              : "No ComfyUI workflow template lists this file"}
          </dd>
        </dl>
      )}
      {report.config && (
        <dl>
          <dt>Declared type</dt>
          <dd>
            {[report.config.model_type, ...(report.config.architectures || [])]
              .filter(Boolean)
              .join(" · ") || "Not declared"}
          </dd>
        </dl>
      )}
      <details>
        <summary>
          {report.config ? "Full configuration" : "Full header"}
        </summary>
        <pre>{JSON.stringify(report, null, 2)}</pre>
      </details>
    </div>
  );
}

/** The library's "On this computer" tab: what is in ComfyUI's folders and what this app can do with each file. */
export default function LocalLibrary({
  catalog,
  host,
  busy,
  action,
  onExplore,
  onLoad,
  packages,
  onGetModels,
  onUseComfy,
  onUseWorkflow,
}) {
  const [search, setSearch] = useState("");
  const [inspected, setInspected] = useState(null);
  // A model in several ready workflows lists them to choose from.
  const [choosing, setChoosing] = useState(null);
  const artifacts = withoutPackageFiles(catalog.artifacts)
    .filter(
      (a) =>
        !a.relative_path?.includes("/.cache/") &&
        (a.format !== "unknown" ||
          /\.(gguf|onnx|ckpt|pt|pth|bin)$/i.test(a.name)),
    )
    .filter((a) =>
      `${a.name} ${a.format} ${a.relative_path}`
        .toLowerCase()
        .includes(search.toLowerCase()),
    );
  const standalone = host === "standalone";
  const inComfy = host === "comfyui";
  // Packages load in Model Run in either host; inside ComfyUI, ComfyUI's own files run through ComfyUI.
  const canLoad = standalone || packages;
  const runnable = catalog.artifacts.some((a) => groupOf(a) === "here");
  return (
    <>
      {standalone && packages && !runnable && (
        <div className="library-callout" role="note">
          <div>
            <b>Nothing here runs in this app yet</b>
            <span>
              ComfyUI’s single files need ComfyUI to run. Get a complete
              package, such as Qwen3 0.6B (1.4 GB), and it loads here.
            </span>
          </div>
          <button className="primary" onClick={() => onGetModels()}>
            Get a model that runs here
          </button>
        </div>
      )}
      <p className="muted">
        {packages
          ? "Files in your ComfyUI model folders and the packages Model Run downloaded."
          : "Files in your ComfyUI model folders."}{" "}
        {inComfy &&
          "Run any model here: text models on their own, image, video and audio models in a workflow that is ready with your models. "}
        {standalone
          ? "Files marked Runs here load in this app. The rest need ComfyUI: start Model Run inside ComfyUI to run them; here you can read their headers and explore their tensor names."
          : "Inspect and Explore read any file's header and tensor names without loading it."}
      </p>
      <div className="library-toolbar">
        <input
          aria-label="Search models"
          placeholder="Search your models…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <button
          disabled={busy}
          onClick={() =>
            action(async () => {
              await api("scan", {});
              setInspected(null);
            })
          }
        >
          <ArrowClockwise aria-hidden="true" /> Refresh
        </button>
      </div>
      <div className="library-list">
        {artifacts.length > 0 && (
          <p className="muted install-count">
            {artifacts.length} file{artifacts.length === 1 ? "" : "s"}
          </p>
        )}
        {groupArtifacts(artifacts).map(([id, label, list]) => (
          <section className="workflow-section" key={id}>
            <h3>
              {label} <span>{list.length}</span>
            </h3>
            {list.map((a) => (
              <div key={a.id}>
                <div className="artifact">
                  <div>
                    <b>{a.name}</b>
                    <span>
                      {[
                        kindOf(a),
                        a.size_bytes ? formatBytes(a.size_bytes) : null,
                        a.tensor_count
                          ? `${a.tensor_count.toLocaleString()} tensors`
                          : null,
                        a.format === "local_package" ? null : a.format,
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                    </span>
                    <small>{a.relative_path}</small>
                  </div>
                  <span
                    className={`run-badge ${(groupOf(a) === "here" && canLoad) || (inComfy && a.comfy_text) ? "here" : ""}`}
                    title={
                      inComfy &&
                      !a.comfy_text &&
                      !a.comfy_workflows?.length &&
                      !a.comfy_nearest
                        ? "No ComfyUI template uses this file. Put it in a workflow on the canvas, then use Observe this workflow in the Model Run sidebar."
                        : a.comfy_text
                          ? a.comfy_text.workflow
                            ? `Loaded the way ComfyUI's “${a.comfy_text.workflow}” template loads it`
                            : `No template uses it; loaded like ${a.comfy_text.like}`
                          : undefined
                    }
                  >
                    {groupOf(a) === "here" && canLoad
                      ? "Runs here"
                      : inComfy && a.comfy_workflows?.length && !a.comfy_text
                        ? `In ${a.comfy_workflows.length} ready workflow${a.comfy_workflows.length === 1 ? "" : "s"}`
                        : inComfy && a.comfy_nearest && !a.comfy_text
                          ? `${a.comfy_nearest.missing} file${a.comfy_nearest.missing === 1 ? "" : "s"} from a workflow`
                          : inComfy && a.comfy_text
                            ? a.comfy_text.mode === "generate"
                              ? "Writes text"
                              : "Reads prompts"
                            : inComfy
                              ? "In your own workflows"
                              : "Runs in ComfyUI"}
                  </span>
                  <button
                    aria-expanded={inspected?.id === a.id}
                    onClick={() =>
                      inspected?.id === a.id
                        ? setInspected(null)
                        : action(async () =>
                            setInspected({
                              id: a.id,
                              report: await api(`inspect/${a.id}`),
                            }),
                          )
                    }
                  >
                    {inspected?.id === a.id ? "Close" : "Inspect"}
                  </button>
                  {["safetensors", "safetensors_index"].includes(a.format) &&
                    a.status === "metadata_inspected" && (
                      <button onClick={() => onExplore(a)}>Explore</button>
                    )}
                  {inComfy &&
                    onUseWorkflow &&
                    !a.comfy_text &&
                    a.comfy_workflows?.length > 0 && (
                      <button
                        className="primary"
                        disabled={busy}
                        aria-expanded={
                          a.comfy_workflows.length > 1
                            ? choosing === a.id
                            : undefined
                        }
                        title="Run it in a workflow that is ready with your models"
                        onClick={() =>
                          a.comfy_workflows.length === 1
                            ? onUseWorkflow(a.comfy_workflows[0])
                            : setChoosing(choosing === a.id ? null : a.id)
                        }
                      >
                        Run
                      </button>
                    )}
                  {inComfy &&
                    !a.comfy_text &&
                    !a.comfy_workflows?.length &&
                    a.comfy_nearest && (
                      <button
                        title={`${a.comfy_nearest.title} needs ${a.comfy_nearest.missing} more file${a.comfy_nearest.missing === 1 ? "" : "s"} to run`}
                        onClick={() => onGetModels(a.name)}
                      >
                        Get files
                      </button>
                    )}
                  {inComfy && a.comfy_text && (
                    <button
                      className="primary"
                      disabled={busy}
                      title="Run it from Model Run: ComfyUI loads it, Model Run records it"
                      onClick={() => onUseComfy(a)}
                    >
                      Use
                    </button>
                  )}
                  {groupOf(a) === "here" && canLoad && (
                    <button
                      className="primary"
                      disabled={busy}
                      onClick={() => onLoad(a)}
                    >
                      Load
                    </button>
                  )}
                </div>
                {choosing === a.id && (
                  <div
                    className="workflow-choice"
                    role="group"
                    aria-label={`Workflows that run ${a.name}`}
                  >
                    <span>Run it in:</span>
                    {a.comfy_workflows.map((w) => (
                      <button key={w.name} onClick={() => onUseWorkflow(w)}>
                        {w.title}
                      </button>
                    ))}
                  </div>
                )}
                {inspected?.id === a.id && (
                  <InspectSummary report={inspected.report} />
                )}
              </div>
            ))}
          </section>
        ))}
        {!catalog.artifacts.length && (
          <div className="library-empty">
            <h3>No model folders configured yet</h3>
            <p>
              Start Model Run with <code>--comfy-dir</code> pointing to a
              ComfyUI folder. Inside ComfyUI, Model Run uses its live folders.
            </p>
            <p>You can still import and explore traces.</p>
          </div>
        )}
        {catalog.artifacts.length > 0 && !artifacts.length && (
          <p className="muted">No model matches that search.</p>
        )}
      </div>
      {catalog.errors?.length > 0 && (
        <details>
          <summary>{catalog.errors.length} discovery notices</summary>
          <pre>{JSON.stringify(catalog.errors, null, 2)}</pre>
        </details>
      )}
    </>
  );
}
