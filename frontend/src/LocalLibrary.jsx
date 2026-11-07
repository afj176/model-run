import { useState } from "react";
import { ArrowClockwise } from "@phosphor-icons/react";
import { api } from "./api";
import { formatBytes } from "./installData";
import {
  formatCount,
  groupArtifacts,
  groupOf,
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
}) {
  const [search, setSearch] = useState("");
  const [inspected, setInspected] = useState(null);
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
  return (
    <>
      <p className="muted">
        Files in your ComfyUI model folders.{" "}
        {standalone
          ? "Files marked Runs here load in this app. The rest run in ComfyUI, where Model Run · Observe records them; here you can read their headers and explore their tensor names."
          : "They run in ComfyUI, where Model Run · Observe records them; here you can read their headers and explore their tensor names."}
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
                        a.size_bytes ? formatBytes(a.size_bytes) : null,
                        a.tensor_count
                          ? `${a.tensor_count.toLocaleString()} tensors`
                          : null,
                        a.format === "local_package" ? "folder" : a.format,
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                    </span>
                    <small>{a.relative_path}</small>
                  </div>
                  <span
                    className={`run-badge ${groupOf(a) === "here" && standalone ? "here" : ""}`}
                  >
                    {groupOf(a) === "here" && standalone
                      ? "Runs here"
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
                  {groupOf(a) === "here" && standalone && (
                    <button
                      className="primary"
                      disabled={busy}
                      onClick={() => onLoad(a)}
                    >
                      Load
                    </button>
                  )}
                </div>
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
