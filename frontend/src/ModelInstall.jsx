import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { formatBytes, installProgress, planTotals } from "./installData";

const SOURCES = { templates: "ComfyUI templates", manager: "ComfyUI Manager" };
const ACTIVE = ["queued", "downloading", "verifying"];

const requestFor = (model, folder) => ({
  url: model.url,
  category: model.category,
  subfolder: model.subfolder || "",
  filename: model.filename,
  sha256: model.sha256 || undefined,
  ...(folder ? { folder } : {}),
});

function ModelRow({ model, onInstall, categories, onCategory }) {
  const known = model.size_bytes ? formatBytes(model.size_bytes) : null;
  const uses = model.workflows?.length || 0;
  return (
    <div className="artifact">
      <div>
        <b>{model.display || model.name}</b>
        <span>
          {model.category}
          {model.subfolder ? `/${model.subfolder}` : ""}
          {known ? ` · ${known}` : ""}
          {model.base ? ` · ${model.base}` : ""}
          {model.sources
            ? ` · ${model.sources.map((s) => SOURCES[s] || s).join(", ")}`
            : ""}
          {uses ? ` · used by ${uses} workflow${uses === 1 ? "" : "s"}` : ""}
        </span>
        {model.description && <small>{model.description}</small>}
        {model.note && !model.blocked && <small className="install-note">{model.note}</small>}
        {model.blocked && (
          <small className="install-note">{model.blocked}</small>
        )}
      </div>
      {model.guessed_folder && (
        <select
          aria-label={`Folder for ${model.name}`}
          title="Guessed from the file name. Check it."
          value={model.category}
          onChange={(e) => onCategory(e.target.value)}
        >
          {categories.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      )}
      {model.installed ? (
        <span className="install-have">Installed</span>
      ) : (
        <button
          disabled={Boolean(model.blocked)}
          onClick={() => onInstall([model])}
        >
          Install
        </button>
      )}
    </div>
  );
}

function Confirm({ plan, onFolder, onCancel, onConfirm, working }) {
  const totals = planTotals(plan);
  return (
    <div className="install-confirm" role="group" aria-label="Confirm download">
      <h3>Download {plan.items.length === 1 ? "this file" : "these files"}?</h3>
      {plan.items.map((item, index) => (
        <div className="install-plan-item" key={item.url || index}>
          <b>{item.filename}</b>
          {item.ok ? (
            <>
              <span>
                {formatBytes(item.size_bytes)}
                {item.resume_bytes
                  ? ` · resumes from ${formatBytes(item.resume_bytes)}`
                  : ""}{" "}
                · from{" "}
                {new URL(item.url).pathname.split("/").slice(1, 3).join("/")} on
                Hugging Face
              </span>
              <small>{item.destination}</small>
              <small>
                {item.sha256
                  ? "The SHA-256 will be checked against the published one."
                  : "No published SHA-256, so only the file's size and format can be checked."}
              </small>
              {item.folders?.length > 1 && (
                <label className="install-folder">
                  Folder{" "}
                  <select
                    value={
                      item.folders.find((f) =>
                        item.destination.startsWith(f),
                      ) || item.folders[0]
                    }
                    onChange={(e) => onFolder(index, e.target.value)}
                  >
                    {item.folders.map((f) => (
                      <option key={f}>{f}</option>
                    ))}
                  </select>
                </label>
              )}
            </>
          ) : (
            <small className="install-note">{item.reason}</small>
          )}
        </div>
      ))}
      <p className="install-total">
        {totals.ready} of {plan.items.length} ready ·{" "}
        {formatBytes(totals.bytes)} · {formatBytes(totals.free)} free
      </p>
      <div className="install-actions">
        <button onClick={onCancel}>Not now</button>
        <button
          className="primary"
          disabled={!totals.ready || working}
          onClick={onConfirm}
        >
          Download {formatBytes(totals.bytes)}
        </button>
      </div>
    </div>
  );
}

function Jobs({ jobs, onRetry, onError }) {
  if (!jobs.length) return null;
  const cancel = (id, discard) =>
    api("install/cancel", { id, discard }).catch((e) => onError(e.message));
  return (
    <div className="install-jobs" aria-label="Downloads">
      {jobs.map((job) => {
        const progress = installProgress(job);
        const active = ACTIVE.includes(job.status);
        return (
          <div className="install-job" key={job.id}>
            <div>
              <b>{job.filename}</b>
              <span>{progress.label}</span>
              {active && (
                <progress
                  aria-label={`${job.filename} progress`}
                  max={1}
                  value={progress.fraction}
                />
              )}
              {job.error && <small className="install-note">{job.error}</small>}
            </div>
            {active && (
              <button onClick={() => cancel(job.id, false)}>Pause</button>
            )}
            {active && (
              <button onClick={() => cancel(job.id, true)}>Cancel</button>
            )}
            {["failed", "cancelled"].includes(job.status) && (
              <button onClick={() => onRetry(job)}>Retry</button>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Find models in the lists ComfyUI ships or on Hugging Face, and install them into ComfyUI's folders. */
export default function ModelInstall({ jobs, onInstalled }) {
  const [tab, setTab] = useState("models");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [missingOnly, setMissingOnly] = useState(false);
  const [results, setResults] = useState(null);
  const [workflows, setWorkflows] = useState(null);
  const [repos, setRepos] = useState(null);
  const [files, setFiles] = useState(null);
  const [plan, setPlan] = useState(null);
  const [requests, setRequests] = useState([]);
  const [error, setError] = useState("");
  const [working, setWorking] = useState(false);
  const installedCount = useMemo(
    () => jobs.filter((j) => j.status === "installed").length,
    [jobs],
  );

  useEffect(() => {
    let alive = true;
    const timer = setTimeout(() => {
      const run =
        tab === "models"
          ? api("install/search", {
              query,
              category: category || undefined,
              missing_only: missingOnly,
            }).then((r) => alive && setResults(r))
          : tab === "workflows"
            ? api("install/workflows", { query }).then(
                (r) => alive && setWorkflows(r),
              )
            : null;
      run?.catch((e) => alive && setError(e.message));
    }, 200);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [tab, query, category, missingOnly, installedCount]);

  useEffect(() => {
    if (installedCount) onInstalled?.();
  }, [installedCount, onInstalled]);

  async function guarded(fn) {
    setError("");
    setWorking(true);
    try {
      await fn();
    } catch (e) {
      setError(e.message);
    } finally {
      setWorking(false);
    }
  }
  const check = (items) =>
    guarded(async () => {
      setRequests(items);
      setPlan(await api("install/plan", { items }));
    });
  const confirm = () =>
    guarded(async () => {
      const ready = plan.items
        .map((item, i) => [item, requests[i]])
        .filter(([item]) => item.ok)
        .map(([item, request]) => ({
          ...request,
          confirmed_bytes: item.size_bytes,
        }));
      await api("install/start", { items: ready });
      setPlan(null);
    });
  const searchHub = () =>
    guarded(async () => {
      setFiles(null);
      setRepos((await api("install/hub", { query })).repos);
    });
  const openRepo = (repo) =>
    guarded(async () => {
      setFiles(await api("install/hub-files", { repo }));
    });
  const setFileCategory = (id, value) =>
    setFiles((current) => ({
      ...current,
      files: current.files.map((f) =>
        f.id === id ? { ...f, category: value, blocked: null } : f,
      ),
    }));

  return (
    <div className="model-install">
      <div className="tabs" role="tablist" aria-label="Where to look">
        {[
          ["models", "ComfyUI lists"],
          ["workflows", "By workflow"],
          ["hub", "Hugging Face"],
        ].map(([id, label]) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            className={tab === id ? "active" : ""}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="library-toolbar">
        <input
          aria-label="Search for models to install"
          placeholder={
            tab === "workflows"
              ? "Search workflows…"
              : tab === "hub"
                ? "Search Hugging Face…"
                : "Search models, VAEs, LoRAs…"
          }
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (tab === "hub" && e.key === "Enter" && query.trim()) searchHub();
          }}
        />
        {tab === "models" && (
          <>
            <select
              aria-label="Model type"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              <option value="">All types</option>
              {(results?.categories || []).map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <label className="install-check">
              <input
                type="checkbox"
                checked={missingOnly}
                onChange={(e) => setMissingOnly(e.target.checked)}
              />{" "}
              Not installed
            </label>
          </>
        )}
        {tab === "hub" && (
          <button disabled={working || !query.trim()} onClick={searchHub}>
            Search online
          </button>
        )}
      </div>
      {error && <p className="install-note">{error}</p>}
      {plan && (
        <Confirm
          plan={plan}
          working={working}
          onCancel={() => setPlan(null)}
          onConfirm={confirm}
          onFolder={(index, folder) => {
            const next = requests.map((r, i) =>
              i === index ? { ...r, folder } : r,
            );
            check(next);
          }}
        />
      )}
      <Jobs
        jobs={jobs}
        onError={setError}
        onRetry={(job) =>
          check([
            {
              url: job.url,
              category: job.category,
              filename: job.filename,
              ...(job.sha256 ? { sha256: job.sha256 } : {}),
            },
          ])
        }
      />
      <div className="library-list">
        {tab === "models" && results && (
          <>
            <p className="muted install-count">
              {results.total.toLocaleString()} found
              {results.total > results.models.length
                ? `, showing the first ${results.models.length}`
                : ""}
            </p>
            {results.models.map((m) => (
              <ModelRow
                key={m.id}
                model={m}
                onInstall={(items) => check(items.map((x) => requestFor(x)))}
              />
            ))}
            {results.notes?.map((n) => (
              <p className="muted" key={n}>
                {n}
              </p>
            ))}
          </>
        )}
        {tab === "workflows" &&
          workflows?.workflows.map((w) => {
            const missing = w.models.filter((m) => !m.installed && !m.blocked);
            return (
              <div className="artifact" key={w.name}>
                <div>
                  <b>{w.title}</b>
                  <span>
                    {w.models.length} model
                    {w.models.length === 1 ? "" : "s"} · {w.missing} missing
                  </span>
                  <small>
                    {w.models
                      .map(
                        (m) =>
                          `${m.installed ? "✓" : "○"} ${m.category}/${m.filename}`,
                      )
                      .join("  ")}
                  </small>
                </div>
                {missing.length ? (
                  <button
                    onClick={() =>
                      check(missing.slice(0, 32).map((m) => requestFor(m)))
                    }
                  >
                    Install {missing.length} missing
                  </button>
                ) : (
                  <span className="install-have">
                    {w.missing ? "Cannot install" : "Ready"}
                  </span>
                )}
              </div>
            );
          })}
        {tab === "hub" && (
          <>
            {!repos && (
              <p className="muted">
                Nothing is sent to Hugging Face until you search. The folder for
                a file is a guess from its name, so check it before installing.
              </p>
            )}
            {!files &&
              repos?.map((r) => (
                <div className="artifact" key={r.repo}>
                  <div>
                    <b>{r.repo}</b>
                    <span>
                      {(r.downloads || 0).toLocaleString()} downloads
                      {r.task ? ` · ${r.task}` : ""}
                      {r.formats?.length === 1 && r.formats[0] === "gguf" ? " · GGUF" : ""}
                      {r.gated ? " · needs licence acceptance" : ""}
                    </span>
                  </div>
                  <button onClick={() => openRepo(r.repo)}>Files</button>
                </div>
              ))}
            {files && (
              <>
                <button onClick={() => setFiles(null)}>Back to results</button>
                <p className="muted install-count">
                  {files.repo}: {files.total} model file
                  {files.total === 1 ? "" : "s"}
                </p>
                {files.files.map((f) => (
                  <ModelRow
                    key={f.id}
                    model={f}
                    categories={files.categories}
                    onCategory={(value) => setFileCategory(f.id, value)}
                    onInstall={(items) =>
                      check(items.map((x) => requestFor(x)))
                    }
                  />
                ))}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
