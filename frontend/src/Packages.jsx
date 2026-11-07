import { formatBytes } from "./installData";

const BADGES = {
  runs_here: ["here", "Runs here"],
  comfyui: ["", "For ComfyUI"],
  inspect: ["", "Inspect only"],
};

/** What a Hugging Face repository is to this app, as a badge. */
export function VerdictBadge({ verdict }) {
  const [className, label] = BADGES[verdict] || BADGES.inspect;
  return <span className={`run-badge ${className}`}>{label}</span>;
}

/** The action for a repository: a complete package when it runs here, its single files otherwise. */
function RepoAction({ repo, onPackage, onFiles }) {
  if (repo.installed) return <span className="install-have">Installed</span>;
  if (repo.verdict === "runs_here")
    return (
      <button onClick={() => onPackage(repo.repo)}>
        Get{repo.size_bytes ? ` · ${formatBytes(repo.size_bytes)}` : ""}
      </button>
    );
  return onFiles ? (
    <button onClick={() => onFiles(repo.repo)}>Files</button>
  ) : null;
}

/** A model checked end to end in this app, ready to download as a whole package. */
export function StarterRow({ starter: s, onPackage }) {
  return (
    <div className="artifact starter">
      <div>
        <b>{s.repo.split("/")[1]}</b>
        <span>
          {[
            s.kind_label,
            s.size_bytes ? formatBytes(s.size_bytes) : null,
            s.repo,
          ]
            .filter(Boolean)
            .join(" · ")}
        </span>
        <small>{s.blurb}</small>
        {s.gated && !s.installed && (
          <small className="install-note">
            Needs a Hugging Face login and the model’s licence accepted on its
            page.
          </small>
        )}
        {s.verdict !== "runs_here" && (
          <small className="install-note">{s.reason}</small>
        )}
      </div>
      <RepoAction repo={s} onPackage={onPackage} />
    </div>
  );
}

/** One Hugging Face search result, with what it is to this app. */
export function RepoRow({ repo: r, onPackage, onFiles }) {
  return (
    <div className="artifact">
      <div>
        <b>{r.repo}</b>
        <span>
          {[
            r.kind_label,
            `${(r.downloads || 0).toLocaleString()} downloads`,
            r.kind_label ? null : r.task,
            r.gated ? "needs licence acceptance" : null,
          ]
            .filter(Boolean)
            .join(" · ")}
        </span>
        {r.verdict !== "runs_here" && <small>{r.reason}</small>}
      </div>
      <VerdictBadge verdict={r.verdict} />
      <RepoAction repo={r} onPackage={onPackage} onFiles={onFiles} />
    </div>
  );
}

/** What downloading a repository as a package would do, for the person to confirm. */
export function PackageConfirm({ plan, working, onCancel, onConfirm }) {
  const left = plan.files.filter((f) => !f.have);
  return (
    <div
      className="install-confirm"
      role="group"
      aria-label="Confirm package download"
    >
      <h3>Download {plan.repo}?</h3>
      <div className="install-plan-item">
        <b>
          {plan.kind_label || "Package"} · {left.length} file
          {left.length === 1 ? "" : "s"}
          {left.length < plan.files.length
            ? ` (${plan.files.length - left.length} already here)`
            : ""}
        </b>
        <span>
          {formatBytes(plan.download_bytes)} from Hugging Face ·{" "}
          {formatBytes(plan.free_bytes)} free
        </span>
        <small>{plan.destination}</small>
        <small>
          Weights are checked against their published SHA-256 and the settings
          files against Hugging Face’s file hashes. Only safetensors weights and
          plain settings files are downloaded; no code.
        </small>
        {!plan.ok && <small className="install-note">{plan.reason}</small>}
      </div>
      <div className="install-actions">
        <button onClick={onCancel}>Not now</button>
        {plan.ok && (
          <button className="primary" disabled={working} onClick={onConfirm}>
            Download {formatBytes(plan.download_bytes)}
          </button>
        )}
      </div>
    </div>
  );
}
