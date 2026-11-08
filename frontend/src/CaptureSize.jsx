import { useEffect, useState } from "react";
import { api } from "./api";

const PRESETS = {
  standard: "What Model Run was tested with. Runs on any computer.",
  large:
    "About 4 to 8 times more of everything. Suits 32 GB of RAM or a large graphics card.",
  maximum:
    "As much as this computer's free memory allows. Big runs take longer to open and can slow the page.",
  custom: "Set each limit yourself, between a quarter of Standard and Maximum.",
};
export const LABELS = {
  standard: "Standard",
  large: "Large",
  maximum: "Maximum",
  custom: "Custom",
};
const BYTES = new Set(["trace_bytes", "history_bytes"]);
const STORE = "model-run-capture-size";

export const bytes = (n) =>
  !Number.isFinite(n)
    ? "—"
    : n >= 1024 ** 3
      ? `${(n / 1024 ** 3).toFixed(n % 1024 ** 3 ? 1 : 0)} GB`
      : `${Math.round(n / 1024 ** 2).toLocaleString()} MB`;
const shown = (key, n) => (BYTES.has(key) ? bytes(n) : n.toLocaleString());

function remembered() {
  try {
    return JSON.parse(localStorage.getItem(STORE) || "null");
  } catch {
    return null;
  }
}
function remember(choice) {
  try {
    localStorage.setItem(STORE, JSON.stringify(choice));
  } catch {
    /* private window: the server still holds the choice until it restarts */
  }
}

/** Put back the capture size chosen in this browser when the server has started over on Standard. */
export function useRememberedCaptureSize(active) {
  const [done, setDone] = useState(false);
  useEffect(() => {
    if (done || !active) return;
    setDone(true);
    const choice = remembered();
    if (!choice?.profile || choice.profile === active.profile) return;
    api("capture-limits", choice).catch(() => {});
  }, [active, done]);
}

/** Settings: how much each run keeps, from Standard to what this computer can hold. */
export default function CaptureSize({ onError }) {
  const [info, setInfo] = useState(null);
  const [draft, setDraft] = useState(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    api("capture-limits")
      .then((d) => {
        setInfo(d);
        setDraft(d.values);
      })
      .catch((e) => onError?.(e.message));
  }, []);
  if (!info) return null;
  async function choose(profile, values) {
    setSaving(true);
    try {
      const d = await api("capture-limits", { profile, values });
      setInfo(d);
      setDraft(d.values);
      remember({ profile, values: profile === "custom" ? values : undefined });
    } catch (e) {
      onError?.(e.message);
    } finally {
      setSaving(false);
    }
  }
  const machine = info.machine || {};
  const custom = info.profile === "custom";
  const changed =
    custom &&
    draft &&
    info.fields.some((f) => draft[f.key] !== info.values[f.key]);
  return (
    <section className="capture-size" aria-label="Capture size">
      <h3>Capture size</h3>
      <p className="micro">
        How much each run keeps: prompt length, tokens per layer, attention,
        layers, steps and history. Anything over a limit is still run and kept
        smaller, and Run Metrics says what was kept.
      </p>
      <div className="capture-size-presets" role="radiogroup">
        {Object.keys(LABELS).map((p) => (
          <label key={p} className={info.profile === p ? "chosen" : ""}>
            <input
              type="radio"
              name="capture-size"
              checked={info.profile === p}
              disabled={saving}
              onChange={() =>
                choose(p, p === "custom" ? info.values : undefined)
              }
            />
            <b>{LABELS[p]}</b>
            <small>{PRESETS[p]}</small>
          </label>
        ))}
      </div>
      <p className="micro capture-size-machine">
        This computer:{" "}
        {machine.ram_total
          ? `${bytes(machine.ram_total)} RAM, ${bytes(machine.ram_free)} free`
          : "RAM unknown"}
        {machine.gpu_total
          ? ` · ${machine.gpu_name || "GPU"} ${bytes(machine.gpu_total)}, ${bytes(machine.gpu_free)} free`
          : ""}
        . Before a run that would not fit, Model Run stops and says why.
      </p>
      <details className="capture-size-table">
        <summary>{custom ? "Set each limit" : "See every limit"}</summary>
        <table>
          <thead>
            <tr>
              <th>Limit</th>
              <th>Standard</th>
              <th>Large</th>
              <th>Maximum</th>
              <th>Now</th>
            </tr>
          </thead>
          <tbody>
            {info.fields.map((f) => (
              <tr key={f.key} title={f.help}>
                <th scope="row">{f.label}</th>
                {["standard", "large", "maximum"].map((p) => (
                  <td key={p}>{shown(f.key, info.profiles[p][f.key])}</td>
                ))}
                <td>
                  {custom ? (
                    <input
                      type="number"
                      aria-label={f.label}
                      min={
                        BYTES.has(f.key)
                          ? Math.ceil(info.floor[f.key] / 1024 ** 2)
                          : info.floor[f.key]
                      }
                      max={
                        BYTES.has(f.key)
                          ? info.ceiling[f.key] / 1024 ** 2
                          : info.ceiling[f.key]
                      }
                      value={
                        BYTES.has(f.key)
                          ? Math.round(draft[f.key] / 1024 ** 2)
                          : draft[f.key]
                      }
                      onChange={(e) => {
                        const n = Math.round(Number(e.target.value) || 0);
                        setDraft((d) => ({
                          ...d,
                          [f.key]: BYTES.has(f.key) ? n * 1024 ** 2 : n,
                        }));
                      }}
                    />
                  ) : (
                    <b>{shown(f.key, info.values[f.key])}</b>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {custom && (
          <p className="micro">
            Sizes are in MB. Values outside a quarter of Standard to Maximum are
            moved to the nearest end.{" "}
            <button
              className="text-button"
              disabled={!changed || saving}
              onClick={() => choose("custom", draft)}
            >
              Apply
            </button>
          </p>
        )}
      </details>
    </section>
  );
}

/** Run Metrics: what the run held against what it kept, on the capture size it ran with. */
export function KeptTable({ trace, record, retention }) {
  if (!trace) return null;
  const rows = trace.kept || [];
  const limits = trace.capture_limits;
  const size = record?.size_bytes;
  const big = size > 128 * 1024 ** 2;
  return (
    <section className="kept-table" aria-label="What was kept">
      <header>
        <h3>What was kept</h3>
        {limits && (
          <span
            className="chip"
            title="The capture size this run was recorded with"
          >
            {LABELS[limits.profile] || limits.profile} capture size
          </span>
        )}
      </header>
      {rows.length ? (
        <table>
          <thead>
            <tr>
              <th>What</th>
              <th>This run had</th>
              <th>Kept</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.what} className={r.note ? "short" : ""}>
                <th scope="row">{r.what}</th>
                <td>{r.needed}</td>
                <td>
                  {r.kept}
                  {r.note && <small>{r.note}</small>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="micro">
          {limits
            ? "Nothing in this run had to be cut."
            : "This run has no record of what was kept: it was imported, or recorded before Model Run kept one."}
        </p>
      )}
      {(size || retention) && (
        <p className="micro">
          {size ? `This run takes ${bytes(size)}` : ""}
          {size && limits
            ? ` of the ${bytes(limits.values.trace_bytes)} one run may use`
            : ""}
          {retention
            ? `${size ? "; history" : "History"} holds ${bytes(retention.bytes)} of ${bytes(retention.max_bytes)} across ${retention.runs} of ${retention.max_runs} runs`
            : ""}
          .
          {big &&
            " A run this large can take a moment to open, and the 3D view may be slower."}
        </p>
      )}
    </section>
  );
}
