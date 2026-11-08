import { useMemo } from "react";
import { Lightbulb } from "@phosphor-icons/react";
import { eventPath } from "./api";
import {
  activationStats,
  attentionHeadProfile,
  callDeltas,
  describeActivation,
  layerSeries,
  pct,
  rowSummary,
  tokenLabel,
} from "./insights";
import { mostOf, leastOf } from "./extent.js";

const fmt = (n, d = 3) => (Number.isFinite(n) ? n.toFixed(d) : "—");
const short = (n) =>
  Number.isFinite(n)
    ? Math.abs(n) >= 0.01 || n === 0
      ? n.toFixed(2)
      : n.toExponential(1)
    : "—";
const ordinalOf = (e, i) => e.denoiser_call ?? e.step ?? i;

/** What the captured values of one event say, plus how they moved across calls. */
export function ActivationInsight({ event, events = [], onPickEvent }) {
  const stats = useMemo(
    () => (event ? activationStats(event.values) : null),
    [event],
  );
  const deltas = useMemo(
    () =>
      events.some((e) => e.denoiser_call !== undefined || e.step !== undefined)
        ? callDeltas(events)
        : null,
    [events],
  );
  const moduleDeltas = useMemo(() => {
    if (!deltas || !event) return [];
    const key = event.module ?? "step";
    return deltas.filter((d) => (d.module ?? "step") === key);
  }, [deltas, event]);
  if (!stats) return null;
  const name = event.module?.replace(/^model\./, "") || null;
  const peak = Math.max(1e-9, ...moduleDeltas.map((d) => d.delta ?? 0));
  return (
    <section className="matrix-insight" aria-label="Activation insight">
      <h3>
        <Lightbulb size={18} weight="duotone" /> Activation insight
      </h3>
      <p className="matrix-insight-text">
        {describeActivation(stats, name, event.magnitude)}
      </p>
      <div className="matrix-chips">
        <span title="Root mean square of retained finite values">
          RMS {short(stats.rms)}
        </span>
        <span title="Largest magnitude divided by RMS">
          peak {stats.outlierRatio.toFixed(1)}× RMS
        </span>
        <span title="Share of values within 1% of RMS of zero">
          {pct(stats.sparsity)} near zero
        </span>
        {!event.magnitude && (
          <span title="Share of negative values">
            {pct(stats.negativeShare)} negative
          </span>
        )}
        {stats.nonFinite > 0 && <span>{stats.nonFinite} non-finite</span>}
      </div>
      {stats.outliers.length > 0 && (
        <>
          <p className="matrix-insight-sub">Strongest retained values</p>
          <div className="matrix-chips">
            {stats.outliers.map((o) => (
              <span key={o.index} title={`Flattened index ${o.index}`}>
                #{o.index}
                <b> {short(o.value)}</b>
              </span>
            ))}
          </div>
        </>
      )}
      {moduleDeltas.length > 1 && (
        <>
          <p className="matrix-insight-sub">
            Change between calls{" "}
            <small>(RMS of the difference, same values)</small>
          </p>
          <div className="matrix-deltas">
            {moduleDeltas.map((d) => (
              <button
                key={d.index}
                className={d.index === events.indexOf(event) ? "picked" : ""}
                title={`${eventPath(events[d.index])}: RMS ${fmt(d.rms)}${d.delta !== null ? ` · Δ ${fmt(d.delta)} (${pct(d.relativeDelta)} of RMS)` : " · first call"}`}
                onClick={() => onPickEvent?.(d.index)}
              >
                <i
                  style={{
                    height: `${d.delta === null ? 2 : Math.max(2, (d.delta / peak) * 48)}px`,
                  }}
                />
                <small>{d.ordinal}</small>
              </button>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

/** Profile of the captured attention head and the strongest keys for the last query. */
export function HeadInsight({ trace }) {
  const profile = useMemo(
    () =>
      trace?.kind === "attention_capture"
        ? attentionHeadProfile(trace.values)
        : null,
    [trace],
  );
  if (!profile) return null;
  const last = trace.values.length - 1;
  const attends = rowSummary(trace.values, last, trace.tokens, 4);
  const label = (i) =>
    trace.tokens?.[i] ? `“${tokenLabel(trace.tokens[i], i)}”` : `key ${i}`;
  return (
    <section className="matrix-insight" aria-label="Head insight">
      <h3>
        <Lightbulb size={18} weight="duotone" /> What this head does
      </h3>
      <p className="matrix-insight-label">{profile.label}</p>
      <p className="matrix-insight-text">{profile.description}</p>
      <div className="matrix-chips">
        <span>sink {pct(profile.sinkMass)}</span>
        <span>prev-token {pct(profile.previousTokenMass)}</span>
        <span>self {pct(profile.diagonalMass)}</span>
        <span>{profile.entropyBits.toFixed(2)} bits</span>
      </div>
      {attends.length > 0 && (
        <>
          <p className="matrix-insight-sub">{label(last)} attends to</p>
          <ul className="matrix-attends">
            {attends.map((a) => (
              <li key={a.key}>
                <button disabled>
                  <span className="matrix-attends-token">{label(a.key)}</span>
                  <span
                    className="matrix-attends-bar"
                    style={{ width: `${Math.max(2, a.weight * 100)}%` }}
                  />
                  <b>{pct(a.weight)}</b>
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="matrix-note">
        Open the Tensor Exploration Table for the head gallery, last-token
        attention and ablation.
      </p>
    </section>
  );
}

export function Sparkline({ points, label }) {
  if (!points?.length) return null;
  const values = points.map((p) => p.value).filter(Number.isFinite);
  const lo = leastOf(values),
    hi = mostOf(values);
  const w = 320,
    h = 44;
  const d = points
    .map((p, i) => {
      const x = (i / Math.max(1, points.length - 1)) * w;
      const y = h - 4 - ((p.value - lo) / Math.max(1e-9, hi - lo)) * (h - 8);
      return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <figure className="metrics-spark">
      <figcaption>
        {label} · min {fmt(lo)} · max {fmt(hi)}
      </figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        <path d={d} />
      </svg>
      <div>
        <span>{points[0].layer}</span>
        <span>{points.at(-1).layer}</span>
      </div>
    </figure>
  );
}

/** Per-event statistics table, per-layer series and attention layer summaries for Run Metrics. */
export function MetricsTable({ trace, events = [], eventIndex, onPickEvent }) {
  const rows = useMemo(
    () =>
      events.slice(0, 64).map((e, i) => ({
        index: i,
        name: eventPath(e),
        ordinal: ordinalOf(e, i),
        shape: (e.shape || []).join("×"),
        stats: activationStats(e.values),
      })),
    [events],
  );
  const series = useMemo(() => layerSeries(trace, events), [trace, events]);
  if (!trace) return null;
  const attention = trace.kind === "attention_capture";
  return (
    <div className="metrics-insights">
      {!attention && rows.length > 0 && (
        <>
          <h3>Captured events</h3>
          <div className="metrics-table-wrap">
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Event</th>
                  <th>Shape</th>
                  <th>RMS</th>
                  <th>Peak / RMS</th>
                  <th>Near zero</th>
                  <th>Reading</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr
                    key={r.index}
                    className={r.index === eventIndex ? "picked" : ""}
                    onClick={() => onPickEvent?.(r.index)}
                  >
                    <td>
                      {r.name}
                      {events[r.index].denoiser_call !== undefined && (
                        <small> call {r.ordinal}</small>
                      )}
                    </td>
                    <td>{r.shape}</td>
                    <td>{short(r.stats?.rms)}</td>
                    <td>
                      {r.stats ? `${r.stats.outlierRatio.toFixed(1)}×` : "—"}
                    </td>
                    <td>{r.stats ? pct(r.stats.sparsity) : "—"}</td>
                    <td className="reading">
                      {r.stats
                        ? describeActivation(r.stats, null).replace(
                            /^This tensor /,
                            "",
                          )
                        : "No finite values"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {events.length > 64 && (
            <p className="micro">First 64 of {events.length} events shown.</p>
          )}
        </>
      )}
      {series && (
        <>
          <h3>Across layers</h3>
          <Sparkline points={series.points} label={series.label} />
          {series.secondary && (
            <Sparkline
              points={series.secondary.points}
              label={series.secondary.label}
            />
          )}
        </>
      )}
      {attention && Array.isArray(trace.layers) && (
        <>
          <h3>Per-layer attention summary</h3>
          <div className="metrics-table-wrap">
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Layer</th>
                  <th>Heads</th>
                  <th>Sink mass</th>
                  <th>Prev-token</th>
                  <th>Self</th>
                  <th>Entropy (bits)</th>
                </tr>
              </thead>
              <tbody>
                {trace.layers.map((l) => (
                  <tr
                    key={l.index}
                    className={l.index === trace.layer ? "picked" : ""}
                  >
                    <td>L{l.index}</td>
                    <td>{l.heads ?? "—"}</td>
                    <td>{pct(l.sink_mass)}</td>
                    <td>{pct(l.previous_token_mass)}</td>
                    <td>{pct(l.diagonal_mass)}</td>
                    <td>{fmt(l.entropy_bits, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="micro">
            Head-averaged over the full matrix of every layer from the same
            forward pass.
          </p>
        </>
      )}
    </div>
  );
}
