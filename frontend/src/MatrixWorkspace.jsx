import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowsOut,
  Check,
  Clock,
  Cube,
  Eye,
  GridFour,
  Info,
  Lightbulb,
  MagnifyingGlass,
  Stack,
  Table,
} from "@phosphor-icons/react";
import MatrixScene from "./MatrixScene";
import { extractMatrix } from "./matrixData";
import {
  ablationStory,
  activationStats,
  attentionHeadProfile,
  callDeltas,
  describeActivation,
  layerOf,
  layerStops,
  moduleInStop,
  layerSeries,
  pct,
  rowSummary,
  tokenLabel,
  tokenText,
} from "./insights";
import "./matrix.css";

const fmt = (n) =>
  Number.isFinite(n)
    ? new Intl.NumberFormat("en", { maximumSignificantDigits: 6 }).format(n)
    : "—";
const fmtShort = (n) =>
  Number.isFinite(n)
    ? Math.abs(n) >= 0.01 || n === 0
      ? n.toFixed(2)
      : n.toExponential(1)
    : "—";
const axisWord = (axis) =>
  ({
    query_position: "query",
    key_position: "key",
  })[axis] || axis.replaceAll("_", " ");
const axisTitle = (axis) =>
  ({
    query_position: "query index",
    key_position: "key index",
  })[axis] || axis.replaceAll("_", " ");
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const WINDOW = 64;

export function mix(t) {
  // pale ivory → mint → deep teal, matching the 3D bars
  const stops = [
    [242, 239, 228],
    [127, 217, 194],
    [15, 159, 134],
  ];
  const p = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  const [a, b] = t < 0.5 ? [stops[0], stops[1]] : [stops[1], stops[2]];
  const ch = (i) => Math.round(a[i] + (b[i] - a[i]) * p);
  return `rgb(${ch(0)},${ch(1)},${ch(2)})`;
}

export default function MatrixWorkspace({
  trace,
  event,
  events = [],
  eventIndex = 0,
  onPickEvent,
  onPickHead,
  layers = [],
  busy,
  canAttention,
  onCapture,
  onAblate,
  onPickLayer,
  stageOptions = null,
  sliceNav = null,
  layer,
  setLayer,
  head,
  setHead,
}) {
  const [rowOffset, setRowOffset] = useState(0),
    [colOffset, setColOffset] = useState(0);
  const [selected, setSelected] = useState({ row: 0, col: 0 }),
    [reset, setReset] = useState(0),
    [zoom, setZoom] = useState(1),
    [stageMode, setStageMode] = useState("overview");
  useEffect(() => {
    setRowOffset(0);
    setColOffset(0);
    // Default to the strongest captured value in the middle row of the window.
    try {
      const first = extractMatrix(trace, event, {
        maxRows: WINDOW,
        maxCols: WINDOW,
      });
      if (first?.rows) {
        let pick = null;
        const rows =
          first.kind === "attention"
            ? [Math.floor(first.rows / 2)]
            : first.matrix.map((_, r) => r);
        for (const r of rows)
          first.matrix[r].forEach((v, c) => {
            if (
              Number.isFinite(v) &&
              (pick === null || Math.abs(v) > Math.abs(pick.value))
            )
              pick = { row: r, col: c, value: v };
          });
        setSelected(
          pick ? { row: pick.row, col: pick.col } : { row: 0, col: 0 },
        );
        return;
      }
    } catch {
      // fall through to the origin
    }
    setSelected({ row: 0, col: 0 });
  }, [trace, event?.module, event?.step, event?.slice_label]);
  const extracted = useMemo(() => {
    try {
      return {
        data: extractMatrix(trace, event, {
          rowOffset,
          colOffset,
          maxRows: WINDOW,
          maxCols: WINDOW,
        }),
      };
    } catch (e) {
      return { error: e.message };
    }
  }, [trace, event, rowOffset, colOffset]);
  const data = extracted.data;
  const attention = data?.kind === "attention";
  const tokens =
    attention &&
    Array.isArray(trace?.tokens) &&
    trace.tokens.length === data.shape[0]
      ? trace.tokens
      : null;
  const labelFor = (index) =>
    tokens ? tokenLabel(tokens[index], index) : null;
  const val = data?.matrix[selected.row]?.[selected.col];
  const status = data?.states[selected.row]?.[selected.col];
  const finite = data?.matrix.flat().filter((v) => Number.isFinite(v)) || [];
  const abs = Math.max(0.00001, ...finite.map(Math.abs));
  const rowName = data ? axisWord(data.axes[0]) : "row";
  const colName = data ? axisWord(data.axes[1]) : "column";
  const rowIndex = rowOffset + selected.row;
  const colIndex = colOffset + selected.col;
  const rowToken = labelFor(rowIndex);
  const colToken = labelFor(colIndex);
  const valueLabel = attention ? "Weight" : "Value";
  const valueText =
    status === "captured"
      ? fmt(val)
      : status === "nonfinite"
        ? "Non-finite"
        : status === "uncaptured"
          ? "Not captured"
          : "—";
  const upperTriangleZero = useMemo(() => {
    if (!attention || !data || data.rows < 2) return false;
    let seen = false;
    for (let r = 0; r < data.rows; r++)
      for (let c = r + 1; c < data.cols; c++) {
        const v = data.matrix[r][c];
        if (v === null) continue;
        if (v !== 0) return false;
        seen = true;
      }
    return seen;
  }, [attention, data]);
  const color = (v) =>
    v == null
      ? "#2a2342"
      : attention
        ? mix(Math.max(0, Math.min(1, v)))
        : v < 0
          ? `rgba(240,110,111,${Math.max(0.12, Math.min(1, Math.abs(v) / abs))})`
          : mix(Math.max(0, Math.min(1, Math.abs(v) / abs)));
  function page(axis, delta) {
    if (!data) return;
    const current = axis === "row" ? rowOffset : colOffset,
      limit = data.dimensions[axis === "row" ? 0 : 1];
    const next = Math.max(0, Math.min(Math.max(0, limit - 1), current + delta));
    (axis === "row" ? setRowOffset : setColOffset)(next);
    setSelected({ row: 0, col: 0 });
  }
  // Ordinal event timeline: layer groups between input and output.
  const currentLayer =
    trace?.kind === "attention_capture"
      ? trace.layer
      : layerOf(event?.module ?? events[eventIndex]?.module);
  const stops = useMemo(() => layerStops(layers.map((n) => n.path)), [layers]);
  const stopPositions = stops.map((_, i) => i);
  const stepTotal = stops.length;
  const eventModule = event?.module ?? events[eventIndex]?.module;
  const stepIndex =
    currentLayer === null || currentLayer === undefined
      ? -1
      : (() => {
          if (trace?.kind !== "attention_capture") {
            const exact = stops.findIndex((s) => moduleInStop(eventModule, s));
            if (exact >= 0) return exact;
          }
          return stops.findIndex((s) => s.number === currentLayer);
        })();
  const currentStop = stepIndex >= 0 ? stops[stepIndex] : null;
  const eventLabel =
    trace?.kind === "attention_capture"
      ? `L${trace.layer} · ${trace.head_mean ? "head mean" : `H${trace.head}`} · Attention (Q × K)`
      : event?.module
        ? `${event.module.replace(/^model\./, "")}`
        : event?.step !== undefined
          ? `Step ${event.step}`
          : "—";
  function pickLayer(stop) {
    if (onPickLayer) {
      onPickLayer(stop.number);
      return;
    }
    setLayer(stop.number);
    let at = events.findIndex((e) => moduleInStop(e.module, stop));
    if (at < 0 && !stop.repeated)
      at = events.findIndex((e) => layerOf(e.module) === stop.number);
    if (at >= 0 && trace?.kind !== "attention_capture") onPickEvent?.(at);
  }
  const neighborhood = useMemo(() => {
    if (!data) return [];
    const out = [];
    for (let dr = -1; dr <= 1; dr++) {
      const row = [];
      for (let dc = -2; dc <= 2; dc++) {
        const r = selected.row + dr,
          c = selected.col + dc;
        const inside = r >= 0 && r < data.rows && c >= 0 && c < data.cols;
        row.push(
          inside
            ? { r, c, v: data.matrix[r][c], state: data.states[r][c] }
            : null,
        );
      }
      out.push(row);
    }
    return out;
  }, [data, selected]);
  const ticks = (n, offset, dense) => {
    const limit = dense ? 16 : 8;
    const step =
      [1, 2, 4, 5, 8, 10, 16, 20, 32].find((s) => s >= n / limit) || 32;
    const list = [];
    for (let i = 0; i < n; i += step) list.push(i);
    if (list.at(-1) !== n - 1) list.push(n - 1);
    return list.map((i) => ({
      i,
      label: labelFor(offset + i) ?? offset + i,
    }));
  };
  const windowed =
    data && (data.rows < data.dimensions[0] || data.cols < data.dimensions[1]);
  const title = attention ? "Attention weights" : "Tensor values";

  // Insights derived from the recorded values.
  const profile = useMemo(
    () => (attention ? attentionHeadProfile(trace.values) : null),
    [attention, trace],
  );
  const attendsTo = useMemo(
    () => (attention ? rowSummary(trace.values, rowIndex, trace.tokens) : []),
    [attention, trace, rowIndex],
  );
  const gallery = useMemo(
    () =>
      attention && Array.isArray(trace.heads)
        ? trace.heads.map((matrix, index) => ({
            index,
            matrix,
            profile: attentionHeadProfile(matrix),
          }))
        : [],
    [attention, trace],
  );
  const ablation = useMemo(
    () => (attention ? ablationStory(trace.ablation, trace.head) : null),
    [attention, trace],
  );
  const ablationByHead = useMemo(() => {
    const map = new Map();
    for (const entry of trace?.ablation?.heads || [])
      map.set(entry.head, entry);
    return map;
  }, [trace]);
  const measured = Array.isArray(trace?.attribution?.last_position_attention);
  const attribution =
    attention && Array.isArray(trace.attribution?.values)
      ? measured
        ? trace.attribution.last_position_attention
        : trace.attribution.values
      : null;
  const stats = useMemo(
    () => (!attention && event ? activationStats(event.values) : null),
    [attention, event],
  );
  const deltas = useMemo(
    () =>
      !attention &&
      events.some((e) => e.denoiser_call !== undefined || e.step !== undefined)
        ? callDeltas(events)
        : null,
    [attention, events],
  );
  const moduleDeltas = useMemo(() => {
    if (!deltas || !event) return [];
    const key = event.module ?? "step";
    return deltas.filter((d) => (d.module ?? "step") === key);
  }, [deltas, event]);
  const series = useMemo(() => layerSeries(trace, events), [trace, events]);
  return (
    <div className="matrix-workspace">
      <section
        className="matrix-stage"
        aria-label="Tensor matrix visualization"
      >
        <header className="matrix-stage-header">
          <div className="matrix-stage-title">
            <h2>{data ? title : "Tensor exploration table"}</h2>
            {data ? (
              <p className="matrix-meta">
                <span>
                  {currentStop
                    ? `${currentStop.label} · `
                    : currentLayer !== null && currentLayer !== undefined
                      ? `L${currentLayer} · `
                      : ""}
                  {attention
                    ? `Attention${trace.head_mean ? " · head mean" : trace.head !== undefined ? ` · head ${trace.head}` : ""}`
                    : data.source}
                  {event?.slice_label ? ` · ${event.slice_label}` : ""}
                </span>
                <span>{(event?.source_shape ?? data.shape).join(" × ")}</span>
                <span>
                  {cap(rowName)} {rowIndex}
                  {rowToken ? ` “${rowToken}”` : ""} · {cap(colName)} {colIndex}
                  {colToken ? ` “${colToken}”` : ""}
                </span>
              </p>
            ) : (
              <p className="matrix-meta">
                <span>Select a capture or run an attention probe</span>
              </p>
            )}
          </div>
          <div className="matrix-tools">
            <div
              className="matrix-segment"
              role="group"
              aria-label="Stage view"
            >
              <button
                className={stageMode === "overview" ? "selected" : ""}
                aria-pressed={stageMode === "overview"}
                onClick={() => setStageMode("overview")}
              >
                <Cube size={17} /> Overview: aggregated
              </button>
              <button
                className={stageMode === "detail" ? "selected" : ""}
                aria-pressed={stageMode === "detail"}
                onClick={() => setStageMode("detail")}
              >
                <Table size={17} /> Cell detail
              </button>
            </div>
            {stageOptions &&
              (stageOptions.attention || stageOptions.outputs) && (
                <div
                  className="matrix-segment matrix-stage-kind"
                  role="group"
                  aria-label="Stage tensor"
                >
                  <button
                    className={
                      stageOptions.current === "attention" ? "selected" : ""
                    }
                    aria-pressed={stageOptions.current === "attention"}
                    disabled={!stageOptions.attention}
                    onClick={() => stageOptions.open(stageOptions.attention)}
                    title="Attention weights of the selected layer and head"
                  >
                    Attention
                  </button>
                  <button
                    className={
                      stageOptions.current === "outputs" ? "selected" : ""
                    }
                    aria-pressed={stageOptions.current === "outputs"}
                    disabled={!stageOptions.outputs}
                    onClick={() => stageOptions.open(stageOptions.outputs)}
                    title="Captured output values of the selected layer"
                  >
                    Layer output
                  </button>
                </div>
              )}
            <label className="matrix-zoom">
              <MagnifyingGlass size={16} />
              <select
                aria-label="Zoom"
                disabled={stageMode !== "overview"}
                title={
                  stageMode === "overview"
                    ? "Stage zoom"
                    : "Zoom applies to the 3D overview"
                }
                value={zoom}
                onChange={(e) => {
                  setZoom(Number(e.target.value));
                }}
              >
                {[0.5, 0.75, 1, 1.25, 1.5, 2].map((z) => (
                  <option key={z} value={z}>
                    {Math.round(z * 100)}%
                  </option>
                ))}
              </select>
            </label>
            <button
              className="matrix-icon-button"
              aria-label="Expand matrix visualization"
              onClick={(e) =>
                e.currentTarget.closest(".matrix-stage").requestFullscreen()
              }
            >
              <ArrowsOut size={18} />
            </button>
          </div>
        </header>
        {sliceNav}
        {data && data.rows && data.cols ? (
          <>
            {stageMode === "overview" ? (
              <div className="matrix-canvas">
                <MatrixScene
                  matrix={data.matrix}
                  selected={selected}
                  onSelect={(row, col) => setSelected({ row, col })}
                  mode={data.kind}
                  resetKey={reset}
                  zoom={zoom}
                  axisLabels={{
                    row: tokens ? "query token" : axisTitle(data.axes[0]),
                    col: tokens ? "key token" : axisTitle(data.axes[1]),
                    rowOffset,
                    colOffset,
                    rowLabels: tokens
                      ? data.matrix.map((_, r) => labelFor(rowOffset + r))
                      : null,
                    colLabels: tokens
                      ? data.matrix[0].map((_, c) => labelFor(colOffset + c))
                      : null,
                  }}
                  callout={
                    <>
                      <span>
                        {tokens
                          ? `“${rowToken}” → “${colToken}”`
                          : `${cap(rowName.charAt(0))} ${rowIndex}, ${cap(colName.charAt(0))} ${colIndex}`}
                      </span>
                      <strong>
                        {valueLabel} <em>{valueText}</em>
                        {tokens && (
                          <small>
                            Q {rowIndex} · K {colIndex}
                          </small>
                        )}
                      </strong>
                    </>
                  }
                />
              </div>
            ) : (
              <div className="matrix-table-wrap">
                <table className="matrix-table">
                  <thead>
                    <tr>
                      <th>
                        {rowName} \ {colName}
                      </th>
                      {data.matrix[0].map((_, c) => (
                        <th
                          key={c}
                          className={c === selected.col ? "picked" : ""}
                        >
                          {labelFor(colOffset + c) ?? colOffset + c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.matrix.map((row, r) => (
                      <tr key={r}>
                        <th className={r === selected.row ? "picked" : ""}>
                          {labelFor(rowOffset + r) ?? rowOffset + r}
                        </th>
                        {row.map((v, c) => (
                          <td
                            key={c}
                            className={
                              r === selected.row && c === selected.col
                                ? "picked"
                                : r === selected.row || c === selected.col
                                  ? "lit"
                                  : ""
                            }
                            style={{ background: color(v) }}
                            onClick={() => setSelected({ row: r, col: c })}
                            title={`${rowOffset + r}, ${colOffset + c}: ${data.states[r][c] === "captured" ? fmt(v) : data.states[r][c]}`}
                          >
                            {data.states[r][c] === "captured"
                              ? fmtShort(v)
                              : "·"}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {stageMode === "overview" && (
              <aside className="matrix-inset" aria-label="Cell detail zoom">
                <h4>
                  Cell detail <span>(zoom)</span>
                </h4>
                <div className="matrix-inset-body">
                  <div className="matrix-inset-grid">
                    {neighborhood.map((row, i) => (
                      <div key={i} className="matrix-inset-row">
                        {row.map((cell, j) => (
                          <button
                            key={j}
                            disabled={!cell}
                            aria-label={
                              cell
                                ? `${rowName} ${rowOffset + cell.r}, ${colName} ${colOffset + cell.c}: ${cell.state === "captured" ? fmt(cell.v) : cell.state}`
                                : "outside window"
                            }
                            className={
                              !cell
                                ? "void"
                                : cell.r === selected.row &&
                                    cell.c === selected.col
                                  ? "picked"
                                  : ""
                            }
                            style={
                              cell
                                ? {
                                    "--cell": color(cell.v),
                                    "--lift": `${Math.round(
                                      6 +
                                        (attention
                                          ? Math.min(1, cell.v ?? 0)
                                          : Math.min(
                                              1,
                                              Math.abs(cell.v ?? 0) / abs,
                                            )) *
                                          16,
                                    )}px`,
                                  }
                                : undefined
                            }
                            onClick={() =>
                              cell && setSelected({ row: cell.r, col: cell.c })
                            }
                          />
                        ))}
                      </div>
                    ))}
                  </div>
                  <dl>
                    <dt>{cap(rowName.charAt(0))}:</dt>
                    <dd>
                      {rowIndex}
                      {rowToken ? ` “${rowToken}”` : ""}
                    </dd>
                    <dt>{cap(colName.charAt(0))}:</dt>
                    <dd>
                      {colIndex}
                      {colToken ? ` “${colToken}”` : ""}
                    </dd>
                    <dt>{valueLabel}:</dt>
                    <dd className="mint">{valueText}</dd>
                  </dl>
                </div>
              </aside>
            )}
            {stageMode === "overview" && upperTriangleZero && (
              <p className="matrix-annotation">
                <em>Causal mask</em>
                <span>(weights above the diagonal are 0)</span>
              </p>
            )}
            <div className="matrix-scale">
              <span className="matrix-scale-label">
                {attention
                  ? "Attention weight (post-softmax)"
                  : "Activation value"}
              </span>
              <div className="matrix-ramp-wrap">
                <div
                  className={
                    attention ? "matrix-ramp attention" : "matrix-ramp signed"
                  }
                />
                <div className="matrix-ramp-ticks">
                  {attention
                    ? Array.from({ length: 11 }, (_, i) => (
                        <span key={i}>{(i / 10).toFixed(1)}</span>
                      ))
                    : [`−${fmt(abs)}`, "0", `+${fmt(abs)}`].map((t) => (
                        <span key={t}>{t}</span>
                      ))}
                </div>
              </div>
              <span className="matrix-divider" />
              <span className="matrix-pink">
                <i /> Pink = selected
              </span>
            </div>
          </>
        ) : (
          <div className="matrix-empty">
            <GridFour size={44} />
            <h3>No matrix captured yet</h3>
            <p>
              {extracted.error ||
                "Run a text attention probe, or select a recorded tensor event. Only retained values are displayed."}
            </p>
          </div>
        )}
        <footer className="matrix-timeline" aria-label="Layer / ordinal events">
          <span className="matrix-timeline-label">
            Layer / Ordinal Events{" "}
            <Info
              size={15}
              weight="regular"
              aria-hidden="true"
              title="Each dot is a decoder layer. Click one to look at that layer: attention is re-probed there and the layer output view jumps to it."
            />
          </span>
          <div className="matrix-track">
            {series && series.points.length > 1 && !stops[0]?.repeated && (
              <Sparkline series={series} activeLayer={currentLayer} />
            )}
            <div className="matrix-track-dots">
              <span className="matrix-track-line" />
              <OrdinalDot label="Input" />
              {condense(stopPositions, stepIndex).map((entry, i) =>
                entry === "gap" ? (
                  <span key={`gap${i}`} className="matrix-ellipsis">
                    ···
                  </span>
                ) : (
                  <OrdinalDot
                    key={entry}
                    label={stops[entry].label}
                    active={entry === stepIndex}
                    pending={
                      busy &&
                      stops[entry].number === layer &&
                      entry !== stepIndex
                    }
                    onClick={() => pickLayer(stops[entry])}
                  />
                ),
              )}
              <OrdinalDot label="Output" />
            </div>
          </div>
          <span
            className="matrix-step"
            title={
              stepIndex >= 0
                ? "Which layer the stage shows; click a dot or use the Layer select to move"
                : undefined
            }
          >
            {stepIndex >= 0
              ? `${currentStop.name[0].toUpperCase()}${currentStop.name.slice(1)} · ${stepIndex + 1} of ${stepTotal}`
              : event?.denoiser_call !== undefined
                ? `Call ${event.denoiser_call} · ${eventIndex + 1} of ${events.length}`
                : event?.step !== undefined
                  ? `Step ${event.step} · ${eventIndex + 1} of ${events.length}`
                  : events.length
                    ? `Event ${eventIndex + 1} of ${events.length}`
                    : "No events"}
          </span>
        </footer>
      </section>
      <aside className="matrix-inspector">
        <h2>
          <Stack size={22} weight="duotone" /> Inspector
        </h2>
        {profile && (
          <section className="matrix-insight" aria-label="Head insight">
            <h3>
              <Lightbulb size={18} weight="duotone" /> What this head does
            </h3>
            <p className="matrix-insight-label">{profile.label}</p>
            <p className="matrix-insight-text">{profile.description}</p>
            <div className="matrix-chips">
              <span title="Average weight on the first token">
                sink {pct(profile.sinkMass)}
              </span>
              <span title="Average weight on the previous token">
                prev-token {pct(profile.previousTokenMass)}
              </span>
              <span title="Average weight on the same position">
                self {pct(profile.diagonalMass)}
              </span>
              <span title="Mean row entropy">
                {profile.entropyBits.toFixed(2)} bits
              </span>
            </div>
            {attendsTo.length > 0 && (
              <>
                <p className="matrix-insight-sub">
                  {tokens ? `“${rowToken}”` : `${cap(rowName)} ${rowIndex}`}{" "}
                  attends to
                </p>
                <ul className="matrix-attends">
                  {attendsTo.map((entry) => (
                    <li key={entry.key}>
                      <button
                        onClick={() =>
                          setSelected({
                            row: selected.row,
                            col: entry.key - colOffset,
                          })
                        }
                        disabled={
                          entry.key < colOffset ||
                          entry.key >= colOffset + data.cols
                        }
                      >
                        <span className="matrix-attends-token">
                          {tokens
                            ? `“${entry.label}”`
                            : `${colName} ${entry.key}`}
                        </span>
                        <span
                          className="matrix-attends-bar"
                          style={{
                            width: `${Math.max(2, entry.weight * 100)}%`,
                          }}
                        />
                        <b>{pct(entry.weight)}</b>
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </section>
        )}
        {attention && (
          <section className="matrix-insight" aria-label="Head ablation">
            <h3>
              <Lightbulb size={18} weight="duotone" /> What happens without this
              head
            </h3>
            {ablation?.entry ? (
              <>
                <p className={`matrix-insight-label effect-${ablation.effect}`}>
                  {ablation.effect === "decisive"
                    ? "Decisive head"
                    : ablation.effect === "noticeable"
                      ? "Noticeable effect"
                      : "Negligible effect"}
                </p>
                <p className="matrix-insight-text">{ablation.text}</p>
                <div className="matrix-ablation-compare">
                  {[
                    ["With head", ablation.baseline],
                    ["Without head", ablation.entry.top_tokens[0]],
                  ].map(([label, token]) => (
                    <div key={label}>
                      <small>{label}</small>
                      <span className="matrix-ablation-token">
                        “{tokenText(token.text)}”
                      </span>
                      <span className="matrix-ablation-bar">
                        <i
                          style={{
                            width: `${Math.max(1, token.probability * 100)}%`,
                          }}
                        />
                      </span>
                      <b>{pct(token.probability)}</b>
                    </div>
                  ))}
                </div>
                <div className="matrix-chips">
                  <span title="KL divergence from the baseline distribution">
                    {ablation.entry.kl_bits.toFixed(3)} bits shift
                  </span>
                  <span title="Entropy of the next-token distribution without this head">
                    {ablation.entry.entropy_bits.toFixed(2)} bits entropy (was{" "}
                    {trace.ablation.baseline.entropy_bits.toFixed(2)})
                  </span>
                </div>
              </>
            ) : (
              <p className="matrix-insight-text">
                Zero this head's output and re-run the same prompt to measure
                how much the next-token prediction depends on it. This is a
                causal intervention; attention weights alone are not.
              </p>
            )}
            {ablation?.ranked?.length > 1 && (
              <>
                <p className="matrix-insight-sub">
                  Heads of layer {trace.layer} ranked by effect
                  {ablation.complete
                    ? ""
                    : ` (${ablation.ranked.length} of ${trace.ablation.head_count} measured)`}
                </p>
                <ul className="matrix-attends matrix-ablation-rank">
                  {ablation.ranked.slice(0, 8).map((entry) => (
                    <li key={entry.head}>
                      <button
                        onClick={() => onPickHead?.(entry.head)}
                        aria-pressed={entry.head === trace.head}
                        className={entry.head === trace.head ? "picked" : ""}
                        title={`Head ${entry.head}: ${entry.kl_bits.toFixed(3)} bits${entry.top_token_changed ? ", answer changes" : ""}`}
                      >
                        <span className="matrix-attends-token">
                          H{entry.head}
                          {entry.top_token_changed && (
                            <em title="The top token changes without this head">
                              {" "}
                              flips
                            </em>
                          )}
                        </span>
                        <span
                          className="matrix-attends-bar"
                          style={{
                            width: `${Math.max(2, (entry.kl_bits / ablation.peak) * 100)}%`,
                          }}
                        />
                        <b>
                          {entry.kl_bits.toFixed(entry.kl_bits >= 0.1 ? 2 : 3)}
                        </b>
                      </button>
                    </li>
                  ))}
                </ul>
                {ablation.changers.length > 0 && (
                  <p className="matrix-note">
                    Removing{" "}
                    {ablation.changers.length === 1
                      ? `head ${ablation.changers[0]}`
                      : `heads ${ablation.changers.join(", ")}`}{" "}
                    on its own changes the predicted token.
                  </p>
                )}
              </>
            )}
            <div className="matrix-ablation-actions">
              <button
                disabled={busy || !canAttention || !onAblate}
                onClick={() => onAblate?.("head")}
              >
                Ablate H{trace.head}
              </button>
              <button
                disabled={busy || !canAttention || !onAblate}
                onClick={() => onAblate?.("layer")}
                title="Zero each head of this layer in turn"
              >
                Sweep all{" "}
                {trace.head_count ? `${trace.head_count} heads` : "heads"}
              </button>
            </div>
            <p className="matrix-note">
              Zero ablation at every position; one extra forward pass per head.
              Effects of several heads together are not additive.
            </p>
          </section>
        )}
        {attribution && (
          <section className="matrix-insight" aria-label="Token attribution">
            <h3>
              What the last token looked at{" "}
              <small>
                {measured
                  ? "(every layer and head, averaged)"
                  : "(attention rollout)"}
              </small>
            </h3>
            <div className="matrix-attribution">
              {attribution.map((value, index) => {
                const peak = Math.max(...attribution, 1e-9);
                return (
                  <span
                    key={index}
                    title={`${index}: ${pct(value)} of rolled-out attention`}
                    style={{
                      "--heat": Math.min(1, value / peak),
                    }}
                  >
                    {trace.tokens?.[index]
                      ? tokenLabel(trace.tokens[index], index)
                      : index}
                    <small>{pct(value)}</small>
                  </span>
                );
              })}
            </div>
            <p className="matrix-note">
              {measured
                ? `Measured attention from the final query position “${tokenLabel(trace.tokens?.at(-1), (trace.tokens?.length || 1) - 1)}” to each key, averaged over all ${trace.layer_count ?? ""} layers and ${trace.head_count ?? ""} heads. Attention is not a causal attribution.`
                : `Rollout multiplies head-averaged attention across all ${trace.layer_count ?? ""} layers with a residual term. It estimates information flow; it is not a causal attribution.`}
            </p>
          </section>
        )}
        {gallery.length > 0 && (
          <section className="matrix-insight" aria-label="Head gallery">
            <h3>
              All {gallery.length} heads of layer {trace.layer}{" "}
              <small>(click to view)</small>
            </h3>
            <div className="matrix-gallery">
              {gallery.map((item) => (
                <button
                  key={item.index}
                  className={item.index === trace.head ? "picked" : ""}
                  onClick={() => onPickHead?.(item.index)}
                  title={`Head ${item.index}: ${item.profile?.label}`}
                  aria-pressed={item.index === trace.head}
                >
                  <HeadThumb matrix={item.matrix} />
                  <span>H{item.index}</span>
                  <small>{item.profile?.label}</small>
                  {ablationByHead.has(item.index) && (
                    <small
                      className={`matrix-gallery-effect ${ablationByHead.get(item.index).top_token_changed ? "flips" : ""}`}
                      title="Bits of shift when this head is zeroed"
                    >
                      {ablationByHead.get(item.index).kl_bits.toFixed(2)} b
                    </small>
                  )}
                </button>
              ))}
            </div>
          </section>
        )}
        {stats && (
          <section className="matrix-insight" aria-label="Activation insight">
            <h3>
              <Lightbulb size={18} weight="duotone" /> Activation insight
            </h3>
            <p className="matrix-insight-text">
              {describeActivation(
                stats,
                event?.module?.replace(/^model\./, "") || null,
                event?.magnitude,
              )}
            </p>
            <div className="matrix-chips">
              <span title="Root mean square of captured values">
                RMS {fmtShort(stats.rms)}
              </span>
              <span title="Largest magnitude divided by RMS">
                peak {stats.outlierRatio.toFixed(1)}× RMS
              </span>
              <span title="Share of values within 1% of RMS of zero">
                {pct(stats.sparsity)} near zero
              </span>
              {!event?.magnitude && (
                <span title="Share of negative values">
                  {pct(stats.negativeShare)} negative
                </span>
              )}
            </div>
            {stats.outliers.length > 0 && (
              <>
                <p className="matrix-insight-sub">
                  Largest values (flat index)
                </p>
                <div className="matrix-chips outliers">
                  {stats.outliers.map((o) => (
                    <button
                      key={o.index}
                      onClick={() => {
                        const width = data?.dimensions[1] || 1;
                        const row = Math.floor(o.index / width);
                        const col = o.index % width;
                        if (
                          data &&
                          row >= rowOffset &&
                          row < rowOffset + data.rows &&
                          col >= colOffset &&
                          col < colOffset + data.cols
                        )
                          setSelected({
                            row: row - rowOffset,
                            col: col - colOffset,
                          });
                      }}
                    >
                      #{o.index} <b>{fmtShort(o.value)}</b>
                    </button>
                  ))}
                </div>
              </>
            )}
            {moduleDeltas.length > 1 && (
              <>
                <p className="matrix-insight-sub">
                  Change between consecutive{" "}
                  {events[0]?.denoiser_call !== undefined
                    ? "denoiser calls"
                    : "steps"}{" "}
                  (RMS of difference)
                </p>
                <DeltaBars
                  deltas={moduleDeltas}
                  activeIndex={eventIndex}
                  onPick={onPickEvent}
                />
              </>
            )}
          </section>
        )}
        <h3>{data ? title : "Tensor"}</h3>
        <dl className="matrix-facts">
          <dt>Source</dt>
          <dd>{attention ? "layer output (Q × K)" : data?.source || "—"}</dd>
          <dt>Axes</dt>
          <dd>{data ? `${rowName} × ${colName}` : "—"}</dd>
          {event?.slice_label && (
            <>
              <dt>Slice</dt>
              <dd>
                <div>
                  {event.slice_label}
                  <small className="slice-of">
                    of {event.source_shape.join(" × ")}
                  </small>
                  {event.slice_note && (
                    <small className="slice-of">{event.slice_note}</small>
                  )}
                </div>
              </dd>
            </>
          )}
          <dt>Shape</dt>
          <dd>{data?.shape.join(" × ") || "—"}</dd>
          <dt className="selection">Selection</dt>
          <dd className="selection">
            {data
              ? tokens
                ? `“${rowToken}” (${rowIndex}) → “${colToken}” (${colIndex})`
                : `${rowName} ${rowIndex}, ${colName} ${colIndex}`
              : "—"}
          </dd>
          <dt>{valueLabel}</dt>
          <dd>
            <span className="matrix-badge">{valueText}</span>
          </dd>
        </dl>
        {data && (
          <>
            <h3>
              {attention ? "Attention heatmap" : "Value heatmap"}{" "}
              <small>(linked)</small>
            </h3>
            <div
              className={`matrix-heatmap-figure ${tokens ? "with-tokens" : ""}`}
            >
              <span className="matrix-heatmap-ylabel">
                {tokens ? "query token" : axisTitle(data.axes[0])}
              </span>
              <div className="matrix-heatmap-yticks">
                {ticks(data.rows, rowOffset, Boolean(tokens)).map(
                  ({ i, label }) => (
                    <span
                      key={i}
                      className={i === selected.row ? "picked" : ""}
                      style={{ top: `${((i + 0.5) / data.rows) * 100}%` }}
                    >
                      {label}
                    </span>
                  ),
                )}
              </div>
              <div
                className="matrix-heatmap"
                style={{
                  gridTemplateColumns: `repeat(${data.cols},1fr)`,
                  aspectRatio: `${data.cols} / ${data.rows}`,
                }}
              >
                {data.matrix.map((row, r) =>
                  row.map((v, c) => (
                    <button
                      key={`${r}:${c}`}
                      data-row={r}
                      data-col={c}
                      tabIndex={
                        selected.row === r && selected.col === c ? 0 : -1
                      }
                      onKeyDown={(e) => {
                        const delta = {
                          ArrowUp: [-1, 0],
                          ArrowDown: [1, 0],
                          ArrowLeft: [0, -1],
                          ArrowRight: [0, 1],
                        }[e.key];
                        if (!delta) return;
                        e.preventDefault();
                        const nextRow = Math.max(
                          0,
                          Math.min(data.rows - 1, r + delta[0]),
                        );
                        const nextCol = Math.max(
                          0,
                          Math.min(data.cols - 1, c + delta[1]),
                        );
                        setSelected({ row: nextRow, col: nextCol });
                        e.currentTarget.parentElement
                          .querySelector(
                            `[data-row="${nextRow}"][data-col="${nextCol}"]`,
                          )
                          ?.focus();
                      }}
                      aria-label={`${rowName} ${r + rowOffset}, ${colName} ${c + colOffset}: ${data.states[r][c] === "captured" ? fmt(v) : data.states[r][c]}`}
                      title={`${labelFor(r + rowOffset) ?? r + rowOffset} → ${labelFor(c + colOffset) ?? c + colOffset}: ${data.states[r][c] === "captured" ? fmt(v) : data.states[r][c]}`}
                      className={
                        selected.row === r && selected.col === c
                          ? "picked"
                          : selected.row === r || selected.col === c
                            ? "lit"
                            : ""
                      }
                      style={{ background: color(v) }}
                      onClick={() => setSelected({ row: r, col: c })}
                    />
                  )),
                )}
              </div>
              <div className="matrix-colorbar">
                <div
                  className={
                    attention
                      ? "matrix-ramp attention vertical"
                      : "matrix-ramp signed vertical"
                  }
                />
                <div className="matrix-colorbar-ticks">
                  {(attention
                    ? ["1.0", "0.8", "0.6", "0.4", "0.2", "0.0"]
                    : [`+${fmtShort(abs)}`, "0", `−${fmtShort(abs)}`]
                  ).map((t) => (
                    <span key={t}>{t}</span>
                  ))}
                </div>
              </div>
              <div className="matrix-heatmap-xticks">
                {ticks(data.cols, colOffset, Boolean(tokens)).map(
                  ({ i, label }) => (
                    <span
                      key={i}
                      className={i === selected.col ? "picked" : ""}
                      style={{ left: `${((i + 0.5) / data.cols) * 100}%` }}
                    >
                      {label}
                    </span>
                  ),
                )}
              </div>
              <span className="matrix-heatmap-xlabel">
                {tokens ? "key token" : axisTitle(data.axes[1])}
              </span>
            </div>
            {windowed && (
              <div className="matrix-paging">
                <span>Rows</span>
                <button
                  aria-label="Previous rows"
                  disabled={!rowOffset}
                  onClick={() => page("row", -WINDOW)}
                >
                  ‹
                </button>
                <span>
                  {rowOffset}–{rowOffset + data.rows - 1}
                </span>
                <button
                  aria-label="Next rows"
                  disabled={rowOffset + data.rows >= data.dimensions[0]}
                  onClick={() => page("row", WINDOW)}
                >
                  ›
                </button>
                <span>Columns</span>
                <button
                  aria-label="Previous columns"
                  disabled={!colOffset}
                  onClick={() => page("col", -WINDOW)}
                >
                  ‹
                </button>
                <span>
                  {colOffset}–{colOffset + data.cols - 1}
                </span>
                <button
                  aria-label="Next columns"
                  disabled={colOffset + data.cols >= data.dimensions[1]}
                  onClick={() => page("col", WINDOW)}
                >
                  ›
                </button>
              </div>
            )}
            <h3>
              Cell detail{" "}
              <small>
                ({rowName} {rowIndex}, {colName} {colIndex})
              </small>
            </h3>
            <dl className="matrix-facts">
              <dt>{cap(rowName)} (row)</dt>
              <dd>
                {rowIndex}
                {rowToken ? ` “${rowToken}”` : ""}
              </dd>
              <dt>{cap(colName)} (column)</dt>
              <dd>
                {colIndex}
                {colToken ? ` “${colToken}”` : ""}
              </dd>
              <dt>{valueLabel}</dt>
              <dd>
                <span className="matrix-badge">{valueText}</span>
              </dd>
            </dl>
            <h3>
              <Clock size={18} /> Layer/Ordinal Event
            </h3>
            <dl className="matrix-facts">
              <dt>Event</dt>
              <dd>{eventLabel}</dd>
              <dt>Step (ordinal)</dt>
              <dd>
                {stepIndex >= 0
                  ? `${stepIndex + 1} / ${stepTotal}`
                  : events.length
                    ? `${eventIndex + 1} / ${events.length}`
                    : "—"}
              </dd>
              <dt>Recorded</dt>
              <dd>
                <Check size={15} weight="bold" className="mint" />{" "}
                {attention ? "Attention weight" : "Activation prefix"}
              </dd>
            </dl>
            <p className="matrix-note">
              {data.capturedCount.toLocaleString()} /{" "}
              {data.totalElements.toLocaleString()} values retained.{" "}
              {data.fixedIndices.length > 0 &&
                `Leading indices: ${data.fixedIndices.join(", ")}. `}
              {event?.slice_note ? `${event.slice_note} ` : ""}
              Uncaptured and non-finite values are not plotted. No attention
              mask is inferred.
            </p>
          </>
        )}
        <div className="matrix-capture">
          <label>
            Head
            <input
              aria-label="Attention head"
              type="number"
              min="0"
              max="1024"
              value={head}
              onChange={(e) => setHead(Number(e.target.value))}
            />
          </label>
          <button disabled={busy || !canAttention} onClick={onCapture}>
            <Eye size={16} /> Capture L{layer} · H{head}
          </button>
        </div>
        <p className="matrix-footnote">Measured values</p>
      </aside>
    </div>
  );
}

function OrdinalDot({ label, active, pending, onClick }) {
  const Tag = onClick ? "button" : "span";
  return (
    <Tag
      className={`matrix-dot ${active ? "active" : ""} ${pending ? "pending" : ""}`}
      onClick={onClick}
      aria-current={active ? "step" : undefined}
      aria-label={onClick ? `Select layer ${label}` : undefined}
    >
      <i />
      <small>{label}</small>
    </Tag>
  );
}

/** A per-layer metric drawn above the ordinal track. */
function Sparkline({ series, activeLayer }) {
  const points = series.points;
  const layers = points.map((p) => p.layer);
  const minLayer = Math.min(...layers),
    maxLayer = Math.max(...layers);
  const values = points.map((p) => p.value);
  const lo = Math.min(...values),
    hi = Math.max(...values);
  const w = 100,
    h = 26;
  const x = (layer) =>
    maxLayer === minLayer
      ? 50
      : ((layer - minLayer) / (maxLayer - minLayer)) * w;
  const y = (v) =>
    hi === lo ? h / 2 : h - 3 - ((v - lo) / (hi - lo)) * (h - 6);
  const path = points
    .map((p, i) => `${i ? "L" : "M"}${x(p.layer)},${y(p.value)}`)
    .join(" ");
  const active = points.find((p) => p.layer === activeLayer);
  return (
    <div className="matrix-sparkline" aria-label={series.label}>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        <path d={path} />
        {active && <circle cx={x(active.layer)} cy={y(active.value)} r="1.6" />}
      </svg>
      <span>
        {series.label}
        {active ? `: ${fmtShort(active.value)}` : ""}
      </span>
    </div>
  );
}

function DeltaBars({ deltas, activeIndex, onPick }) {
  const peak = Math.max(...deltas.map((d) => d.delta ?? 0), 1e-9);
  return (
    <div className="matrix-deltas">
      {deltas.map((d) => (
        <button
          key={d.index}
          className={d.index === activeIndex ? "picked" : ""}
          title={
            d.delta === null
              ? `Call ${d.ordinal}: first observation`
              : `Call ${d.ordinal}: Δ RMS ${fmtShort(d.delta)} (${pct(d.relativeDelta)} of RMS)`
          }
          onClick={() => onPick?.(d.index)}
        >
          <i
            style={{
              height: `${d.delta === null ? 4 : Math.max(4, (d.delta / peak) * 100)}%`,
            }}
          />
          <small>{d.ordinal}</small>
        </button>
      ))}
    </div>
  );
}

function HeadThumb({ matrix }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !matrix?.length) return;
    const n = matrix.length,
      m = matrix[0].length;
    canvas.width = m;
    canvas.height = n;
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(m, n);
    for (let r = 0; r < n; r++)
      for (let c = 0; c < m; c++) {
        const v = Math.max(0, Math.min(1, matrix[r][c] ?? 0));
        const rgb = mix(v).match(/\d+/g).map(Number);
        const at = (r * m + c) * 4;
        image.data[at] = rgb[0];
        image.data[at + 1] = rgb[1];
        image.data[at + 2] = rgb[2];
        image.data[at + 3] = 255;
      }
    ctx.putImageData(image, 0, 0);
  }, [matrix]);
  return <canvas ref={ref} aria-hidden="true" />;
}

/** Keep the first three, last one, and a neighbourhood around the active step. */
function condense(list, activeIndex) {
  if (list.length <= 10) return list;
  const keep = new Set([0, 1, 2, list.length - 1]);
  if (activeIndex >= 0)
    for (let i = activeIndex - 1; i <= activeIndex + 1; i++)
      if (i >= 0 && i < list.length) keep.add(i);
  const out = [];
  let gap = false;
  list.forEach((value, i) => {
    if (keep.has(i)) {
      out.push(value);
      gap = false;
    } else if (!gap) {
      out.push("gap");
      gap = true;
    }
  });
  return out;
}
