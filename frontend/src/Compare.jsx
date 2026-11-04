import { useEffect, useMemo, useState } from "react";
import { GitDiff } from "@phosphor-icons/react";
import { api } from "./api";
import {
  activationStats,
  attentionHeadProfile,
  compareMatrices,
  pct,
  tokenLabel,
} from "./insights";
import { mix } from "./MatrixWorkspace";

const short = (n) => (Number.isFinite(n) ? n.toPrecision(3) : "—");
const layerOf = (module) => {
  const m =
    /(?:layers|blocks|transformer_blocks|double_blocks|single_blocks|joint_blocks)\.(\d+)/.exec(
      module || "",
    );
  return m ? Number(m[1]) : null;
};

function describe(trace) {
  if (!trace) return "—";
  if (trace.kind === "attention_capture")
    return `${trace.model_name || "model"} · L${trace.layer} H${trace.head} · ${trace.shape.join("×")}`;
  return `${trace.model_name || "model"} · ${trace.events?.length || 0} events`;
}

/** Side-by-side comparison of two retained traces. All numbers come from the recorded values. */
export default function Compare({ traceA, traces, traceId }) {
  const [otherId, setOtherId] = useState("");
  const [traceB, setTraceB] = useState(null);
  const [error, setError] = useState("");
  const candidates = (traces || []).filter((t) => t.id !== traceId);
  useEffect(() => {
    if (!otherId) {
      setTraceB(null);
      return;
    }
    let alive = true;
    api(`traces/${otherId}`)
      .then((t) => alive && setTraceB(t))
      .catch((e) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [otherId]);
  const attention =
    traceA?.kind === "attention_capture" &&
    traceB?.kind === "attention_capture";
  const diff = useMemo(
    () => (attention ? compareMatrices(traceA.values, traceB.values) : null),
    [attention, traceA, traceB],
  );
  const profiles = useMemo(
    () =>
      attention
        ? [
            attentionHeadProfile(traceA.values),
            attentionHeadProfile(traceB.values),
          ]
        : null,
    [attention, traceA, traceB],
  );
  const layerRms = (trace) => {
    const map = new Map();
    for (const e of trace?.events || []) {
      const layer = layerOf(e.module);
      const stats = activationStats(e.values);
      if (layer === null || !stats || map.has(layer)) continue;
      map.set(layer, stats.rms);
    }
    return map;
  };
  const rmsA = useMemo(() => layerRms(traceA), [traceA]);
  const rmsB = useMemo(() => layerRms(traceB), [traceB]);
  const sharedLayers = [...rmsA.keys()]
    .filter((l) => rmsB.has(l))
    .sort((a, b) => a - b);
  const topA = traceA?.output?.top_tokens || [];
  const topB = traceB?.output?.top_tokens || [];
  return (
    <section className="compare panel" aria-label="Compare traces">
      <header>
        <h2>
          <GitDiff size={22} /> Compare
        </h2>
        <div className="compare-pick">
          <span>
            <b>A</b> {describe(traceA)}
          </span>
          <label>
            <b>B</b>
            <select
              value={otherId}
              onChange={(e) => setOtherId(e.target.value)}
            >
              <option value="">Choose a retained trace…</option>
              {candidates.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name} · {t.kind.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>
      {error && <p className="error">{error}</p>}
      {!traceA && (
        <p className="muted">Open a capture first, then pick a second one.</p>
      )}
      {traceA && !traceB && (
        <p className="muted">
          Pick a second retained trace. Attention captures compare weight by
          weight; module captures compare per-layer activation scale; both
          compare next-token predictions when present.
        </p>
      )}
      {attention && (
        <div className="compare-grid">
          {[traceA, traceB].map((t, i) => (
            <div key={i} className="compare-card">
              <h3>
                {i ? "B" : "A"} · L{t.layer} H{t.head}
              </h3>
              <Thumb matrix={t.values} />
              <p className="compare-label">{profiles[i].label}</p>
              <div className="matrix-chips">
                <span>sink {pct(profiles[i].sinkMass)}</span>
                <span>prev {pct(profiles[i].previousTokenMass)}</span>
                <span>{profiles[i].entropyBits.toFixed(2)} bits</span>
              </div>
            </div>
          ))}
          <div className="compare-card">
            <h3>A − B</h3>
            {diff ? (
              <>
                <Thumb matrix={diff.diff} signed />
                <p className="compare-label">
                  mean |Δ| {short(diff.meanAbs)} · largest {short(diff.maxDiff)}{" "}
                  at query {diff.at[0]}
                  {traceA.tokens?.[diff.at[0]] &&
                    ` “${tokenLabel(traceA.tokens[diff.at[0]])}”`}{" "}
                  → key {diff.at[1]}
                  {traceA.tokens?.[diff.at[1]] &&
                    ` “${tokenLabel(traceA.tokens[diff.at[1]])}”`}
                </p>
                <p className="micro">
                  Teal where A attends more, coral where B attends more.
                </p>
              </>
            ) : (
              <p className="muted">
                Shapes differ ({traceA.shape.join("×")} vs{" "}
                {traceB.shape.join("×")}
                ), so weights cannot be subtracted. Profiles above still
                compare.
              </p>
            )}
          </div>
        </div>
      )}
      {traceB && sharedLayers.length > 0 && (
        <div className="compare-card wide">
          <h3>Activation RMS per layer</h3>
          <table className="compare-table">
            <thead>
              <tr>
                <th>Layer</th>
                <th>A</th>
                <th>B</th>
                <th>B / A</th>
              </tr>
            </thead>
            <tbody>
              {sharedLayers.map((l) => (
                <tr key={l}>
                  <td>L{l}</td>
                  <td>{short(rmsA.get(l))}</td>
                  <td>{short(rmsB.get(l))}</td>
                  <td
                    className={
                      Math.abs(Math.log(rmsB.get(l) / rmsA.get(l))) >
                      Math.log(1.5)
                        ? "hot"
                        : ""
                    }
                  >
                    {short(rmsB.get(l) / rmsA.get(l))}×
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {traceB && (topA.length || topB.length) > 0 && (
        <div className="compare-card wide">
          <h3>Next-token prediction</h3>
          <div className="compare-tokens">
            {[topA, topB].map((list, i) => (
              <div key={i}>
                <b>{i ? "B" : "A"}</b>
                {list.slice(0, 5).map((t) => {
                  const other = (i ? topA : topB).find(
                    (o) => o.token_id === t.token_id,
                  );
                  return (
                    <span key={t.token_id} className={other ? "" : "only"}>
                      “{t.text ?? t.token_id}” {pct(t.probability)}
                      {!other && <small>only here</small>}
                    </span>
                  );
                })}
                {!list.length && (
                  <span className="muted">no prediction recorded</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function Thumb({ matrix, signed }) {
  const n = matrix.length,
    m = matrix[0]?.length || 0;
  const peak = signed
    ? Math.max(...matrix.flat().map((v) => Math.abs(v ?? 0)), 1e-9)
    : 1;
  return (
    <div
      className="compare-thumb"
      style={{
        gridTemplateColumns: `repeat(${m},1fr)`,
        aspectRatio: `${m}/${n}`,
      }}
      aria-hidden="true"
    >
      {matrix.map((row, r) =>
        row.map((v, c) => (
          <i
            key={`${r}:${c}`}
            style={{
              background:
                v === null
                  ? "#2a2342"
                  : signed
                    ? v >= 0
                      ? `rgba(15,159,134,${Math.min(1, v / peak)})`
                      : `rgba(241,109,131,${Math.min(1, -v / peak)})`
                    : mix(Math.max(0, Math.min(1, v))),
            }}
          />
        )),
      )}
    </div>
  );
}
