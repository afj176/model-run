import { useMemo, useState } from "react";
import { pct, tokenText } from "./insights";
import { attentionArcs, headFanOut, samplingWhatIf } from "./viewData";

const label = (text) => (text === undefined ? "?" : tokenText(text) || "∅");

/** Raw scores for the next token, and what temperature, top-K and top-P would make of them. */
export function LogitsPanel({ trace }) {
  const view = trace?.logits_view;
  const [temperature, setTemperature] = useState(1);
  const [topK, setTopK] = useState(50);
  const [topP, setTopP] = useState(0.95);
  const names = useMemo(() => {
    const map = new Map();
    for (const t of trace?.logits_view?.top || [])
      if (t.text !== undefined) map.set(t.token_id, t.text);
    return map;
  }, [trace]);
  const result = useMemo(
    () => samplingWhatIf(view, names, { temperature, topK, topP }),
    [view, names, temperature, topK, topP],
  );
  if (!view || !result) return null;
  const counts = view.histogram.counts;
  const tallest = Math.max(...counts.map((c) => Math.log1p(c)), 1);
  const shown = result.rows.slice(0, 10);
  const peak = Math.max(
    ...shown.map((r) => Math.max(r.probability, r.filtered)),
    1e-9,
  );
  return (
    <div className="logits-panel">
      <h2>
        Scores before the choice
        <small>
          {view.vocab_size.toLocaleString()} tokens scored · what-if on this one
          step
        </small>
      </h2>
      {trace.tokens?.length > 0 && (
        <div className="token-strip" aria-label="Prompt tokens">
          {trace.tokens.map((t, i) => (
            <span key={i} className={`token-chip hue-${i % 6}`}>
              <b>{label(t.text)}</b>
              <small>{t.id ?? "no id"}</small>
            </span>
          ))}
        </div>
      )}
      <div className="logits-body">
        <div>
          <h3>
            All scores <small>token count on a log scale</small>
          </h3>
          <div
            className="logit-histogram"
            role="img"
            aria-label={`Histogram of ${view.vocab_size} logits from ${view.histogram.edges[0]} to ${view.histogram.edges.at(-1)}`}
          >
            {counts.map((count, i) => (
              <i
                key={i}
                title={`${view.histogram.edges[i]} to ${view.histogram.edges[i + 1]}: ${count.toLocaleString()} tokens`}
                style={{
                  height: `${count ? Math.max(3, (Math.log1p(count) / tallest) * 100) : 0}%`,
                }}
              />
            ))}
          </div>
          <p className="micro">
            Scores run from {view.histogram.edges[0].toFixed(1)} to{" "}
            {view.histogram.edges.at(-1).toFixed(1)}, averaging{" "}
            {view.mean.toFixed(1)}. Only the far right tail has any real chance.
          </p>
          <div className="sampling-controls">
            <label>
              Temperature <b>{result.temperature.toFixed(1)}</b>
              <input
                type="range"
                min={0.1}
                max={2}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </label>
            <label>
              Top-K <b>{topK}</b>
              <input
                type="range"
                min={1}
                max={view.top.length}
                value={Math.min(topK, view.top.length)}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </label>
            <label>
              Top-P <b>{topP.toFixed(2)}</b>
              <input
                type="range"
                min={0.05}
                max={1}
                step={0.05}
                value={topP}
                onChange={(e) => setTopP(Number(e.target.value))}
              />
            </label>
          </div>
        </div>
        <div>
          <h3>
            Chances with these settings{" "}
            <small>
              {result.kept} token{result.kept === 1 ? "" : "s"} left ·{" "}
              {result.entropy.toFixed(2)} bits
            </small>
          </h3>
          <ol className="sampling-rows">
            {shown.map((row) => (
              <li key={row.token_id} className={row.kept ? "" : "cut"}>
                <span>{label(row.text)}</span>
                <span
                  className="sampling-bar"
                  title={`${pct(row.probability)} before filters, ${pct(row.filtered)} after`}
                >
                  <i
                    className="before"
                    style={{ width: `${(row.probability / peak) * 100}%` }}
                  />
                  <i
                    className="after"
                    style={{ width: `${(row.filtered / peak) * 100}%` }}
                  />
                </span>
                <span>{row.kept ? pct(row.filtered) : "cut"}</span>
              </li>
            ))}
          </ol>
          <div className="sampling-key">
            <span>
              <i className="before" /> at this temperature
            </span>
            <span>
              <i className="after" /> after top-K and top-P
            </span>
          </div>
          {!result.reachedTopP && (
            <p className="micro">
              The {view.top.length} recorded tokens hold only {pct(result.seen)}{" "}
              at this temperature, so top-P could not be applied exactly.
            </p>
          )}
        </div>
      </div>
      <p className="micro">{view.limitations?.join(" ")}</p>
    </div>
  );
}

/** The residual stream at the last position: its size, and what each layer wrote onto it. */
export function ResidualStream({ lens }) {
  const layers = (lens?.layers || []).filter((l) =>
    Number.isFinite(l.residual_norm),
  );
  if (layers.length < 2 || !layers.some((l) => Number.isFinite(l.written_norm)))
    return null;
  const peakNorm = Math.max(...layers.map((l) => l.residual_norm), 1e-9);
  const peakWritten = Math.max(...layers.map((l) => l.written_norm ?? 0), 1e-9);
  return (
    <div className="residual-stream">
      <h3>
        The residual stream
        <small> last position · each layer reads it and writes an update</small>
      </h3>
      <div className="residual-track" role="list">
        {layers.map((layer) => (
          <div
            role="listitem"
            key={layer.index}
            title={`${layer.label}: size ${layer.residual_norm.toFixed(1)}${Number.isFinite(layer.written_norm) ? `, wrote ${layer.written_norm.toFixed(1)}, ${((layer.cosine_to_previous ?? 0) * 100).toFixed(0)}% aligned with before` : ""}`}
          >
            <i
              className="written"
              style={{
                height: `${((layer.written_norm ?? 0) / peakWritten) * 34}px`,
              }}
            />
            <i
              className="stream"
              style={{
                height: `${Math.max(3, (layer.residual_norm / peakNorm) * 16)}px`,
              }}
            />
            <small>{layer.index === 0 ? "emb" : layer.index - 1}</small>
          </div>
        ))}
      </div>
      <p className="micro">
        The lower band is the size of the stream. The bars above it are how much
        each layer changed it.
      </p>
    </div>
  );
}

/** Memory held by the key-value cache, read from the tensors the model returned at each step. */
export function CacheGrowth({ steps }) {
  const series = (steps || []).filter((s) => s.cache);
  if (series.length < 2) return null;
  const peak = Math.max(...series.map((s) => s.cache.bytes), 1);
  const megabytes = (bytes) => `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  const last = series.at(-1);
  return (
    <div className="cache-growth">
      <h3>
        Key-value cache
        <small>
          {" "}
          {megabytes(last.cache.bytes)} by the last token ·{" "}
          {last.cache.tokens ?? "?"} tokens remembered
        </small>
      </h3>
      <div className="cache-bars" role="list">
        {series.map((step) => (
          <i
            key={step.index}
            role="listitem"
            title={`Token ${step.index}: ${megabytes(step.cache.bytes)}, fed ${step.fed_tokens} token${step.fed_tokens === 1 ? "" : "s"}`}
            style={{
              height: `${Math.max(4, (step.cache.bytes / peak) * 100)}%`,
            }}
          />
        ))}
      </div>
      <p className="micro">
        The first step feeds {series[0].fed_tokens} tokens. Every later step
        feeds {series[1].fed_tokens}, because the cache already holds the rest.
      </p>
    </div>
  );
}

/** Lines between tokens, and how each head spreads its attention from one token. */
export function AttentionLinks({ trace }) {
  const tokens = trace?.tokens;
  const values = trace?.values;
  const [query, setQuery] = useState(null);
  const count = values?.length || 0;
  const arcs = useMemo(() => attentionArcs(values, 14), [values]);
  const from = query ?? Math.max(0, count - 1);
  const heads = useMemo(
    () => headFanOut(trace?.heads, from),
    [trace?.heads, from],
  );
  if (
    trace?.kind !== "attention_capture" ||
    count < 2 ||
    !Array.isArray(tokens)
  )
    return null;
  const step = 100 / count;
  const x = (i) => step * (i + 0.5);
  const peak = Math.max(...arcs.map((a) => a.weight), 1e-9);
  return (
    <div className="attention-links">
      <h2>
        Which tokens look at which
        <small>
          Layer {trace.layer} ·{" "}
          {trace.head_mean ? "head mean" : `head ${trace.head}`} · the{" "}
          {arcs.length} strongest links, not counting a token to itself
        </small>
      </h2>
      <div className="arc-canvas">
        <svg viewBox="0 0 100 34" preserveAspectRatio="none" aria-hidden="true">
          {arcs.map((arc) => {
            const a = x(arc.from);
            const b = x(arc.to);
            const lift = Math.min(32, 4 + Math.abs(a - b) * 0.75);
            const lit =
              query === null || arc.from === query || arc.to === query;
            return (
              <path
                key={`${arc.from}-${arc.to}`}
                d={`M ${a} 34 Q ${(a + b) / 2} ${34 - lift} ${b} 34`}
                style={{
                  strokeWidth: 0.2 + (arc.weight / peak) * 0.9,
                  opacity: lit ? 0.85 : 0.12,
                }}
              />
            );
          })}
        </svg>
        <div
          className="arc-tokens"
          style={{ gridTemplateColumns: `repeat(${count}, 1fr)` }}
        >
          {tokens.map((t, i) => (
            <button
              key={i}
              className={i === from && query !== null ? "active" : ""}
              title={`Token ${i}, id ${t.id ?? "unknown"}`}
              onClick={() => setQuery(query === i ? null : i)}
            >
              {label(t.text)}
            </button>
          ))}
        </div>
      </div>
      {heads.length > 1 && (
        <>
          <h3>
            Every head from “{label(tokens[from]?.text)}”
            <small> pick a token above to change it</small>
          </h3>
          <div className="fan-out" role="list">
            {heads.map((head) => (
              <div role="listitem" key={head.head}>
                <b>Head {head.head}</b>
                <span className="fan-row" aria-hidden="true">
                  {head.row.map((value, i) => (
                    <i
                      key={i}
                      style={{ opacity: Math.max(0.06, value) }}
                      title={`${label(tokens[i]?.text)}: ${pct(value)}`}
                    />
                  ))}
                </span>
                <small>
                  “{label(tokens[head.peak]?.text)}” {pct(head.weight)}
                </small>
              </div>
            ))}
          </div>
          <p className="micro">
            This shows where each head looked on this input. It does not say
            what a head is for.
          </p>
        </>
      )}
    </div>
  );
}
