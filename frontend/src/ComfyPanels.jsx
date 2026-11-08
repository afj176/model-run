import { useEffect, useMemo, useRef, useState } from "react";
import { Lightbulb, Pause, Play } from "@phosphor-icons/react";
import { api } from "./api";
import {
  denoisingStory,
  featureSheet,
  featureStrip,
  fitPoints,
  seriesRange,
  unitTokens,
} from "./viewData";
import { mostOf } from "./extent.js";

const PANELS = [
  ["latent", "Noisy latent", "What the sampler holds at this step"],
  ["noise", "Predicted noise", "What the model would subtract"],
  ["estimate", "Estimate of the final image", "Where this step points"],
  ["positive", "Positive prompt", "The model's estimate from your prompt"],
  ["negative", "Negative prompt", "The estimate from the negative prompt"],
  ["guidance", "Guidance push", "The difference, times the guidance scale"],
];
const jpeg = (data) => `data:image/jpeg;base64,${data}`;
const fixed = (v, digits = 3) =>
  Number.isFinite(v) ? v.toFixed(digits) : "not recorded";

function StepBars({ title, values, picked, onPick, labels }) {
  const { high } = seriesRange(values);
  return (
    <div className="step-bars">
      <h4>{title}</h4>
      <div role="list">
        {values.map((value, i) => (
          <button
            key={i}
            role="listitem"
            className={i === picked ? "active" : ""}
            title={`${labels[i]}: ${fixed(value)}`}
            aria-label={`${labels[i]}: ${fixed(value)}`}
            onClick={() => onPick(i)}
          >
            <i
              style={{
                height: `${Number.isFinite(value) && high > 0 ? Math.max(4, (value / high) * 100) : 0}%`,
              }}
            />
          </button>
        ))}
      </div>
    </div>
  );
}

/**
 * The image the run made, with its progress: a strip of every step's estimate ends at the final
 * decode, and the large picture is whichever one is chosen. Follows the replay's denoiser call.
 */
export function ImageProgress({ trace, call = null, onCall }) {
  const steps = (trace?.denoising?.steps || []).filter(
    (s) => s.images?.estimate,
  );
  const final =
    trace?.preview?.mime === "image/png"
      ? `data:image/png;base64,${trace.preview.base64}`
      : null;
  // The last position is the final image; before it, each recorded step's estimate.
  const last = steps.length;
  const [position, setPosition] = useState(last);
  // The replay's call moves this only when it changes after the run opens; a new run starts at the final image.
  const seen = useRef(call);
  useEffect(() => {
    setPosition(steps.length);
    seen.current = call;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace?.id, steps.length]);
  useEffect(() => {
    if (call === seen.current) return;
    seen.current = call;
    if (call === null || !steps.length) return;
    let found = -1;
    steps.forEach((s, i) => {
      if (s.index <= call) found = i;
    });
    if (found >= 0 && found < steps.length - 1) setPosition(found);
  }, [call, steps]);
  if (!final && !steps.length) return null;
  const atFinal = position >= last || !steps[position];
  const shown = atFinal ? final : jpeg(steps[position].images.estimate);
  const go = (next) => {
    setPosition(next);
    if (next < last) onCall?.(steps[next].index);
  };
  const size =
    trace?.preview?.width && trace?.preview?.height
      ? `${trace.preview.width} × ${trace.preview.height}`
      : "";
  return (
    <div className="image-progress">
      <div className="xattn-head">
        <h2>
          What it made
          <small>
            {atFinal
              ? `the decoded image${size ? ` · ${size}` : ""}`
              : `step ${steps[position].index + 1}${trace.denoising?.total_steps ? ` of ${trace.denoising.total_steps}` : ""} · the sampler's estimate of the final image`}
          </small>
        </h2>
        {steps.length > 0 && (
          <div className="denoising-controls">
            <input
              type="range"
              aria-label="Step, ending at the final image"
              min={0}
              max={last}
              value={Math.min(position, last)}
              onChange={(e) => go(Number(e.target.value))}
            />
            <span>{atFinal ? "final" : `${steps[position].index + 1}`}</span>
          </div>
        )}
      </div>
      <div className="image-progress-body">
        <SourceFigures trace={trace} />
        <figure className="image-progress-main">
          {shown ? (
            <img
              src={shown}
              alt={
                atFinal
                  ? "The decoded image"
                  : `Estimate at step ${steps[position].index + 1}`
              }
            />
          ) : (
            <div className="image-progress-missing">
              The decoded image was not attached to this run. Add Model Run ·
              Attach Output after your VAE Decode, or use Model Run's VAE
              Decode, to keep it.
            </div>
          )}
          <figcaption>
            {atFinal ? (
              <b>Final image</b>
            ) : (
              <b>Step {steps[position].index + 1}</b>
            )}
            {!atFinal && <span>drag to the end for the decoded image</span>}
          </figcaption>
        </figure>
      </div>
      {steps.length > 0 && (
        <div className="image-progress-strip" role="list" aria-label="Steps">
          {steps.map((s, i) => (
            <button
              key={s.index}
              role="listitem"
              className={i === position ? "picked" : ""}
              title={`Step ${s.index + 1}: the estimate of the final image`}
              onClick={() => go(i)}
            >
              <img src={jpeg(s.images.estimate)} alt="" />
              <small>{s.index + 1}</small>
            </button>
          ))}
          {final && (
            <button
              role="listitem"
              className={`final ${atFinal ? "picked" : ""}`}
              title="The decoded image"
              onClick={() => go(last)}
            >
              <img src={final} alt="" />
              <small>final</small>
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/** The sampler's work at each step: the latent, the noise it predicts and where that leaves the image. */
/** The pictures the run read, so what it made can be seen beside where it started. */
export function SourceFigures({ trace }) {
  return (trace?.source_images || []).map((source, i) => (
    <figure key={`source-${i}`} className="source-figure">
      <img
        src={`data:image/png;base64,${source.base64}`}
        alt={`Source image: ${source.file}`}
      />
      <figcaption>
        <b>Source</b>
        <span title={`${source.title} reads ${source.file}`}>
          {source.file.split("/").pop()}
        </span>
      </figcaption>
    </figure>
  ));
}

export function Denoising({ trace, call = null, onCall, lastCall = null }) {
  const section = trace?.denoising;
  const steps = section?.steps || [];
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const at = useRef(0);
  at.current = position;
  useEffect(() => {
    setPosition(Math.max(0, steps.length - 1));
    setPlaying(false);
  }, [trace?.id, steps.length]);
  // A step is one denoiser call (the recorder counts calls as the replay does): follow the replay.
  useEffect(() => {
    if (call === null || !steps.length) return;
    let found = -1;
    steps.forEach((s, i) => {
      if (s.index <= call) found = i;
    });
    if (found >= 0) setPosition(found);
  }, [call, steps]);
  // Only a move made here goes back to the replay, so following it never pulls it back.
  const go = (next) => {
    setPosition(next);
    onCall?.(steps[next]?.index);
  };
  useEffect(() => {
    if (!playing) return undefined;
    const timer = setInterval(() => {
      if (at.current >= steps.length - 1) {
        setPlaying(false);
        return;
      }
      go(at.current + 1);
    }, 500);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, steps.length]);
  if (!steps.length) return null;
  const step = steps[Math.min(position, steps.length - 1)];
  const labels = steps.map((s) => `Step ${s.index + 1}`);
  const hasGuidance = steps.some((s) => Number.isFinite(s.guidance_rms));
  return (
    <div className="denoising">
      <div className="xattn-head">
        <h2>
          How the image came out of the noise
          <small>
            {section.projection === "latent_rgb_factors"
              ? "Quick linear previews, not VAE decodes"
              : "First latent channels, not VAE decodes"}
          </small>
        </h2>
        <div className="denoising-controls">
          <button
            aria-label={playing ? "Pause" : "Play the steps"}
            onClick={() => {
              if (!playing && position >= steps.length - 1) go(0);
              setPlaying(!playing);
            }}
          >
            {playing ? <Pause size={15} /> : <Play size={15} />}
          </button>
          <input
            type="range"
            aria-label="Sampler step"
            min={0}
            max={steps.length - 1}
            value={position}
            onChange={(e) => {
              setPlaying(false);
              go(Number(e.target.value));
            }}
          />
          <span>
            {step.index + 1}
            {section.total_steps ? ` / ${section.total_steps}` : ""}
          </span>
        </div>
      </div>
      <p className="xattn-story">
        <Lightbulb size={16} weight="duotone" />{" "}
        {denoisingStory(section, position)}
      </p>
      {section.skipped && <p className="micro">{section.skipped}</p>}
      {lastCall !== null && step.index > lastCall && (
        <p className="micro">
          Layers were recorded on denoiser calls 0–{lastCall}, so the event
          replay stays on call {call ?? lastCall} while this shows step{" "}
          {step.index + 1}.
        </p>
      )}
      <div className="denoising-grid">
        <SourceFigures trace={trace} />
        {PANELS.filter(([key]) => step.images?.[key]).map(
          ([key, title, hint]) => (
            <figure key={key}>
              <img
                src={jpeg(step.images[key])}
                alt={`${title}, ${labels[position]}`}
              />
              <figcaption>
                <b>{title}</b>
                <span>{hint}</span>
              </figcaption>
            </figure>
          ),
        )}
      </div>
      <dl className="denoising-stats">
        <div>
          <dt>Noise level (sigma)</dt>
          <dd>{fixed(step.sigma, 2)}</dd>
        </div>
        <div>
          <dt>Latent RMS</dt>
          <dd>{fixed(step.rms?.latent)}</dd>
        </div>
        <div>
          <dt>Predicted noise RMS</dt>
          <dd>{fixed(step.rms?.noise)}</dd>
        </div>
        <div>
          <dt>Estimate moved by</dt>
          <dd>{fixed(step.estimate_change)}</dd>
        </div>
        {hasGuidance && (
          <div>
            <dt>Guidance push RMS</dt>
            <dd>{fixed(step.guidance_rms)}</dd>
          </div>
        )}
      </dl>
      <div className="step-bars-row">
        <StepBars
          title="How far the estimate moved"
          values={steps.map((s) => s.estimate_change)}
          picked={position}
          onPick={setPosition}
          labels={labels}
        />
        {hasGuidance && (
          <StepBars
            title="How hard guidance pushed"
            values={steps.map((s) => s.guidance_rms)}
            picked={position}
            onPick={setPosition}
            labels={labels}
          />
        )}
      </div>
      <p className="micro">{section.limitations?.join(" ")}</p>
    </div>
  );
}

function Heat({ values, rows, cols, label, size = 72, onClick, active }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !values?.length) return;
    const context = canvas.getContext("2d");
    const image = context.createImageData(cols, rows);
    const magnitudes = values.map((v) =>
      Number.isFinite(v) ? Math.abs(v) : 0,
    );
    const peak = Math.max(1e-9, ...magnitudes);
    magnitudes.forEach((value, i) => {
      const t = value / peak;
      // One hue, light to dark: stronger is darker.
      image.data[i * 4] = 238 - t * 186;
      image.data[i * 4 + 1] = 231 - t * 193;
      image.data[i * 4 + 2] = 243 - t * 80;
      image.data[i * 4 + 3] = 255;
    });
    context.putImageData(image, 0, 0);
  }, [values, rows, cols]);
  const canvas = (
    <canvas
      ref={ref}
      width={cols}
      height={rows}
      style={{
        width: size,
        height: Math.round((size * rows) / cols),
        imageRendering: "pixelated",
      }}
    />
  );
  return onClick ? (
    <button
      className={`feature-cell ${active ? "active" : ""}`}
      title={label}
      aria-label={label}
      onClick={onClick}
    >
      {canvas}
    </button>
  ) : (
    <span className="feature-cell" title={label}>
      {canvas}
    </span>
  );
}

/** What each block of a convolutional model produced, block by block and channel by channel. */
export function FeatureMaps({ trace }) {
  const events = trace?.events;
  const strip = useMemo(() => featureStrip(events), [events]);
  const [picked, setPicked] = useState(0);
  useEffect(() => setPicked(0), [trace?.id]);
  const module = strip[Math.min(picked, strip.length - 1)];
  const sheet = useMemo(
    () =>
      module
        ? featureSheet(events.find((e) => e.module === module.module))
        : null,
    [events, module],
  );
  if (strip.length < 1) return null;
  return (
    <div className="feature-maps">
      <h2>
        Feature maps by block
        <small>
          Magnitude only. Each map is scaled to itself, so compare shapes, not
          darkness.
        </small>
      </h2>
      <div
        className="feature-strip"
        role="list"
        aria-label="Blocks in the order they ran"
      >
        {strip.map((item, i) => (
          <div role="listitem" key={item.module}>
            <Heat
              values={item.values}
              rows={item.rows}
              cols={item.cols}
              label={`${item.module}, ${item.shape?.join(" × ")}`}
              active={i === picked}
              onClick={() => setPicked(i)}
            />
            <small>{item.shape?.slice(-2).join("×")}</small>
          </div>
        ))}
      </div>
      {sheet && (
        <>
          <h3>
            {module.module}
            <small>
              {" "}
              {sheet.channels.length} of {sheet.size.toLocaleString()} channels
              recorded, evenly spaced
            </small>
          </h3>
          <div className="feature-sheet">
            {sheet.channels.map((channel) => (
              <div key={channel.index}>
                <Heat
                  values={channel.values}
                  rows={sheet.rows}
                  cols={sheet.cols}
                  size={64}
                  label={`Channel ${channel.index}`}
                />
                <small>{channel.index}</small>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

const TOKEN_HUES = 6;

/** A text encoder's work: how the prompt was cut up, how each layer responded, and which tokens move a unit. */
export function TextEncoderPanel({ trace, stagePath = null, onPath }) {
  const [otherCaptures, setOtherCaptures] = useState([]);
  const section = trace?.text_encoder;
  const encoders = section?.encoders || [];
  const [encoderAt, setEncoderAt] = useState(0);
  const [cell, setCell] = useState(null);
  const [mapLayer, setMapLayer] = useState(null);
  useEffect(() => {
    setEncoderAt(0);
    setCell(null);
    setMapLayer(null);
  }, [trace?.id]);
  const asked = Boolean(cell);
  useEffect(() => {
    // Other prompts through the same encoder, only once someone asks about a unit.
    if (!asked) return undefined;
    let alive = true;
    api("state")
      .then((state) =>
        Promise.all(
          state.traces
            .filter((t) => t.name === trace.model_name && t.id !== trace.id)
            .slice(-8)
            .map((t) => api(`traces/${t.id}`)),
        ),
      )
      .then((others) => alive && setOtherCaptures(others))
      .catch(() => alive && setOtherCaptures([]));
    return () => {
      alive = false;
    };
  }, [asked, trace?.id, trace?.model_name]);
  const encoder = encoders[Math.min(encoderAt, encoders.length - 1)];
  const layers = encoder?.layers || [];
  // A start token can be many times stronger than any word; scale to the words and say so.
  const { peak, clipped } = useMemo(() => {
    const special = (encoder?.tokens || []).map((t) => t.special);
    const scale = mostOf(
      1e-9,
      layers.flatMap((l) => l.token_rms.filter((_, i) => !special[i])),
    );
    const strongest = Math.max(0, ...layers.flatMap((l) => l.token_rms));
    return { peak: scale, clipped: strongest > scale ? strongest : null };
  }, [layers, encoder?.tokens]);
  const points = useMemo(
    () => fitPoints(encoder?.map?.points, encoder?.tokens),
    [encoder?.map?.points, encoder?.tokens],
  );
  // The stage's layer, when it belongs to this encoder, is the one the token map shows.
  const stageAt = layers.findIndex((l) => l.path === stagePath);
  useEffect(() => {
    if (stageAt >= 0) setMapLayer(stageAt);
  }, [stageAt]);
  if (!encoder) return null;
  const tokens = encoder.tokens;
  const layer = cell ? layers[cell.layer] : null;
  const strongest = layer
    ? layer.units[
        layer.unit_values[cell.token]
          .map((v, i) => [Math.abs(v), i])
          .sort((a, b) => b[0] - a[0])[0][1]
      ]
    : null;
  const moved = layer ? unitTokens(layer, strongest, tokens) : [];
  const elsewhere = (otherCaptures || [])
    .map((other) => {
      const match = other.text_encoder?.encoders?.find(
        (e) => e.encoder === encoder.encoder,
      );
      const same = match?.layers?.find((l) => l.path === layer?.path);
      return same?.units?.includes(strongest)
        ? {
            note: other.note,
            top: unitTokens(same, strongest, match.tokens)[0],
          }
        : null;
    })
    .filter(Boolean);
  const shownLayer = mapLayer ?? layers.length - 1;
  return (
    <div className="text-encoder">
      <div className="xattn-head">
        <h2>
          How the text encoder read the prompt
          <small>
            {encoder.hidden_size.toLocaleString()} numbers per token ·{" "}
            {encoder.token_total} tokens in, {tokens.length} before padding
          </small>
        </h2>
        {encoders.length > 1 && (
          <div className="tabs" role="tablist" aria-label="Text encoder">
            {encoders.map((e, i) => (
              <button
                key={e.encoder}
                role="tab"
                aria-selected={i === encoderAt}
                className={i === encoderAt ? "active" : ""}
                onClick={() => {
                  setEncoderAt(i);
                  setCell(null);
                  setMapLayer(null);
                }}
              >
                {e.encoder}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="token-strip" aria-label="Tokens">
        {tokens.map((t, i) => (
          <span key={i} className={`token-chip hue-${i % TOKEN_HUES}`}>
            <b>{t.text.trim() || "␣"}</b>
            <small>{t.id ?? "no id"}</small>
            {t.weight !== 1 && <small>× {t.weight}</small>}
          </span>
        ))}
      </div>
      {encoder.numbered && (
        <p className="micro">
          The tokens could not be matched to this encoder, so they are numbered.
        </p>
      )}
      <div className="text-encoder-body">
        <div className="activity-grid-wrap">
          <h3>
            Activity by layer
            <small> token by layer · pick a cell to see its units</small>
          </h3>
          <div
            className="activity-grid"
            style={{
              gridTemplateColumns: `72px repeat(${tokens.length}, minmax(22px, 1fr))`,
            }}
            role="grid"
          >
            <span />
            {tokens.map((t, i) => (
              <span key={i} className="activity-head" title={t.text}>
                {t.text.trim().slice(0, 5) || "␣"}
              </span>
            ))}
            {layers.map((l, row) => (
              <div key={l.path} role="row" style={{ display: "contents" }}>
                <span className="activity-label" title={l.path}>
                  {l.path.split(".").at(-1)}
                </span>
                {l.token_rms.map((value, column) => (
                  <button
                    key={column}
                    role="gridcell"
                    className={
                      cell?.layer === row && cell?.token === column
                        ? "active"
                        : ""
                    }
                    style={{ "--heat": Math.min(1, value / peak) }}
                    title={`${tokens[column].text} at ${l.path}: ${value.toFixed(3)}`}
                    aria-label={`${tokens[column].text} at ${l.path}: ${value.toFixed(3)}`}
                    onClick={() => setCell({ layer: row, token: column })}
                  />
                ))}
              </div>
            ))}
          </div>
          {clipped && (
            <p className="micro">
              The shading is scaled to the words. Start and end tokens go off
              the scale, reaching {clipped.toFixed(1)} against {peak.toFixed(1)}{" "}
              for the strongest word.
            </p>
          )}
          {layer && (
            <div className="unit-card">
              <b>
                Unit {strongest} in {layer.path.split(".").at(-1)}
              </b>
              <span>
                The strongest unit for “{tokens[cell.token].text.trim()}” in
                this layer. Tokens that move it, in this prompt:
              </span>
              <ol>
                {moved.slice(0, 5).map((m) => (
                  <li key={m.index}>
                    “{m.text.trim()}” <small>{m.value.toFixed(2)}</small>
                  </li>
                ))}
              </ol>
              {elsewhere.length > 0 && (
                <span>
                  In {elsewhere.length} other capture
                  {elsewhere.length === 1 ? "" : "s"}:{" "}
                  {elsewhere
                    .slice(0, 4)
                    .map((e) => `“${e.top?.text.trim()}”`)
                    .join(", ")}
                </span>
              )}
              <small>
                This only says what moved it here. It does not say what the unit
                does in general.
              </small>
            </div>
          )}
        </div>
        <div className="token-map">
          <h3>
            Tokens moving through the layers
            <small>
              {" "}
              PCA shows{" "}
              {(
                (encoder.map.explained[0] + (encoder.map.explained[1] ?? 0)) *
                100
              ).toFixed(1)}
              % of the variance
            </small>
          </h3>
          <svg
            viewBox="0 0 100 100"
            role="img"
            aria-label="Each token's path through the layers"
          >
            {tokens.map((t, i) =>
              t.special ? null : (
                <polyline
                  key={i}
                  className={`token-path hue-${i % TOKEN_HUES}`}
                  points={points
                    .slice(0, shownLayer + 1)
                    .map((l) => l[i].join(","))
                    .join(" ")}
                />
              ),
            )}
            {tokens.map((t, i) => {
              const at = points[shownLayer]?.[i];
              return at && !t.special ? (
                <g key={i}>
                  <circle
                    className={`token-dot hue-${i % TOKEN_HUES}`}
                    cx={at[0]}
                    cy={at[1]}
                    r="1.3"
                  />
                  <text x={at[0] + 1.6} y={at[1] + 1}>
                    {t.text.trim().slice(0, 10)}
                  </text>
                </g>
              ) : null;
            })}
          </svg>
          <label className="token-map-layer">
            Layer {layers[shownLayer]?.path.split(".").at(-1)}
            <input
              type="range"
              aria-label="Layer"
              min={0}
              max={layers.length - 1}
              value={shownLayer}
              onChange={(e) => {
                const i = Number(e.target.value);
                setMapLayer(i);
                onPath?.(layers[i]?.path);
              }}
            />
          </label>
        </div>
      </div>
      <p className="micro">{section.limitations?.join(" ")}</p>
    </div>
  );
}
