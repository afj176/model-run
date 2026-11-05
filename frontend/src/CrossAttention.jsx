import { useEffect, useMemo, useRef, useState } from "react";
import { Lightbulb } from "@phosphor-icons/react";
import { crossAttentionStory, pct, pctFine, tokenText } from "./insights";

/** Where the prompt is read: measured image-to-text attention maps per token. */
export default function CrossAttention({ trace }) {
  const section = trace?.cross_attention;
  const events = section?.events || [];
  const [eventIndex, setEventIndex] = useState(0);
  const [frame, setFrame] = useState(0);
  const [token, setToken] = useState(null);
  const [overlay, setOverlay] = useState(true);
  const event = events[Math.min(eventIndex, events.length - 1)] || null;
  const labels = section?.text_labels || null;
  const story = useMemo(
    () => (event ? crossAttentionStory(event, labels) : null),
    [event, labels],
  );
  useEffect(() => {
    setEventIndex(0);
    setToken(null);
    setFrame(0);
  }, [trace?.cross_attention]);
  if (!section) return null;
  const picked = token ?? story?.index ?? 0;
  // Latent frames map to decoded frames by position; the two have different frame counts.
  const decoded = trace?.preview_frames?.frames;
  const frameCount = event?.frames ?? 1;
  const frameImage =
    decoded?.length && frameCount > 1
      ? decoded[Math.round((frame / (frameCount - 1)) * (decoded.length - 1))]
      : decoded?.[0];
  const audioStream = event?.stream === "audio";
  // An audio strip has no place on an image.
  const preview = audioStream
    ? null
    : frameImage
      ? `data:image/png;base64,${frameImage}`
      : trace?.preview?.mime === "image/png"
        ? `data:image/png;base64,${trace.preview.base64}`
        : null;
  const label = (i) => (labels?.[i] ? `“${tokenText(labels[i])}”` : `#${i}`);
  const frames = event?.frames ?? 1;
  const cells = event ? event.grid[0] * event.grid[1] : 0;
  const slice = (values) =>
    frames > 1 && Array.isArray(values)
      ? values.slice(frame * cells, (frame + 1) * cells)
      : values;
  const focus = story
    ? crossAttentionStory(event, labels, picked, frame)
    : null;
  // Runs of repeated special tokens (padding) collapse into one chip and leave the ranking.
  const isPad = (i) =>
    labels?.[i] !== undefined &&
    /^<\|?(endoftext|pad)\|?>$/i.test(labels[i]) &&
    i > 0 &&
    labels[i - 1] === labels[i];
  const padCount = event
    ? event.token_mass.filter((_, i) => isPad(i)).length
    : 0;
  const rankedVisible = story ? story.ranked.slice(0, 6) : [];
  return (
    <div className="xattn">
      <div className="xattn-head">
        <h2>
          {audioStream
            ? "Where the audio reads the prompt"
            : "Where the prompt is read"}
          <small>
            {section.selection} · measured attention probabilities ·{" "}
            {event ? `${event.heads} heads averaged` : "no maps recorded"}
          </small>
        </h2>
        {frames > 1 && (
          <label className="xattn-call">
            Frame
            <select
              value={frame}
              onChange={(e) => setFrame(Number(e.target.value))}
              aria-label="Video frame"
            >
              {(event.frame_indices || [...Array(frames).keys()]).map(
                (source, i) => (
                  <option key={i} value={i}>
                    {source}
                    {event.source_frames && event.source_frames > frames
                      ? ` of ${event.source_frames}`
                      : ""}
                  </option>
                ),
              )}
            </select>
          </label>
        )}
        {events.length > 1 && (
          <label className="xattn-call">
            Denoiser call
            <select
              value={eventIndex}
              onChange={(e) => setEventIndex(Number(e.target.value))}
            >
              {events.map((e, i) => (
                <option key={i} value={i}>
                  {e.denoiser_call}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {!event && (
        <p className="muted">
          No attention call matched {section.selection}.{" "}
          {section.skipped?.[0]?.reason}
        </p>
      )}
      {event && focus && (
        <>
          <p className="xattn-story">
            <Lightbulb size={16} weight="duotone" /> {focus.text}
          </p>
          <div className="xattn-body">
            <div className="xattn-map">
              <HeatCanvas
                values={slice(event.maps[picked])}
                grid={event.grid}
                image={overlay ? preview : null}
              />
              <div className="xattn-map-caption">
                <b>{label(picked)}</b>
                <span>
                  {audioStream
                    ? `peak ${pctFine(focus.peak)} at step ${focus.peakCell[1] + 1} of ${event.grid[1]} · time runs left to right, from ${event.source_grid[1]} audio tokens`
                    : `peak ${pctFine(focus.peak)} at row ${focus.peakCell[0] + 1}, col ${focus.peakCell[1] + 1} · ${event.grid[0]}×${event.grid[1]} grid from ${event.source_grid[0]}×${event.source_grid[1]} tokens`}
                </span>
                {preview && (
                  <label className="xattn-toggle">
                    <input
                      type="checkbox"
                      checked={overlay}
                      onChange={(e) => setOverlay(e.target.checked)}
                    />
                    over image
                  </label>
                )}
              </div>
              {event.attention_kind === "joint" && (
                <>
                  <HeatCanvas
                    values={slice(event.text_mass)}
                    grid={event.grid}
                    image={overlay ? preview : null}
                    small
                  />
                  <div className="xattn-map-caption">
                    <b>Text share</b>
                    <span>
                      how much of each region's attention goes to text at all
                      (mean {pct(focus.textShare)})
                    </span>
                  </div>
                </>
              )}
            </div>
            <div className="xattn-tokens">
              <p className="xattn-sub">
                Tokens by how much the {audioStream ? "audio" : "image"} reads
                them <small>({event.batch})</small>
              </p>
              <div className="xattn-strip" role="list">
                {event.token_mass.map((mass, i) =>
                  isPad(i) ? null : (
                    <button
                      key={i}
                      role="listitem"
                      className={i === picked ? "picked" : ""}
                      style={{ "--heat": Math.min(1, mass / story.peakMass) }}
                      title={`${label(i)}: ${pct(mass)} mean attention per region`}
                      onClick={() => setToken(i)}
                    >
                      {labels?.[i] ? tokenText(labels[i]) : i}
                    </button>
                  ),
                )}
                {padCount > 0 && (
                  <span className="xattn-pad" title="Repeated padding tokens">
                    +{padCount} padding
                  </span>
                )}
              </div>
              <ol className="xattn-rank">
                {rankedVisible.map((entry) => (
                  <li key={entry.index}>
                    <button onClick={() => setToken(entry.index)}>
                      <span className="xattn-rank-token">
                        {label(entry.index)}
                      </span>
                      <span
                        className="xattn-rank-bar"
                        style={{
                          width: `${Math.max(2, (entry.mass / story.peakMass) * 100)}%`,
                        }}
                      />
                      <b>{pctFine(entry.mass)}</b>
                    </button>
                  </li>
                ))}
              </ol>
            </div>
          </div>
          <p className="micro">
            Probabilities are computed from the exact query and key tensors the
            attention kernel received, averaged over heads and area-pooled to
            the grid. Attention shows where text is read, not what caused{" "}
            {audioStream ? "a sound" : "a pixel"}.{" "}
            {section.limitations?.find((l) =>
              l.includes("attend to each other"),
            )}{" "}
            {!preview &&
            !audioStream &&
            trace?.source === "comfyui_native_denoiser"
              ? "Add Model Run · Attach Output after your VAE Decode to draw these maps over the result. "
              : ""}
            {section.skipped?.length
              ? `${section.skipped.length} call(s) skipped: ${section.skipped[0].reason}.`
              : ""}
          </p>
        </>
      )}
    </div>
  );
}

function HeatCanvas({ values, grid, image, small }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !values?.length) return;
    const [rows, cols] = grid;
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext("2d");
    const peak = Math.max(1e-9, ...values);
    const img = ctx.createImageData(cols, rows);
    for (let i = 0; i < rows * cols; i++) {
      const t = Math.min(1, (values[i] ?? 0) / peak);
      const at = i * 4;
      // ivory → mint → teal, matching the stage palette
      const a = t < 0.5 ? t * 2 : (t - 0.5) * 2;
      const from = t < 0.5 ? [242, 239, 228] : [127, 217, 194];
      const to = t < 0.5 ? [127, 217, 194] : [15, 159, 134];
      img.data[at] = Math.round(from[0] + (to[0] - from[0]) * a);
      img.data[at + 1] = Math.round(from[1] + (to[1] - from[1]) * a);
      img.data[at + 2] = Math.round(from[2] + (to[2] - from[2]) * a);
      img.data[at + 3] = image ? Math.round(60 + 195 * t) : 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [values, grid, image]);
  return (
    <div
      className={`xattn-canvas ${small ? "small" : ""}`}
      style={{ aspectRatio: `${grid[1]} / ${grid[0]}` }}
    >
      {image && <img src={image} alt="" />}
      <canvas ref={ref} aria-label="Attention map" />
    </div>
  );
}
