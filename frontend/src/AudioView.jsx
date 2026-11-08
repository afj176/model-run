import { useEffect, useMemo, useRef, useState } from "react";
import { Play, Pause } from "@phosphor-icons/react";
import {
  attentionPastAudio,
  pct,
  pctFine,
  speechTokenStory,
  tokenText,
  validIndex,
} from "./insights";

// One hue per plot: magnitude runs light to dark on the paper surface.
const LOW = [252, 247, 239];
const HIGH = [76, 64, 157];
const shade = (t) =>
  LOW.map((c, i) =>
    Math.round(c + (HIGH[i] - c) * Math.max(0, Math.min(1, t))),
  );

function paint(canvas, rows, cols, read) {
  if (!canvas) return;
  canvas.width = cols;
  canvas.height = rows;
  const context = canvas.getContext("2d");
  const image = context.createImageData(cols, rows);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const [red, green, blue] = shade(read(r, c));
      const at = (r * cols + c) * 4;
      image.data[at] = red;
      image.data[at + 1] = green;
      image.data[at + 2] = blue;
      image.data[at + 3] = 255;
    }
  context.putImageData(image, 0, 0);
}

const isSpecial = (text) => /^<\|.*\|>$/.test((text || "").trim());

/** What the model heard, and where in the audio it read for each token it wrote. */
export default function AudioView({ trace, stageLayer = null, onLayer }) {
  const view = trace?.audio_view;
  const layers = view?.layers || [];
  const tokens = view?.tokens || [];
  const [layerAt, setLayerAt] = useState(null);
  const [picked, setPicked] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const melRef = useRef(null);
  const mapRef = useRef(null);
  const audioRef = useRef(null);
  useEffect(() => {
    setLayerAt(null);
    setPicked(null);
    setHovered(null);
    setTime(0);
    setPlaying(false);
  }, [trace]);
  // On a decoder layer, the stage and this view show the same layer.
  useEffect(() => {
    const i = layers.findIndex((l) => l.layer === stageLayer);
    if (i >= 0) setLayerAt(i);
  }, [stageLayer, layers]);
  const layerIndex = validIndex(layerAt, layers.length, layers.length - 1);
  const layer = layers[layerIndex];
  const hover = hovered && tokens[hovered.row] ? hovered : null;
  // Rows stay readable on short transcripts and bounded on long ones.
  const rowHeight = Math.max(
    10,
    Math.min(15, Math.floor(900 / Math.max(1, tokens.length))),
  );
  const defaultToken = Math.max(
    0,
    tokens.findIndex((t) => !isSpecial(t.text) && (t.text || "").trim()),
  );
  const index = validIndex(picked, tokens.length, defaultToken);
  const peak = useMemo(() => {
    let most = 1e-9;
    for (const row of layer?.maps || [])
      for (const v of row) if (v > most) most = v;
    return most;
  }, [layer]);
  useEffect(() => {
    if (!view) return;
    const { bins, cols, values } = view.mel;
    let low = Infinity,
      high = -Infinity;
    for (const v of values)
      if (v !== null) {
        if (v < low) low = v;
        if (v > high) high = v;
      }
    const span = Math.max(1e-9, high - low);
    // Low frequencies at the bottom, as a spectrogram is read.
    paint(melRef.current, bins, cols, (r, c) => {
      const v = values[(bins - 1 - r) * cols + c];
      return v === null ? 0 : (v - low) / span;
    });
  }, [view]);
  useEffect(() => {
    if (!view || !layer) return;
    paint(
      mapRef.current,
      tokens.length,
      view.time_bins,
      (r, c) => (layer.maps[r][c] ?? 0) / peak,
    );
  }, [view, layer, tokens.length, peak]);
  if (!view) return null;
  const past = attentionPastAudio(layers);
  const story = layer
    ? speechTokenStory(layer.maps[index], layer.outside[index], view.seconds)
    : null;
  const seek = (seconds, play = false) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = Math.max(0, Math.min(view.seconds, seconds));
    setTime(audio.currentTime);
    if (play) audio.play().catch(() => setPlaying(false));
  };
  const clickTime = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    seek(((e.clientX - box.left) / box.width) * view.seconds, playing);
  };
  const moveOverMap = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const row = Math.floor(
      ((e.clientY - box.top) / box.height) * tokens.length,
    );
    const col = Math.floor(
      ((e.clientX - box.left) / box.width) * view.time_bins,
    );
    if (tokens[row] && col >= 0 && col < view.time_bins)
      setHovered({
        row,
        seconds: ((col + 0.5) / view.time_bins) * view.seconds,
        value: layer.maps[row][col],
      });
  };
  const pickFromMap = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const row = Math.floor(
      ((e.clientY - box.top) / box.height) * tokens.length,
    );
    if (tokens[row]) setPicked(row);
  };
  const ticks = Math.min(6, Math.max(1, Math.floor(view.seconds)));
  // The axis counts from where the window starts in the file, not from the window's own start.
  const offset = trace?.input_media?.start_seconds || 0;
  const playhead = `${(time / view.seconds) * 100}%`;
  return (
    <div className="audio-view">
      <div className="audio-view-head">
        <h2>
          What it heard
          <small>
            {trace?.input_media?.source_seconds !== undefined &&
            (trace.input_media.start_seconds ||
              trace.input_media.used_seconds <
                trace.input_media.source_seconds - 1e-3)
              ? `${view.seconds.toFixed(1)} s of ${trace.input_media.source_seconds} s, from ${trace.input_media.start_seconds || 0} s`
              : `${view.seconds.toFixed(1)} s`}{" "}
            · log-mel input · measured decoder attention
          </small>
        </h2>
        {view.audio && (
          <button
            className="audio-play"
            aria-label={playing ? "Pause audio" : "Play audio"}
            onClick={() => {
              const audio = audioRef.current;
              if (!audio) return;
              if (audio.paused) audio.play().catch(() => setPlaying(false));
              else audio.pause();
            }}
          >
            {playing ? (
              <Pause size={15} weight="fill" />
            ) : (
              <Play size={15} weight="fill" />
            )}
            {time.toFixed(1)} s
          </button>
        )}
        {view.audio && (
          <audio
            ref={audioRef}
            preload="metadata"
            src={`data:${view.audio.mime};base64,${view.audio.base64}`}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
            onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
          />
        )}
      </div>
      {past?.doubtful && (
        <p className="audio-doubt" role="note">
          This transcript may not come from the audio. {pct(past.share)} of the
          final layer's attention landed past the end of the audio, which in our
          tests was typical of silence, tones and noise rather than speech.
        </p>
      )}
      <div className="audio-grid">
        <span className="audio-row-label">Spectrogram</span>
        <div className="audio-plot" onClick={clickTime}>
          <canvas
            ref={melRef}
            className="audio-mel"
            role="img"
            aria-label={`Log-mel spectrogram with ${view.mel.bins} frequency bands`}
          />
          <i className="audio-playhead" style={{ left: playhead }} />
        </div>
        {layer ? (
          <>
            <div
              className="audio-tokens"
              role="list"
              aria-label="Written tokens"
            >
              {tokens.map((t, i) => (
                <button
                  key={t.step}
                  className={i === index ? "selected" : ""}
                  aria-pressed={i === index}
                  title={`${pctFine(t.probability)} probability`}
                  style={{
                    height: `${rowHeight}px`,
                    lineHeight: `${rowHeight}px`,
                  }}
                  onClick={() => setPicked(i)}
                >
                  {tokenText(t.text)}
                </button>
              ))}
            </div>
            <div
              className="audio-plot audio-map"
              style={{ height: `${tokens.length * rowHeight}px` }}
              onClick={pickFromMap}
              onMouseMove={moveOverMap}
              onMouseLeave={() => setHovered(null)}
            >
              <canvas
                ref={mapRef}
                role="img"
                aria-label="Attention of each written token over audio time"
              />
              <i
                className="audio-row-mark"
                style={{
                  top: `${index * rowHeight}px`,
                  height: `${rowHeight}px`,
                }}
              />
              <i className="audio-playhead" style={{ left: playhead }} />
            </div>
          </>
        ) : (
          <p className="muted audio-unavailable">
            {view.attention_unavailable}
          </p>
        )}
        <span />
        <div className="audio-axis" aria-hidden="true">
          {Array.from({ length: ticks + 1 }, (_, i) => (
            <span key={i}>
              {(offset + (i / ticks) * view.seconds).toFixed(1)} s
            </span>
          ))}
        </div>
      </div>
      {layer && (
        <div className="audio-readout">
          {layers.length > 1 && (
            <label className="image-layer">
              Decoder layer {layer.layer}
              <input
                type="range"
                min="0"
                max={layers.length - 1}
                value={layerIndex}
                aria-label="Decoder layer"
                onChange={(e) => {
                  const i = Number(e.target.value);
                  setLayerAt(i);
                  onLayer?.(layers[i]?.layer);
                }}
              />
            </label>
          )}
          <p className="muted">
            {hover
              ? `“${tokenText(tokens[hover.row].text)}” at ${hover.seconds.toFixed(2)} s: ${pctFine(hover.value ?? 0)} of its attention.`
              : `“${tokenText(tokens[index].text)}” was written with ${pct(tokens[index].probability)} probability, reading most around ${story.peakSeconds.toFixed(2)} s (${pctFine(story.peak)} of its attention in that slice). ${pct(story.outside)} of its attention landed past the end of the audio.`}
          </p>
          {view.audio && !hover && (
            <button
              className="text-button"
              onClick={() => seek(Math.max(0, story.peakSeconds - 0.3), true)}
            >
              Play from there
            </button>
          )}
        </div>
      )}
      <p className="micro">{view.limitations?.join(" ")}</p>
    </div>
  );
}
