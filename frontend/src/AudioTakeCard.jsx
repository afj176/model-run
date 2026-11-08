import { useEffect, useMemo, useRef, useState } from "react";
import {
  FolderOpen,
  Microphone,
  Pause,
  Play,
  Stop,
  X,
} from "@phosphor-icons/react";
import {
  formatTime,
  peaksFromBuffer,
  wavFromBuffer,
  windowOf,
} from "./audioData.js";

/** Whisper hears this many seconds at once. */
export const WINDOW_SECONDS = 30;
/** Recordings stop here on their own. */
const MOST_RECORD_SECONDS = 120;
const COLUMNS = 240;

/**
 * The audio input as one card: a take (a file you chose or a clip you recorded) as a waveform
 * with the 30-second window the model hears laid on it, a record button with a live level, and
 * the details of what the model gets. Several takes sit in a strip; one is selected and run.
 */
export default function AudioTakeCard({
  onFile,
  onStart,
  start = 0,
  heard = null,
  busy = false,
  onError,
}) {
  const [takes, setTakes] = useState([]);
  const [selected, setSelected] = useState(0);
  const take = takes[selected] ?? null;
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const [livePeaks, setLivePeaks] = useState([]);
  const [playing, setPlaying] = useState(false);
  const [playhead, setPlayhead] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef(null);
  const recorder = useRef(null);
  const stream = useRef(null);
  const timer = useRef(null);
  const audio = useRef(null);
  const waveBox = useRef(null);
  const canvas = useRef(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    if (!waveBox.current) return;
    const watch = new ResizeObserver(([entry]) =>
      setWidth(Math.floor(entry.contentRect.width)),
    );
    watch.observe(waveBox.current);
    return () => watch.disconnect();
  }, []);
  useEffect(
    () => () => {
      stopTracks();
      audio.current?.pause();
      takes.forEach((t) => URL.revokeObjectURL(t.url));
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  /** Decode a file in the browser for its waveform and facts, then make it the selected take. */
  async function addTake(file, source) {
    try {
      const bytes = await file.arrayBuffer();
      const context = new AudioContext();
      let decoded = null;
      try {
        decoded = await context.decodeAudioData(bytes.slice(0));
      } catch {
        decoded = null; // the server may still read it (FFmpeg formats); the card shows no waveform
      }
      await context.close();
      const next = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        file,
        name: file.name,
        source,
        at: new Date(),
        url: URL.createObjectURL(file),
        duration: decoded?.duration ?? null,
        rate: decoded?.sampleRate ?? null,
        channels: decoded?.numberOfChannels ?? null,
        peaks: decoded ? peaksFromBuffer(decoded, COLUMNS) : null,
      };
      setTakes((list) => {
        const all = [...list, next];
        setSelected(all.length - 1);
        return all;
      });
      onStart?.(0);
      onFile?.(file);
    } catch (err) {
      onError?.(`The file could not be read: ${err.message}`);
    }
  }
  function choose(index) {
    const t = takes[index];
    if (!t) return;
    stopPlay();
    setSelected(index);
    onStart?.(0);
    onFile?.(t.file);
  }
  function remove(index) {
    const t = takes[index];
    if (!t) return;
    URL.revokeObjectURL(t.url);
    const rest = takes.filter((_, i) => i !== index);
    setTakes(rest);
    const nextIndex = Math.max(
      0,
      Math.min(
        rest.length - 1,
        index === selected
          ? index - 1
          : selected > index
            ? selected - 1
            : selected,
      ),
    );
    setSelected(nextIndex);
    if (rest[nextIndex]) {
      onStart?.(0);
      onFile?.(rest[nextIndex].file);
    } else onFile?.(null);
  }

  // Recording: MediaRecorder for the clip, an analyser for the level ring and the live waveform.
  function stopTracks() {
    clearInterval(timer.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
  }
  async function startRecording() {
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.current = media;
      const context = new AudioContext();
      const analyser = context.createAnalyser();
      analyser.fftSize = 2048;
      context.createMediaStreamSource(media).connect(analyser);
      const chunks = [];
      const rec = new MediaRecorder(media);
      recorder.current = rec;
      rec.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      rec.onstop = async () => {
        stopTracks();
        setRecording(false);
        setLevel(0);
        try {
          const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
          const decoded = await context.decodeAudioData(
            await blob.arrayBuffer(),
          );
          await context.close();
          const stamp = new Date()
            .toISOString()
            .slice(11, 19)
            .replace(/:/g, "-");
          await addTake(
            new File([wavFromBuffer(decoded)], `recording-${stamp}.wav`, {
              type: "audio/wav",
            }),
            "mic",
          );
        } catch (err) {
          onError?.(`The recording could not be read: ${err.message}`);
        }
      };
      rec.start();
      const began = Date.now();
      const samples = new Float32Array(analyser.fftSize);
      setLivePeaks([]);
      setElapsed(0);
      setRecording(true);
      timer.current = setInterval(() => {
        analyser.getFloatTimeDomainData(samples);
        let top = 0;
        for (let i = 0; i < samples.length; i++)
          if (Math.abs(samples[i]) > top) top = Math.abs(samples[i]);
        setLevel(top);
        setLivePeaks((p) => (p.length < 600 ? [...p, top] : p));
        const s = (Date.now() - began) / 1000;
        setElapsed(s);
        if (s >= MOST_RECORD_SECONDS) stopRecording();
      }, 100);
    } catch (err) {
      stopTracks();
      onError?.(
        err?.name === "NotAllowedError"
          ? "The browser did not allow the microphone. Allow it for this site and try again."
          : err?.name === "NotFoundError"
            ? "No microphone was found."
            : `The microphone could not be opened: ${err.message}`,
      );
    }
  }
  function stopRecording() {
    clearInterval(timer.current);
    if (recorder.current?.state === "recording") recorder.current.stop();
  }

  // Playback of the window, with a playhead on the waveform.
  const window_ = useMemo(
    () => (take ? windowOf(start, take.duration ?? 0, WINDOW_SECONDS) : null),
    [take, start],
  );
  function stopPlay() {
    audio.current?.pause();
    setPlaying(false);
    setPlayhead(null);
  }
  function togglePlay() {
    if (!take) return;
    if (playing) return stopPlay();
    const el = audio.current ?? (audio.current = new Audio());
    el.src = take.url;
    el.currentTime = window_.start;
    el.ontimeupdate = () => {
      setPlayhead(el.currentTime);
      if (el.currentTime >= window_.end - 0.05) stopPlay();
    };
    el.onended = stopPlay;
    el.play()
      .then(() => setPlaying(true))
      .catch((err) => onError?.(`Playback failed: ${err.message}`));
  }

  // The waveform, the window frame and the playhead.
  const H = 72;
  useEffect(() => {
    const cv = canvas.current;
    if (!cv || !width) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(width * dpr);
    cv.height = Math.round(H * dpr);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, H);
    const peaks = recording ? livePeaks : take?.peaks;
    const total = recording
      ? Math.max(WINDOW_SECONDS, elapsed)
      : (take?.duration ?? 0);
    if (!peaks || !peaks.length || !total) return;
    const colW = recording ? (width * 0.1) / 60 : width / peaks.length;
    // Columns outside the window are dimmed, so the part the model hears stands out.
    const win = recording ? windowOf(0, total, WINDOW_SECONDS) : window_;
    for (let k = 0; k < peaks.length; k++) {
      const x = recording ? k * colW : k * colW;
      if (x > width) break;
      const t = recording ? k * 0.1 : (k / peaks.length) * total;
      const inside = win && t >= win.start && t < win.end;
      const a = Math.max(1, peaks[k] * (H / 2 - 4));
      ctx.fillStyle = recording ? "#ff9ec9" : inside ? "#7fd9c2" : "#7fd9c255";
      ctx.fillRect(x + 0.5, H / 2 - a, Math.max(1, colW - 1), a * 2);
    }
    if (!recording && win && take?.duration) {
      const x0 = (win.start / total) * width;
      const x1 = (win.end / total) * width;
      ctx.strokeStyle = "#ff3d8f";
      ctx.lineWidth = 2;
      ctx.strokeRect(x0 + 1, 1, Math.max(4, x1 - x0 - 2), H - 2);
      ctx.fillStyle = "#ff3d8f";
      ctx.fillRect(x0 - 3, H / 2 - 11, 6, 22);
      ctx.fillRect(x1 - 3, H / 2 - 11, 6, 22);
      if (playhead !== null) {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect((playhead / total) * width - 0.5, 0, 1, H);
      }
    }
  }, [width, take, start, window_, recording, livePeaks, elapsed, playhead]);

  // Dragging the frame moves the start; the keyboard nudges it.
  function timeAt(clientX) {
    const b = waveBox.current.getBoundingClientRect();
    return ((clientX - b.left) / b.width) * (take?.duration ?? 0);
  }
  const canSlide = take?.duration > WINDOW_SECONDS + 1e-6;
  function dragStart(e) {
    if (!canSlide || recording) return;
    e.preventDefault();
    stopPlay();
    const grab = timeAt(e.clientX);
    const inside = grab >= window_.start && grab < window_.end;
    const offset = inside ? grab - window_.start : WINDOW_SECONDS / 2;
    const place = (clientX) =>
      onStart?.(
        Math.round(
          windowOf(timeAt(clientX) - offset, take.duration, WINDOW_SECONDS)
            .start * 10,
        ) / 10,
      );
    place(e.clientX);
    setDragging(true);
    const move = (ev) => place(ev.clientX);
    const up = () => {
      setDragging(false);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }
  function keys(e) {
    if (!canSlide) return;
    const step = e.shiftKey ? 5 : 1;
    const to = {
      ArrowLeft: start - step,
      ArrowRight: start + step,
      Home: 0,
      End: take.duration,
    }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    onStart?.(windowOf(to, take.duration, WINDOW_SECONDS).start);
  }

  const supported =
    typeof navigator !== "undefined" &&
    Boolean(navigator.mediaDevices?.getUserMedia) &&
    typeof MediaRecorder !== "undefined";
  const ruler = useMemo(() => {
    const total = recording
      ? Math.max(WINDOW_SECONDS, elapsed)
      : (take?.duration ?? 0);
    if (!total) return [];
    const n = width < 420 ? 3 : 6;
    return Array.from({ length: n + 1 }, (_, i) => (i / n) * total);
  }, [take, width, recording, elapsed]);
  const heardHere = heard && take && heard.name === take.name ? heard : null;

  return (
    <section
      className={`take-card ${dragging ? "dragging" : ""}`}
      aria-label="Audio input"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        const file = e.dataTransfer.files?.[0];
        if (file) addTake(file, "file");
      }}
    >
      <div className="take-record">
        {recording ? (
          <button
            type="button"
            className="take-button recording"
            style={{ "--level": level }}
            onClick={stopRecording}
            aria-label="Stop recording"
            title="Stop recording"
          >
            <Stop size={26} weight="fill" />
          </button>
        ) : (
          <button
            type="button"
            className="take-button"
            onClick={startRecording}
            disabled={busy || !supported}
            aria-label="Record a take from the microphone"
            title={
              supported
                ? `Record from the microphone, up to ${MOST_RECORD_SECONDS} seconds; it stays on this computer`
                : "This browser cannot record audio"
            }
          >
            <Microphone size={28} />
          </button>
        )}
        <span className="take-time mono">
          {recording ? formatTime(elapsed, true) : "Record a take"}
        </span>
        <span className="take-sub">
          {recording ? (
            `stops at ${formatTime(MOST_RECORD_SECONDS)}`
          ) : (
            <button
              type="button"
              className="text-button"
              onClick={() => fileInput.current?.click()}
              disabled={busy}
            >
              <FolderOpen size={13} /> or choose a file
            </button>
          )}
        </span>
        <input
          ref={fileInput}
          hidden
          type="file"
          accept=".wav,.flac,.ogg,.mp3,audio/*"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) addTake(file, "file");
            e.target.value = "";
          }}
        />
      </div>

      <div className="take-wave">
        <div className="take-wave-head">
          <span className="mono">
            {recording
              ? "recording…"
              : take
                ? `${take.name}${take.duration ? ` · ${formatTime(take.duration, true)}` : ""}`
                : "No take yet"}
          </span>
          <span className="take-chips">
            {take && !recording && take.duration > 0 && (
              <button type="button" className="chip" onClick={togglePlay}>
                {playing ? (
                  <Pause size={12} weight="fill" />
                ) : (
                  <Play size={12} weight="fill" />
                )}{" "}
                {playing ? "pause" : window_.whole ? "play" : "play window"}
              </button>
            )}
            {recording ? (
              <span className="chip live">
                <i /> live
              </span>
            ) : take && window_ ? (
              <span className="chip on">
                {window_.whole
                  ? `hears all ${formatTime(window_.used, true)}`
                  : `hears ${formatTime(window_.start)} → ${formatTime(window_.end)}`}
              </span>
            ) : null}
          </span>
        </div>
        <div
          className={`take-wave-plot ${canSlide ? "slides" : ""}`}
          ref={waveBox}
          tabIndex={take ? 0 : -1}
          role={canSlide ? "slider" : undefined}
          aria-label={
            canSlide ? "Where the 30-second window starts" : undefined
          }
          aria-valuemin={canSlide ? 0 : undefined}
          aria-valuemax={
            canSlide ? Math.max(0, take.duration - WINDOW_SECONDS) : undefined
          }
          aria-valuenow={canSlide ? start : undefined}
          onPointerDown={dragStart}
          onKeyDown={keys}
        >
          <canvas ref={canvas} style={{ width, height: H }} />
          {!take && !recording && (
            <div className="take-empty">
              Drop a WAV, FLAC, OGG or MP3 here, choose a file, or record. It
              stays on this computer.
            </div>
          )}
          {take && !take.peaks && !recording && (
            <div className="take-empty">
              The browser could not decode this file for a waveform; the server
              may still read it.
            </div>
          )}
        </div>
        <div className="take-ruler mono">
          {ruler.map((t, i) => (
            <span key={i}>{formatTime(t)}</span>
          ))}
        </div>
      </div>

      <dl className="take-facts">
        {recording ? (
          <>
            <dt>Microphone</dt>
            <dd>default</dd>
            <dt>Model gets</dt>
            <dd className="mono">16 kHz · mono WAV</dd>
            <dt>Window</dt>
            <dd className="mono">first {WINDOW_SECONDS} s</dd>
            <dt>Where</dt>
            <dd>stays in the browser until Run</dd>
          </>
        ) : take ? (
          <>
            <dt>Length</dt>
            <dd className="mono">
              {take.duration ? formatTime(take.duration, true) : "unknown"}
            </dd>
            <dt>Source</dt>
            <dd className="mono">
              {[
                take.channels === 1
                  ? "mono"
                  : take.channels === 2
                    ? "stereo"
                    : take.channels
                      ? `${take.channels} ch`
                      : null,
                heardHere?.source_rate
                  ? `${(heardHere.source_rate / 1000).toFixed(heardHere.source_rate % 1000 ? 1 : 0)} kHz`
                  : null,
                take.source === "mic" ? "microphone" : "file",
              ]
                .filter(Boolean)
                .join(" · ")}
            </dd>
            <dt>Model gets</dt>
            <dd className="mono">16 kHz · mono</dd>
            <dt>Window</dt>
            <dd className="mono">
              {window_.whole
                ? `all ${formatTime(window_.used, true)}`
                : `${formatTime(window_.used)} from ${formatTime(window_.start)}`}
              {canSlide && (
                <label className="take-start">
                  <input
                    type="number"
                    min={0}
                    max={Math.max(0, Math.floor(take.duration - 1))}
                    step={1}
                    value={Math.round(start)}
                    onChange={(e) =>
                      onStart?.(
                        windowOf(
                          Number(e.target.value) || 0,
                          take.duration,
                          WINDOW_SECONDS,
                        ).start,
                      )
                    }
                    aria-label="Start time in seconds"
                  />
                  s
                </label>
              )}
            </dd>
            {window_.leftOut.length > 0 && (
              <>
                <dt>Left out</dt>
                <dd className="mono">
                  {window_.leftOut
                    .map(([a, b]) => `${formatTime(a)}–${formatTime(b)}`)
                    .join(" and ")}
                </dd>
              </>
            )}
            {heardHere && (
              <>
                <dt>Last run heard</dt>
                <dd className="mono">
                  {heardHere.used_seconds} s from {heardHere.start_seconds || 0}{" "}
                  s
                </dd>
              </>
            )}
            <dt>Where</dt>
            <dd>this computer only</dd>
          </>
        ) : (
          <>
            <dt>Model gets</dt>
            <dd className="mono">16 kHz · mono · {WINDOW_SECONDS} s</dd>
            <dt>Where</dt>
            <dd>this computer only</dd>
          </>
        )}
      </dl>

      {takes.length > 1 && (
        <div className="take-strip" role="list" aria-label="Takes">
          {takes.map((t, i) => (
            <div
              key={t.id}
              role="listitem"
              className={`take-tile ${i === selected ? "picked" : ""}`}
            >
              <button
                type="button"
                className="take-tile-body"
                onClick={() => choose(i)}
                title={t.name}
              >
                <span className="take-tile-head mono">
                  <span>
                    {t.source === "mic"
                      ? `mic · ${t.at.toTimeString().slice(0, 5)}`
                      : "file"}
                  </span>
                  <span>{t.duration ? formatTime(t.duration, true) : ""}</span>
                </span>
                <TileWave peaks={t.peaks} />
                <span className="take-tile-name">{t.name}</span>
              </button>
              <button
                type="button"
                className="take-tile-remove"
                onClick={() => remove(i)}
                aria-label={`Remove ${t.name}`}
                title="Remove this take"
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function TileWave({ peaks }) {
  const cols = 60;
  if (!peaks) return <span className="take-tile-wave" />;
  const per = peaks.length / cols;
  const bars = Array.from({ length: cols }, (_, k) => {
    let top = 0;
    for (
      let i = Math.floor(k * per);
      i < Math.max(Math.floor(k * per) + 1, Math.floor((k + 1) * per)) &&
      i < peaks.length;
      i++
    )
      if (peaks[i] > top) top = peaks[i];
    return top;
  });
  return (
    <svg
      className="take-tile-wave"
      viewBox={`0 0 ${cols} 20`}
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      {bars.map((v, k) => (
        <rect
          key={k}
          x={k + 0.2}
          y={10 - Math.max(0.5, v * 9)}
          width={0.6}
          height={Math.max(1, v * 18)}
          fill="#7fd9c2"
        />
      ))}
    </svg>
  );
}
