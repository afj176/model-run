import { useEffect, useRef, useState } from "react";
import { Microphone, Stop } from "@phosphor-icons/react";
import { wavFromBuffer } from "./audioData.js";

/** Recordings stop here on their own; the start time picks the 30 seconds a model hears. */
const MOST_SECONDS = 120;

/**
 * Record from the computer's microphone and hand the recording over as a WAV file.
 * Nothing leaves the browser until the run sends it to the local server.
 */
export default function AudioRecorder({
  onRecorded,
  onError,
  disabled = false,
}) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const recorder = useRef(null);
  const stream = useRef(null);
  const timer = useRef(null);
  const supported =
    typeof navigator !== "undefined" &&
    Boolean(navigator.mediaDevices?.getUserMedia) &&
    typeof MediaRecorder !== "undefined";

  useEffect(() => () => stopTracks(), []);
  function stopTracks() {
    clearInterval(timer.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
  }
  async function start() {
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.current = media;
      const chunks = [];
      const rec = new MediaRecorder(media);
      recorder.current = rec;
      rec.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      rec.onstop = async () => {
        stopTracks();
        setRecording(false);
        try {
          const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
          const context = new AudioContext();
          const decoded = await context.decodeAudioData(
            await blob.arrayBuffer(),
          );
          await context.close();
          const wav = wavFromBuffer(decoded);
          const stamp = new Date()
            .toISOString()
            .slice(11, 19)
            .replace(/:/g, "-");
          onRecorded(
            new File([wav], `recording-${stamp}.wav`, { type: "audio/wav" }),
          );
        } catch (err) {
          onError?.(`The recording could not be read: ${err.message}`);
        }
      };
      rec.start();
      setSeconds(0);
      setRecording(true);
      const began = Date.now();
      timer.current = setInterval(() => {
        const s = (Date.now() - began) / 1000;
        setSeconds(s);
        if (s >= MOST_SECONDS) stop();
      }, 250);
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
  function stop() {
    clearInterval(timer.current);
    if (recorder.current?.state === "recording") recorder.current.stop();
  }
  if (!supported) return null;
  return (
    <button
      type="button"
      className={`record-button ${recording ? "recording" : ""}`}
      disabled={disabled}
      aria-pressed={recording}
      title={
        recording
          ? "Stop recording"
          : `Record from the microphone, up to ${MOST_SECONDS} seconds; the recording stays on this computer`
      }
      onClick={() => (recording ? stop() : start())}
    >
      {recording ? <Stop size={17} weight="fill" /> : <Microphone size={17} />}
      {recording ? `${seconds.toFixed(0)} s · stop` : "Record"}
    </button>
  );
}
