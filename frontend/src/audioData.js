// Audio the browser recorded, as a WAV file the server reads like any other. Pure functions;
// AudioRecorder.jsx uses them.

/** 16-bit mono WAV bytes from decoded audio, so the server reads it like any file. */
export function wavFromBuffer(buffer) {
  const channels = buffer.numberOfChannels;
  const length = buffer.length;
  const mono = new Float32Array(length);
  for (let c = 0; c < channels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < length; i++) mono[i] += data[i] / channels;
  }
  const bytes = new ArrayBuffer(44 + length * 2);
  const view = new DataView(bytes);
  const text = (at, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(at + i, s.charCodeAt(i));
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, length * 2, true);
  for (let i = 0; i < length; i++) {
    const v = Math.max(-1, Math.min(1, mono[i]));
    view.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
  }
  return new Blob([bytes], { type: "audio/wav" });
}

/** Seconds as m:ss, with tenths under a minute when asked ("0:03.5", "1:41"). */
export function formatTime(seconds, tenths = false) {
  if (!Number.isFinite(seconds) || seconds < 0) return "–";
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  if (tenths && seconds < 60) return `${m}:${s.toFixed(1).padStart(4, "0")}`;
  return `${m}:${String(Math.floor(s)).padStart(2, "0")}`;
}

/** The 30-second window a model hears from `start`: where it ends, how much it covers, what it leaves out. */
export function windowOf(start, duration, most = 30) {
  const n = Number.isFinite(duration) ? Math.max(0, duration) : 0;
  const from = Math.max(
    0,
    Math.min(Number.isFinite(start) ? start : 0, Math.max(0, n - 0.001)),
  );
  const to = Math.min(n, from + most);
  const whole = from === 0 && to >= n - 1e-6;
  const leftOut = [];
  if (from > 0) leftOut.push([0, from]);
  if (to < n - 1e-6) leftOut.push([to, n]);
  return { start: from, end: to, used: to - from, whole, leftOut };
}

/** One peak per column from a decoded buffer: the largest magnitude over all channels in that slice. */
export function peaksFromBuffer(buffer, columns = 240) {
  const length = buffer.length;
  const channels = buffer.numberOfChannels;
  const cols = Math.max(1, Math.min(columns, length));
  const per = length / cols;
  const peaks = new Float32Array(cols);
  for (let c = 0; c < channels; c++) {
    const data = buffer.getChannelData(c);
    for (let k = 0; k < cols; k++) {
      const a = Math.floor(k * per);
      const b = Math.max(a + 1, Math.floor((k + 1) * per));
      let top = 0;
      for (let i = a; i < b && i < length; i++) {
        const v = Math.abs(data[i]);
        if (v > top) top = v;
      }
      if (top > peaks[k]) peaks[k] = top;
    }
  }
  return peaks;
}
