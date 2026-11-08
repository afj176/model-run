import test from "node:test";
import assert from "node:assert/strict";
import { wavFromBuffer } from "../src/audioData.js";

/** A decoded buffer with two channels, the way the Web Audio API hands one over. */
const buffer = {
  numberOfChannels: 2,
  length: 4,
  sampleRate: 16000,
  getChannelData: (c) =>
    Float32Array.from(c === 0 ? [0, 0.5, -0.5, 2] : [0, 0.5, -0.5, 2]),
};

test("a decoded buffer becomes a 16-bit mono WAV with a correct header", async () => {
  const blob = wavFromBuffer(buffer);
  assert.equal(blob.type, "audio/wav");
  const bytes = new DataView(await blob.arrayBuffer());
  assert.equal(
    String.fromCharCode(...new Uint8Array(bytes.buffer, 0, 4)),
    "RIFF",
  );
  assert.equal(
    String.fromCharCode(...new Uint8Array(bytes.buffer, 8, 4)),
    "WAVE",
  );
  assert.equal(bytes.getUint16(22, true), 1); // mono
  assert.equal(bytes.getUint32(24, true), 16000);
  assert.equal(bytes.getUint16(34, true), 16);
  assert.equal(bytes.getUint32(40, true), 8);
  assert.equal(bytes.getInt16(44, true), 0);
  assert.equal(bytes.getInt16(46, true), Math.trunc(0.5 * 0x7fff));
  assert.equal(bytes.getInt16(48, true), -0.5 * 0x8000);
  assert.equal(bytes.getInt16(50, true), 0x7fff); // clipped, not wrapped
});

import { formatTime, peaksFromBuffer, windowOf } from "../src/audioData.js";

test("times read as minutes and seconds, with tenths for short clips", () => {
  assert.equal(formatTime(3.54, true), "0:03.5");
  assert.equal(formatTime(3.54), "0:03");
  assert.equal(formatTime(101.2), "1:41");
  assert.equal(formatTime(NaN), "–");
});

test("the window a model hears is clamped to the file and says what it leaves out", () => {
  assert.deepEqual(windowOf(0, 3.54), {
    start: 0,
    end: 3.54,
    used: 3.54,
    whole: true,
    leftOut: [],
  });
  const w = windowOf(38, 101.2);
  assert.equal(w.end, 68);
  assert.equal(w.used, 30);
  assert.equal(w.whole, false);
  assert.deepEqual(w.leftOut, [
    [0, 38],
    [68, 101.2],
  ]);
  assert.equal(windowOf(500, 101.2).start < 101.2, true);
  assert.equal(windowOf(90, 101.2).used.toFixed(1), "11.2");
});

test("peaks take the largest magnitude of any channel in each column", () => {
  const peaks = peaksFromBuffer(
    {
      length: 8,
      numberOfChannels: 2,
      getChannelData: (c) =>
        Float32Array.from(
          c ? [0, 0, 0, 0, 0.9, 0, 0, 0] : [0.1, -0.5, 0, 0, 0, 0, 0, 0.2],
        ),
    },
    4,
  );
  assert.deepEqual(
    [...peaks].map((v) => Number(v.toFixed(2))),
    [0.5, 0, 0.9, 0.2],
  );
});
