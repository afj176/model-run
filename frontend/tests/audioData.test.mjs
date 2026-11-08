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
