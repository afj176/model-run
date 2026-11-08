import test from "node:test";
import assert from "node:assert/strict";
import {
  layerSummary,
  layerTokenGrid,
  scaledGrid,
} from "../src/weatherData.js";

const events = [
  { module: "model.layers.1", slices: { token_rms: [1, 2, 3] } },
  { module: "model.layers.0", slices: { token_rms: [10, 1, 1] } },
  { module: "model.embed_tokens", slices: { token_rms: [1, 1, 1] } },
  {
    module: "model.layers.0.self_attn",
    part: "attention",
    part_of: "model.layers.0",
    slices: { token_rms: [0.5, 0.2, 0.2] },
  },
  {
    module: "model.layers.0.mlp",
    part: "mlp",
    part_of: "model.layers.0",
    slices: { token_rms: [9, 0.4, NaN] },
  },
  { module: "model.layers.2", slices: { unavailable: "budget" } },
];

test("events become one row per layer with the parts beside the output", () => {
  const grid = layerTokenGrid(events);
  assert.deepEqual(
    grid.layers.map((l) => l.layer),
    [0, 1],
  );
  assert.deepEqual(grid.layers[0].output, [10, 1, 1]);
  assert.deepEqual(grid.layers[0].mlp, [9, 0.4, null]);
  assert.equal(grid.layers[1].attention, null);
  assert.equal(grid.tokens, 3);
  assert.deepEqual(
    grid.measures.map((m) => m.key),
    ["output", "attention", "mlp"],
  );
});

test("the scaled grid tops out at the 98th percentile and keeps gaps", () => {
  const grid = layerTokenGrid(events);
  const { rows, scale } = scaledGrid(grid, "output");
  assert.equal(rows.length, 2);
  assert.equal(rows[0][0], 1);
  assert.ok(scale >= 3);
  assert.equal(scaledGrid(grid, "mlp").rows[0][2], null);
  assert.equal(scaledGrid(grid, "mlp").rows[1][0], null);
});

test("layer summaries give the mean, the peak and how far it stands from the median", () => {
  const grid = layerTokenGrid(events);
  const [l0, l1] = layerSummary(grid, "output");
  assert.equal(l0.mean, 4);
  assert.equal(l0.peak, 10);
  assert.equal(l0.spread, 10);
  assert.equal(l0.peakToken, 0);
  assert.equal(l1.median, 2);
  assert.equal(layerSummary(grid, "attention")[1].mean, null);
});
