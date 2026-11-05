import test from "node:test";
import assert from "node:assert/strict";
import { extractMatrix, resolveSlice, sliceEvent } from "../src/matrixData.js";

test("attention cells preserve selected head values and axes without inventing masks", () => {
  const trace = {
    kind: "attention_capture",
    layer: 12,
    head: 7,
    axes: ["query_position", "key_position"],
    shape: [2, 2],
    values: [
      [1, 0],
      [0.18, 0.82],
    ],
  };
  const result = extractMatrix(trace, null);
  assert.deepEqual(result.matrix, trace.values);
  assert.deepEqual(result.axes, trace.axes);
  assert.deepEqual(result.states, [
    ["captured", "captured"],
    ["captured", "captured"],
  ]);
  assert.equal(result.source, "Layer 12 · head 7");
});

test("prefix windows preserve row-major coordinates and distinguish missing from nonfinite", () => {
  const event = {
    module: "layer",
    selection: "flattened_prefix",
    shape: [1, 2, 4],
    values: [2, null, 3, 4, 5],
  };
  const result = extractMatrix({}, event, {
    rowOffset: 1,
    colOffset: 0,
    maxRows: 1,
    maxCols: 3,
  });
  assert.deepEqual(result.matrix, [[5, null, null]]);
  assert.deepEqual(result.states, [["captured", "uncaptured", "uncaptured"]]);
  const first = extractMatrix({}, event, { maxRows: 1, maxCols: 3 });
  assert.deepEqual(first.states, [["captured", "nonfinite", "captured"]]);
  assert.deepEqual(first.fixedIndices, [0]);
  assert.deepEqual(first.axes, ["dim 1", "dim 2"]);
});

test("leading indices and large dimension windowing use actual coordinates", () => {
  const event = {
    selection: "flattened_prefix",
    shape: [2, 2, 2],
    values: [1, 2, 3, 4, 5, 6, 7, 8],
  };
  assert.deepEqual(extractMatrix({}, event, { fixedIndices: [1] }).matrix, [
    [5, 6],
    [7, 8],
  ]);
  const large = extractMatrix(
    {},
    { selection: "flattened_prefix", shape: [1, 256], values: [9, 8] },
    { colOffset: 127 },
  );
  assert.equal(large.cols, 128);
  assert.equal(large.states[0][0], "uncaptured");
});

test("reject inconsistent values and unsafe shapes", () => {
  assert.throws(
    () =>
      extractMatrix(
        { kind: "attention_capture", shape: [2, 2], values: [[0], [1]] },
        null,
      ),
    /shape/,
  );
  assert.throws(
    () =>
      extractMatrix(
        { kind: "attention_capture", shape: [1, 1], values: [[-0.5]] },
        null,
      ),
    /between 0 and 1/,
  );
  assert.throws(
    () =>
      extractMatrix(
        {},
        { selection: "flattened_prefix", shape: [1], values: [1, 2] },
      ),
    /prefix/,
  );
  assert.throws(
    () =>
      extractMatrix(
        {},
        {
          selection: "flattened_prefix",
          shape: [Number.MAX_SAFE_INTEGER, 2],
          values: [],
        },
      ),
    /safe/,
  );
  assert.throws(
    () =>
      extractMatrix(
        {},
        { selection: "flattened_prefix", shape: [2, 2], values: [] },
        { colOffset: 3 },
      ),
    /window/,
  );
});

const sliced = () => ({
  module: "denoiser",
  selection: "flattened_prefix",
  shape: [1, 2, 2, 1, 2],
  total_elements: 8,
  values: [1, 2],
  truncated: true,
  slices: {
    layout: "latent",
    axes: [
      {
        dim: 1,
        name: "channel",
        size: 4,
        indices: [0, 2],
        mean_abs: [1, 2],
        max_abs: [1, 2],
      },
      {
        dim: 2,
        name: "frame",
        size: 2,
        indices: [0, 1],
        mean_abs: [1, 2],
        max_abs: [1, 2],
      },
    ],
    row_axis: "height",
    col_axis: "width",
    rows: 1,
    cols: 2,
    source_rows: 2,
    source_cols: 4,
    pooled: true,
    assumption: null,
    // channel 0 frame 0, channel 0 frame 1, channel 2 frame 0, channel 2 frame 1
    values: [1, 2, 3, 4, 5, 6, 7, 8],
    aggregate_axis: 0,
    aggregate: [10, 20, 30, 40],
  },
});

test("resolveSlice defaults to the all-values map and snaps to recorded indices", () => {
  const { slices } = sliced();
  assert.deepEqual(resolveSlice(slices), [null, 0]);
  assert.deepEqual(resolveSlice(slices, { channel: 3, frame: 1 }), [1, 1]);
  assert.deepEqual(resolveSlice(slices, { channel: "all", frame: 9 }), [
    null,
    1,
  ]);
});

test("sliceEvent exposes one slice as a plain 2-D tensor with named axes", () => {
  const one = sliceEvent(sliced(), { channel: 2, frame: 1 });
  assert.deepEqual(one.values, [7, 8]);
  assert.deepEqual(one.shape, [1, 2]);
  assert.equal(one.slice_label, "channel 2 · frame 1");
  assert.match(one.slice_note, /Averaged from 2 × 4 to 1 × 2/);
  assert.match(one.slice_note, /2 of 4 channels recorded/);
  const matrix = extractMatrix({}, one);
  assert.deepEqual(matrix.matrix, [[7, 8]]);
  assert.deepEqual(matrix.axes, ["height", "width"]);
  assert.deepEqual(matrix.states, [["captured", "captured"]]);
  assert.equal(matrix.capturedCount, 2);
  const all = sliceEvent(sliced(), { frame: 1 });
  assert.deepEqual(all.values, [30, 40]);
  assert.equal(all.slice_label, "all channels · frame 1");
  assert.match(all.slice_note, /Mean magnitude across every channel/);
});

test("sliceEvent leaves events without usable slices alone", () => {
  const plain = { module: "m", values: [1] };
  assert.equal(sliceEvent(plain), plain);
  assert.equal(sliceEvent(null), null);
  const skipped = { module: "m", slices: { unavailable: "budget" } };
  assert.equal(sliceEvent(skipped), skipped);
});

test("sliceEvent shows a whole sequence as one feature-by-time map", () => {
  const event = {
    module: "encoder",
    shape: [1, 1500, 384],
    values: [1],
    slices: {
      layout: "sequence",
      axes: [],
      row_axis: "feature",
      col_axis: "time",
      rows: 2,
      cols: 3,
      source_rows: 384,
      source_cols: 50,
      pooled: true,
      assumption: "Only the audio is shown.",
      values: [1, 2, 3, 4, 5, 6],
      aggregate_axis: null,
      aggregate: [],
    },
  };
  const seen = sliceEvent(event);
  assert.deepEqual(seen.values, [1, 2, 3, 4, 5, 6]);
  assert.equal(seen.slice_label, "feature by time");
  assert.equal(seen.magnitude, false);
  assert.equal(seen.grid_cols, 3);
  assert.match(seen.slice_note, /Averaged from 384 × 50 to 2 × 3/);
  assert.match(seen.slice_note, /Only the audio is shown/);
  assert.deepEqual(extractMatrix({}, seen).axes, ["feature", "time"]);
});

test("a batched sequence is read one batch entry at a time", () => {
  const event = {
    module: "audio",
    shape: [2, 2, 8],
    values: [],
    slices: {
      layout: "sequence",
      axes: [
        {
          dim: 0,
          name: "batch",
          size: 2,
          indices: [0, 1],
          mean_abs: [1, 2],
          max_abs: [1, 2],
        },
      ],
      row_axis: "feature",
      col_axis: "time",
      rows: 1,
      cols: 2,
      source_rows: 8,
      source_cols: 2,
      pooled: true,
      assumption: null,
      values: [1, 2, 3, 4],
      aggregate_axis: null,
      aggregate: [],
    },
  };
  assert.deepEqual(sliceEvent(event).values, [1, 2]);
  const second = sliceEvent(event, { batch: 1 });
  assert.deepEqual(second.values, [3, 4]);
  assert.equal(second.slice_label, "feature by time · batch 1");
});
