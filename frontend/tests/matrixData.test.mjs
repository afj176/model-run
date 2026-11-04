import test from "node:test";
import assert from "node:assert/strict";
import { extractMatrix } from "../src/matrixData.js";

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
