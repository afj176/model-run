import test from "node:test";
import assert from "node:assert/strict";
import { packValues, unpackValues } from "../src/packedValues.js";
import { sliceEvent } from "../src/matrixData.js";

test("measured values pack and come back; whole numbers stay as they are", () => {
  const values = Array.from({ length: 100 }, (_, i) =>
    Number((0.1 * i - 3.3).toFixed(4)),
  );
  values[7] = null;
  const value = {
    values,
    indices: Array.from({ length: 100 }, (_, i) => i),
    short: [0.5, 1.5],
  };
  const packed = packValues(value);
  assert.equal(packed.values.__packed__, "float32");
  assert.deepEqual(packed.indices, value.indices);
  assert.deepEqual(packed.short, [0.5, 1.5]);
  assert.deepEqual(unpackValues(packed), value);
});

test("a damaged pack is refused", () => {
  assert.throws(() =>
    unpackValues({ __packed__: "float32", n: 3, data: "AAAA" }),
  );
});

test("a token grid reads as every token by feature, named", () => {
  const event = {
    module: "model.layers.3",
    shape: [1, 3, 2560],
    slices: {
      layout: "tokens",
      axes: [],
      rows: 3,
      cols: 2,
      source_rows: 3,
      source_cols: 2560,
      token_indices: [0, 1, 2],
      feature_indices: [0, 1280],
      row_labels: ["<bos>", "Walk", "park"],
      values: [1, 2, 3, 4, 5, 6],
      token_rms: [1, 2, 3],
      aggregate: [],
    },
  };
  const grid = sliceEvent(event);
  assert.deepEqual(grid.shape, [3, 2]);
  assert.deepEqual(grid.axes, ["token", "feature"]);
  assert.deepEqual(grid.col_labels, ["#0", "#1280"]);
  assert.deepEqual(grid.row_labels, ["<bos>", "Walk", "park"]);
  assert.match(grid.slice_note, /Every token by 2 of its 2,560 features/);
});
