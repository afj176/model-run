import test from "node:test";
import assert from "node:assert/strict";
import {
  barcodeModel,
  isOutlier,
  outlierChannels,
  scaled,
  strongest,
  poolTokenGrid,
} from "../src/barcodeData.js";

// Three tokens by 200 features: small values, and one channel (#150) far above every token's size.
const rows = 3,
  cols = 200;
const values = Array.from({ length: rows * cols }, (_, i) =>
  i % cols === 150 ? 90 + (i / cols) * 5 : ((i * 7) % 11) / 5 - 1,
);
const event = {
  shape: [rows, cols],
  values,
  token_rms: [1.1, 1.2, 1.0],
  row_labels: ["<bos>", "Walk", "park"],
  feature_indices: Array.from({ length: cols }, (_, c) => c * 4),
};

test("a token grid becomes a barcode model; anything else does not", () => {
  const model = barcodeModel(event);
  assert.equal(model.rows, 3);
  assert.equal(model.cols, 200);
  assert.deepEqual(model.labels, ["<bos>", "Walk", "park"]);
  assert.equal(model.features[150], 600);
  assert.equal(barcodeModel({ shape: [2, 2], values: [1, 2, 3, 4] }), null);
  assert.equal(barcodeModel({ ...event, values: values.slice(1) }), null);
});

test("the robust scale saturates the outlier channel and keeps the rest visible", () => {
  const model = barcodeModel(event);
  assert.ok(model.p99 < 5, `p99 ${model.p99}`);
  assert.equal(scaled(95, "robust", model), 1);
  assert.ok(Math.abs(scaled(0.8, "robust", model)) > 0.3);
  assert.ok(Math.abs(scaled(0.8, "linear", model)) < 0.02);
  assert.ok(isOutlier(95, model));
  assert.equal(scaled(null, "robust", model), null);
});

test("a pixel column shows its strongest feature", () => {
  const model = barcodeModel(event);
  assert.equal(strongest(model, 1, 140, 160), 150);
  assert.equal(strongest(model, 0, 300, 400), -1);
});

test("outlier channels name the feature, how many tokens and the peak", () => {
  const [channel] = outlierChannels(barcodeModel(event));
  assert.equal(channel.feature, 600);
  assert.equal(channel.tokens, 3);
  assert.equal(channel.row, 2);
  assert.ok(channel.peak > 99);
});

test("Run View pools each token's features and maps every value to the bar that holds it", () => {
  for (const [rows, features] of [
    [12, 2560],
    [3, 7],
    [128, 1024],
    [5, 130],
  ]) {
    const values = Array.from({ length: rows * features }, (_, i) =>
      Math.sin(i * 1.7),
    );
    values[features + 3] = 50;
    const pool = poolTokenGrid(values, rows, features);
    assert.equal(pool.values.length, rows * pool.cols);
    assert.ok(pool.values.length <= 3072 || pool.cols === 1);
    for (let i = 0; i < rows * features; i += 7) {
      const bar = pool.toBar(i);
      assert.equal(Math.floor(bar / pool.cols), Math.floor(i / features));
      // The bar's source lies in the same token and is at least as strong.
      const source = pool.source[bar];
      assert.equal(Math.floor(source / features), Math.floor(i / features));
      assert.ok(Math.abs(values[source]) >= Math.abs(values[i]));
    }
    assert.equal(pool.source[pool.toBar(features + 3)], features + 3);
  }
});
