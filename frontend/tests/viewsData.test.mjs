import test from "node:test";
import assert from "node:assert/strict";
import { strongestKeys } from "../src/arcData.js";
import { riverLanes } from "../src/riverData.js";
import { wordStepGrid } from "../src/wordStepData.js";
import { headCensus } from "../src/censusData.js";

test("the strongest keys stop at 90% of the row or eight keys", () => {
  const keys = strongestKeys([0.5, 0.3, 0.15, 0.05]);
  assert.deepEqual(
    keys.map((k) => k.key),
    [0, 1, 2],
  );
  const flat = strongestKeys(new Array(40).fill(0.025));
  assert.equal(flat.length, 8);
  assert.deepEqual(strongestKeys(null), []);
  assert.deepEqual(strongestKeys([NaN, 0, null]), []);
});

test("river lanes hold every candidate that reaches a top list, the answer last", () => {
  const lens = {
    final_token_id: 7,
    layers: [
      { index: 0, top_tokens: [{ token_id: 1, text: "a", probability: 0.2 }] },
      {
        index: 1,
        top_tokens: [
          { token_id: 7, text: "b", probability: 0.5 },
          { token_id: 1, text: "a", probability: 0.1 },
        ],
      },
    ],
  };
  const lanes = riverLanes(lens, new Map([[7, "B"]]));
  assert.equal(lanes.length, 2);
  assert.equal(lanes[0].token_id, 7);
  assert.equal(lanes[0].final, true);
  assert.equal(lanes[0].text, "B");
  assert.deepEqual(lanes[0].p, [0, 0.5]);
  assert.deepEqual(lanes[1].p, [0.2, 0.1]);
  assert.deepEqual(riverLanes(null), []);
});

test("the word-by-call grid shares each call's text attention and skips padding runs", () => {
  const labels = ["a", "fox", "<pad>", "<pad>"];
  const events = [
    { denoiser_call: 0, token_mass: [1, 3, 0.5, 0.5] },
    { denoiser_call: 2, token_mass: [1, 1, 0.5, 0.5] },
  ];
  const grid = wordStepGrid(events, labels);
  assert.deepEqual(grid.rows, [0, 1, 2]);
  assert.deepEqual(grid.calls, [0, 2]);
  assert.deepEqual(
    grid.shares[0].map((v) => Number(v.toFixed(3))),
    [0.222, 0.667, 0.111],
  );
  assert.equal(grid.cells[0][1], 1);
  assert.equal(grid.rest, null);
  assert.equal(wordStepGrid([], labels), null);
  // A long encoder prompt keeps only the words that matter, in prompt order, and folds the rest.
  const long = wordStepGrid(
    [{ denoiser_call: 0, token_mass: [0.1, 5, 0.1, 0.1, 3, 0.1] }],
    null,
    2,
  );
  assert.deepEqual(long.rows, [1, 4]);
  assert.equal(long.total, 6);
  assert.equal(long.rest.count, 4);
  assert.ok(Math.abs(long.rest.shares[0] - 0.4 / 8.4) < 1e-9);
});

test("a head census profiles every kept head and falls back to the layer mean", () => {
  const sink = [
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
  ];
  const self = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const trace = {
    kind: "attention_capture",
    layer: 1,
    head: 0,
    head_count: 2,
    layer_count: 3,
    values: sink,
    heads: [sink, self],
    layer_stacks: [{ index: 0, mean: self }],
    ablation: {
      layer: 1,
      heads: [
        { head: 0, kl_bits: 1.2 },
        { head: 1, kl_bits: 0.01 },
      ],
    },
  };
  const census = headCensus(trace);
  assert.deepEqual(
    census.layers.map((l) => l.index),
    [0, 1],
  );
  assert.equal(census.layers[0].heads, null);
  assert.equal(census.layers[0].mean.role, "self");
  assert.equal(census.layers[1].heads.length, 2);
  assert.equal(census.layers[1].heads[0].role, "sink");
  assert.equal(census.layers[1].heads[0].effect, 1.2);
  assert.equal(census.strongest[0].head, 0);
  assert.equal(headCensus({ kind: "generation_capture" }), null);
});
