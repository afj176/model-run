import test from "node:test";
import assert from "node:assert/strict";
import {
  attentionArcs,
  denoisingStory,
  featureSheet,
  featureStrip,
  fitPoints,
  headFanOut,
  samplingWhatIf,
  unitTokens,
} from "../src/viewData.js";

function logitsView(logits, grid = [0.5, 1, 2]) {
  const logTotal = (t) =>
    Math.log(logits.reduce((s, l) => s + Math.exp(l / t), 0));
  return {
    top: logits.map((logit, token_id) => ({ token_id, logit })),
    temperatures: grid.map((temperature) => ({
      temperature,
      log_total: logTotal(temperature),
    })),
  };
}

test("temperature reshapes the measured distribution exactly", () => {
  const view = logitsView([3, 2, 1]);
  const at = (temperature) =>
    samplingWhatIf(view, new Map(), { temperature, topK: 3, topP: 1 });
  const one = at(1);
  const expected = Math.exp(3) / (Math.exp(3) + Math.exp(2) + Math.exp(1));
  assert.ok(Math.abs(one.rows[0].probability - expected) < 1e-9);
  assert.ok(Math.abs(one.seen - 1) < 1e-9);
  assert.ok(at(0.5).rows[0].probability > one.rows[0].probability);
  assert.ok(at(2).rows[0].probability < one.rows[0].probability);
  assert.equal(at(0.6).temperature, 0.5); // snaps to a value that was measured
});

test("top-K and top-P cut tokens and share out what is left", () => {
  const view = logitsView([3, 2, 1, 0]);
  const k = samplingWhatIf(view, new Map(), {
    temperature: 1,
    topK: 2,
    topP: 1,
  });
  assert.equal(k.kept, 2);
  assert.ok(Math.abs(k.rows[0].filtered + k.rows[1].filtered - 1) < 1e-9);
  assert.equal(k.rows[2].filtered, 0);
  const p = samplingWhatIf(view, new Map(), {
    temperature: 1,
    topK: 4,
    topP: 0.5,
  });
  assert.equal(p.kept, 1);
  assert.equal(p.rows[0].filtered, 1);
  assert.equal(p.entropy, 0);
  assert.equal(samplingWhatIf({ top: [] }, new Map(), {}), null);
});

test("top-P says so when the recorded tokens cannot reach it", () => {
  const view = logitsView([1, 1, 1, 1]);
  view.top = view.top.slice(0, 2);
  const result = samplingWhatIf(view, new Map(), {
    temperature: 1,
    topK: 200,
    topP: 0.9,
  });
  assert.equal(result.reachedTopP, false);
  assert.ok(Math.abs(result.seen - 0.5) < 1e-9);
});

test("links leave out a token looking at itself", () => {
  const arcs = attentionArcs([
    [0.9, 0.1],
    [0.7, 0.3],
  ]);
  assert.deepEqual(
    arcs.map((a) => [a.from, a.to]),
    [
      [1, 0],
      [0, 1],
    ],
  );
  assert.deepEqual(attentionArcs(undefined), []);
  assert.equal(
    attentionArcs(
      [
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
      ],
      2,
    ).length,
    2,
  );
});

test("each head reports where it looks from the chosen token", () => {
  const heads = headFanOut(
    [
      [
        [1, 0],
        [0.2, 0.8],
      ],
      [
        [1, 0],
        [0.9, 0.1],
      ],
    ],
    1,
  );
  assert.deepEqual(
    heads.map((h) => [h.head, h.peak, h.weight]),
    [
      [0, 1, 0.8],
      [1, 0, 0.9],
    ],
  );
});

test("the denoising story only says what was recorded", () => {
  const section = {
    total_steps: 3,
    steps: [
      { index: 0, sigma: 14.6, estimate_change: null, guidance_cosine: 0.9 },
      { index: 1, sigma: 5, estimate_change: 0.9, guidance_cosine: 0.9999 },
      { index: 2, sigma: 0.5, estimate_change: 0.1 },
    ],
  };
  assert.match(denoisingStory(section, 0), /Step 1 of 3, at noise level 14.60/);
  assert.match(denoisingStory(section, 0), /moved most at step 2/);
  assert.match(denoisingStory(section, 0), /90.0% alike/);
  assert.match(denoisingStory(section, 1), /more here than at any other/);
  assert.match(denoisingStory(section, 1), /almost the same/);
  assert.match(denoisingStory(section, 2), /Guidance was off/);
  assert.equal(denoisingStory(section, 9), "");
});

test("a unit's tokens are ranked by how far they move it, either way", () => {
  const layer = {
    units: [7, 9],
    unit_values: [
      [1, 0],
      [-3, 0],
      [2, 0],
    ],
  };
  const tokens = [{ text: "a" }, { text: "b" }];
  const ranked = unitTokens(layer, 7, tokens);
  assert.deepEqual(
    ranked.map((r) => r.text),
    ["b", "#2", "a"],
  );
  assert.deepEqual(unitTokens(layer, 8, tokens), []);
});

test("fitting points keeps their proportions", () => {
  const fitted = fitPoints(
    [
      [
        [0, 0],
        [10, 1],
      ],
    ],
    null,
    100,
    0,
  );
  const [a, b] = fitted[0];
  const ratio = Math.abs(a[1] - b[1]) / Math.abs(a[0] - b[0]);
  assert.ok(Math.abs(ratio - 0.1) < 1e-9);
  assert.deepEqual(fitPoints([]), []);
  // A token left out of the fit must not stretch the picture.
  const outlier = fitPoints(
    [
      [
        [0, 0],
        [10, 0],
        [9000, 0],
      ],
    ],
    [{}, {}, { special: true }],
    100,
    0,
  );
  assert.equal(outlier[0][1][0], 100);
});

test("a feature sheet reads the positive prompt's channels from a guidance batch", () => {
  const event = {
    module: "m",
    shape: [2, 4, 1, 2],
    slices: {
      layout: "features",
      rows: 1,
      cols: 2,
      axes: [
        { dim: 0, name: "batch", size: 2, indices: [0, 1], conditional: 1 },
        { dim: 1, name: "channel", size: 4, indices: [0, 3] },
      ],
      values: [1, 1, 2, 2, 3, 3, 4, 4],
      aggregate: [9, 9, 8, 8],
    },
  };
  const sheet = featureSheet(event);
  assert.deepEqual(
    sheet.channels.map((c) => [c.index, c.values]),
    [
      [0, [3, 3]],
      [3, [4, 4]],
    ],
  );
  assert.equal(sheet.size, 4);
  assert.equal(featureSheet({ slices: { layout: "axes" } }), null);
  assert.deepEqual(
    featureStrip([event, event]).map((s) => s.module),
    ["m"],
  );
  assert.deepEqual(featureStrip([event])[0].values, [8, 8]);
});
