import test from "node:test";
import assert from "node:assert/strict";
import { buildReport, heatmap } from "../src/report.js";

const base = {
  schema_version: "1.0",
  model_class: "fixture.Model",
  torch_version: "0",
  model_name: "Fixture",
  limitations: ["Fixture limitation."],
};

test("heatmap falls back to inline SVG outside a browser", () => {
  const svg = heatmap(
    [
      [0, 1],
      [0.5, null],
    ],
    { width: 20, title: "t" },
  );
  assert.match(svg, /^<svg/);
  assert.equal((svg.match(/<rect/g) || []).length, 4);
  assert.equal(heatmap([], {}), "");
});

test("attention report covers head profile, ablation, gallery and lens-free output", () => {
  const trace = {
    ...base,
    kind: "attention_capture",
    layer: 3,
    head: 1,
    head_count: 2,
    layer_count: 2,
    values: [
      [1, 0, 0],
      [0.9, 0.1, 0],
      [0.8, 0.1, 0.1],
    ],
    heads: [
      [
        [1, 0, 0],
        [0.9, 0.1, 0],
        [0.8, 0.1, 0.1],
      ],
      [
        [1, 0, 0],
        [0.5, 0.5, 0],
        [0.1, 0.1, 0.8],
      ],
    ],
    tokens: [
      { id: 1, text: "The" },
      { id: 2, text: " cat" },
      { id: 3, text: " sat" },
    ],
    input: { token_count: 3 },
    attribution: {
      values: [1, 0, 0],
      last_position_attention: [0.6, 0.3, 0.1],
    },
    layers: [
      { index: 0, sink_mass: 0.5, entropy_bits: 1 },
      { index: 1, sink_mass: 0.4, entropy_bits: 1.2 },
    ],
    ablation: {
      method: "zero_ablation",
      layer: 3,
      head_count: 2,
      baseline: {
        top_tokens: [{ token_id: 9, text: " on", probability: 0.7 }],
        entropy_bits: 1,
      },
      heads: [
        {
          head: 0,
          top_tokens: [{ token_id: 9, text: " on", probability: 0.69 }],
          entropy_bits: 1,
          kl_bits: 0.01,
          baseline_top_probability: 0.69,
          top_token_changed: false,
        },
        {
          head: 1,
          top_tokens: [{ token_id: 4, text: " up", probability: 0.4 }],
          entropy_bits: 2,
          kl_bits: 1.2,
          baseline_top_probability: 0.3,
          top_token_changed: true,
        },
      ],
      limitations: ["Zero ablation."],
    },
  };
  const html = buildReport(trace, {
    generated: new Date("2026-11-04T10:00:00Z"),
  });
  assert.match(html, /^<!doctype html>/);
  assert.match(html, /Attention sink/);
  assert.match(html, /What the last token looked at/);
  assert.match(html, /answer changes/);
  assert.match(html, /H1 · flips/);
  assert.match(html, /All 2 heads of layer 3/);
  assert.match(html, /Attention entropy/);
  assert.match(html, /Fixture limitation\./);
  assert.match(html, /generated 2026-11-04 10:00 UTC/);
  assert.doesNotMatch(html, /<script/);
});

test("module report covers prediction, lens, activations and cross-attention", () => {
  const trace = {
    ...base,
    kind: "pytorch_module_capture",
    status: "completed",
    tokens: [{ id: 1, text: "Paris" }],
    output: {
      position: 0,
      entropy_bits: 1.5,
      top_tokens: [
        { token_id: 5, text: " is", probability: 0.6 },
        { token_id: 6, text: " was", probability: 0.2 },
      ],
    },
    logit_lens: {
      final_token_id: 5,
      layers: [
        {
          index: 0,
          label: "embeddings",
          top_tokens: [{ token_id: 6, text: " was", probability: 0.3 }],
          entropy_bits: 3,
          final_token_probability: 0.1,
          residual_norm: 1,
        },
        {
          index: 1,
          label: "layer 0",
          top_tokens: [{ token_id: 5, text: " is", probability: 0.6 }],
          entropy_bits: 1.5,
          final_token_probability: 0.6,
          residual_norm: 2,
        },
      ],
      limitations: ["Lens limitation."],
    },
    events: [
      {
        module: "model.layers.0",
        status: "captured",
        shape: [1, 1, 4],
        values: [...Array(500).fill(0.1), 30],
        denoiser_call: 0,
      },
      {
        module: "model.layers.0",
        status: "captured",
        shape: [1, 1, 4],
        values: [...Array(500).fill(0.2), 30],
        denoiser_call: 1,
      },
    ],
    cross_attention: {
      method: "measured_attention_probabilities",
      source: "fixture",
      selection: "double:0",
      grid_limit: 32,
      text_labels: ["a", " cat", "<|endoftext|>", "<|endoftext|>"],
      events: [
        {
          denoiser_call: 0,
          block: "double:0",
          batch: "batch 0",
          attention_kind: "joint",
          heads: 2,
          text_tokens: 4,
          image_tokens: 4,
          source_grid: [2, 2],
          grid: [2, 2],
          maps: [
            [0.1, 0.1, 0.1, 0.1],
            [0.5, 0.1, 0.1, 0.1],
            [0.01, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.01],
          ],
          text_mass: [0.62, 0.22, 0.22, 0.22],
          token_mass: [0.1, 0.2, 0.01, 0.01],
        },
      ],
      limitations: ["Cross limitation."],
    },
  };
  const html = buildReport(trace, { name: "Custom name" });
  assert.match(html, /Custom name/);
  assert.match(html, /What it predicted next/);
  assert.match(html, /first became the top guess at layer 0/);
  assert.match(html, /How the answer formed/);
  assert.match(html, /Where the prompt is read/);
  assert.match(html, /“␣cat” is read most at the top left/);
  assert.match(html, /Text share/);
  assert.match(html, /Activations/);
  assert.match(html, /Change between consecutive calls/);
  assert.match(html, /massive activations/);
  assert.match(html, /Cross limitation\./);
  assert.throws(() => buildReport(null), /No trace/);
});
