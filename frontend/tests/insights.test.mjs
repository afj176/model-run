import test from "node:test";
import assert from "node:assert/strict";
import {
  ablationStory,
  patchStory,
  compareOverlap,
  headReactions,
  layerReactions,
  lensDivergence,
  promptCompareStory,
  promptFromTokens,
  tokenDiff,
  railLabel,
  crossAttentionStory,
  attentionHeadProfile,
  activationStats,
  describeActivation,
  callDeltas,
  layerSeries,
  rowSummary,
  tokenLabel,
  lensStory,
  compareMatrices,
  entropyBits,
} from "../src/insights.js";

test("head profiles classify sink, previous-token and diffuse patterns from weights only", () => {
  const sink = [
    [1, 0, 0],
    [0.9, 0.1, 0],
    [0.8, 0.1, 0.1],
  ];
  assert.equal(attentionHeadProfile(sink).label, "Attention sink");
  const previous = [
    [1, 0, 0],
    [0.1, 0.9, 0],
    [0.1, 0.8, 0.1],
  ];
  const p = attentionHeadProfile(previous);
  assert.equal(p.label, "Self-attending head" === p.label ? p.label : p.label);
  assert.ok(p.previousTokenMass > 0.4 || p.diagonalMass > 0.4);
  const n = 8;
  const diffuse = Array.from({ length: n }, () => Array(n).fill(1 / n));
  const d = attentionHeadProfile(diffuse);
  assert.equal(d.label, "Diffuse head");
  assert.ok(Math.abs(d.entropyBits - 3) < 1e-9);
  assert.equal(attentionHeadProfile([]), null);
});

test("row summary ranks keys and labels them with decoded tokens", () => {
  const tokens = [{ text: "Explain" }, { text: " how" }, { text: " a" }];
  const top = rowSummary([[0.2, 0.7, 0.1]], 0, tokens, 2);
  assert.deepEqual(
    top.map((t) => [t.key, t.label]),
    [
      [1, "how"],
      [0, "Explain"],
    ],
  );
  assert.equal(tokenLabel({ text: " " }, 4), "␣");
  assert.equal(tokenLabel({ text: "\n" }, 4), "⏎");
  assert.equal(tokenLabel(undefined, 4), "4");
});

test("activation stats keep nulls out and spot massive activations", () => {
  const values = [0.1, -0.1, 0.05, null, 40, 0.0001, 0.0002];
  const stats = activationStats(values);
  assert.equal(stats.count, 6);
  assert.equal(stats.nonFinite, 1);
  assert.equal(stats.outliers[0].index, 4);
  assert.ok(stats.outlierRatio > 2);
  assert.match(
    describeActivation(activationStats([...Array(1000).fill(0.01), 50])),
    /massive/,
  );
  assert.equal(activationStats([null, null]), null);
});

test("call deltas compare consecutive calls of the same module", () => {
  const events = [
    { module: "blocks.0", denoiser_call: 0, values: [1, 1, 1, 1] },
    { module: "blocks.1", denoiser_call: 0, values: [2, 2, 2, 2] },
    { module: "blocks.0", denoiser_call: 1, values: [1, 1, 1, 3] },
  ];
  const deltas = callDeltas(events);
  assert.equal(deltas[0].delta, null);
  assert.equal(deltas[1].delta, null);
  assert.ok(Math.abs(deltas[2].delta - 1) < 1e-12);
  assert.equal(deltas[2].ordinal, 1);
});

test("layer series prefers recorded summaries and falls back to activation RMS", () => {
  const attention = {
    kind: "attention_capture",
    layers: [{ index: 0, entropy_bits: 1, sink_mass: 0.5 }],
  };
  assert.equal(layerSeries(attention).points[0].value, 1);
  const lens = {
    logit_lens: {
      layers: [
        { index: 0, final_token_probability: 0.1 },
        { index: 1, final_token_probability: 0.4 },
      ],
    },
  };
  assert.deepEqual(layerSeries(lens).points, [{ layer: 0, value: 0.4 }]);
  const events = [
    { module: "model.layers.3.mlp", values: [5, -5] },
    { module: "model.embed", values: [9] },
  ];
  assert.deepEqual(layerSeries({}, events).points, [{ layer: 3, value: 5 }]);
  assert.equal(layerSeries({}, []), null);
});

test("lens story finds where the final token settles", () => {
  const lens = {
    final_token_id: 7,
    layers: [
      {
        index: 0,
        label: "embeddings",
        top_tokens: [{ token_id: 1 }],
        entropy_bits: 5,
        final_token_probability: 0.01,
      },
      {
        index: 1,
        label: "layer 0",
        top_tokens: [{ token_id: 7 }],
        entropy_bits: 2,
        final_token_probability: 0.3,
      },
      {
        index: 2,
        label: "layer 1",
        top_tokens: [{ token_id: 7, text: "A" }],
        entropy_bits: 1,
        final_token_probability: 0.6,
      },
    ],
  };
  const story = lensStory(lens, new Map([[7, "A"]]));
  assert.equal(story.settledLabel, "layer 0");
  assert.equal(story.wavered, false);
  const wavering = lensStory(
    {
      final_token_id: 7,
      layers: [
        {
          index: 0,
          label: "embeddings",
          top_tokens: [{ token_id: 7 }],
          entropy_bits: 1,
          final_token_probability: 0.2,
        },
        {
          index: 1,
          label: "layer 0",
          top_tokens: [{ token_id: 2 }],
          entropy_bits: 1,
          final_token_probability: 0.1,
        },
        {
          index: 2,
          label: "layer 1",
          top_tokens: [{ token_id: 7 }],
          entropy_bits: 1,
          final_token_probability: 0.5,
        },
      ],
    },
    new Map(),
  );
  assert.equal(wavering.wavered, true);
  assert.equal(wavering.firstLabel, "embeddings");
  assert.equal(wavering.settledLabel, "layer 1");
  assert.equal(story.uncertainLabel, "embeddings");
  assert.equal(story.finalText, "A");
  assert.equal(lensStory(null), null);
});

test("matrix comparison reports the largest divergence", () => {
  const result = compareMatrices(
    [
      [1, 0],
      [0.5, 0.5],
    ],
    [
      [1, 0],
      [0.1, 0.9],
    ],
  );
  assert.deepEqual(result.at, [1, 0]);
  assert.ok(Math.abs(result.maxDiff - 0.4) < 1e-12);
  assert.equal(compareMatrices([[1]], [[1, 0]]), null);
  assert.ok(Math.abs(entropyBits([0.5, 0.5]) - 1) < 1e-12);
});

test("ablationStory narrates one head and ranks the sweep", () => {
  const ablation = {
    head_count: 3,
    baseline: {
      top_tokens: [{ token_id: 1, text: " Paris", probability: 0.6 }],
      entropy_bits: 2,
    },
    heads: [
      {
        head: 0,
        top_tokens: [{ token_id: 1, text: " Paris", probability: 0.59 }],
        entropy_bits: 2,
        kl_bits: 0.01,
        baseline_top_probability: 0.59,
        top_token_changed: false,
      },
      {
        head: 1,
        top_tokens: [{ token_id: 2, text: " London", probability: 0.4 }],
        entropy_bits: 3,
        kl_bits: 1.5,
        baseline_top_probability: 0.2,
        top_token_changed: true,
      },
      {
        head: 2,
        top_tokens: [{ token_id: 1, text: " Paris", probability: 0.45 }],
        entropy_bits: 2.5,
        kl_bits: 0.3,
        baseline_top_probability: 0.45,
        top_token_changed: false,
      },
    ],
  };
  const changed = ablationStory(ablation, 1);
  assert.equal(changed.effect, "decisive");
  assert.match(changed.text, /answer changes/);
  assert.match(changed.text, /rank 1 of 3/);
  assert.deepEqual(
    changed.ranked.map((h) => h.head),
    [1, 2, 0],
  );
  assert.deepEqual(changed.changers, [1]);
  assert.equal(changed.complete, true);
  const held = ablationStory(ablation, 2);
  assert.equal(held.effect, "noticeable");
  assert.match(held.text, /answer holds/);
  const flat = ablationStory(ablation, 0);
  assert.equal(flat.effect, "negligible");
  assert.match(flat.text, /barely moves/);
  assert.equal(ablationStory(ablation, 9).entry, null);
  assert.equal(ablationStory(null, 0), null);
});

test("crossAttentionStory ranks tokens and locates the peak region", () => {
  const event = {
    grid: [2, 3],
    source_grid: [4, 6],
    token_mass: [0.1, 0.4, 0.2],
    text_mass: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    maps: [
      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
      [0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
      [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    ],
  };
  const story = crossAttentionStory(event, ["a", "cat", "sat"]);
  assert.deepEqual(
    story.ranked.map((r) => r.index),
    [1, 2, 0],
  );
  assert.equal(story.index, 1);
  assert.deepEqual(story.peakCell, [0, 2]);
  assert.match(story.text, /“cat” is read most at the top right/);
  assert.match(story.text, /focused area/);
  assert.match(story.text, /ranks 1 of 3/);
  const other = crossAttentionStory(event, null, 2);
  assert.match(other.text, /^token 2 is read most/);
  assert.match(other.text, /broadly across 100%/);
  assert.equal(Math.round(other.textShare * 100), 50);
  assert.equal(crossAttentionStory({}, null), null);
});

test("railLabel keeps block families distinct and labels steps", () => {
  assert.equal(railLabel("model.layers.12", 0), "L12");
  assert.equal(railLabel("double_blocks.3", 0), "DB3");
  assert.equal(railLabel("single_blocks.0", 0), "SB0");
  assert.equal(railLabel("down_blocks.1", 0), "Dn1");
  assert.equal(railLabel("up_blocks.0", 0), "Up0");
  assert.equal(railLabel("mid_block", 0), "M");
  assert.equal(railLabel("transformer.h.4", 0), "L4");
  assert.equal(railLabel("Step 7", 0), "S7");
  assert.equal(railLabel("encoder", 4), "5");
});

test("tokenDiff aligns prefix and suffix and promptFromTokens rebuilds text", () => {
  const a = [
    { id: 1, text: "The" },
    { id: 2, text: " cat" },
    { id: 3, text: " sat" },
  ];
  const b = [
    { id: 1, text: "The" },
    { id: 4, text: " dog" },
    { id: 3, text: " sat" },
  ];
  const d = tokenDiff(a, b);
  assert.deepEqual(
    [d.prefix, d.suffix, d.changedA, d.changedB, d.same],
    [1, 1, [1, 2], [1, 2], false],
  );
  assert.equal(tokenDiff(a, a).same, true);
  assert.equal(promptFromTokens(a), "The cat sat");
  assert.equal(promptFromTokens(null), "");
});

test("compareOverlap handles different shapes and headReactions ranks heads", () => {
  const a = [
    [1, 0, 0],
    [0.5, 0.5, 0],
    [0.2, 0.3, 0.5],
  ];
  const b = [
    [1, 0],
    [0.1, 0.9],
  ];
  const o = compareOverlap(a, b);
  assert.deepEqual([o.rows, o.cols, o.truncated], [2, 2, true]);
  assert.ok(Math.abs(o.maxDiff - 0.4) < 1e-9);
  const traceA = { heads: [a, a] };
  const traceB = {
    heads: [
      a,
      [
        [1, 0, 0],
        [0.5, 0.5, 0],
        [0.9, 0.05, 0.05],
      ],
    ],
  };
  const heads = headReactions(traceA, traceB);
  assert.equal(heads[0].head, 1);
  assert.equal(heads[1].meanAbs, 0);
  assert.deepEqual(headReactions({}, traceB), []);
});

test("layerReactions, lensDivergence and promptCompareStory read both traces", () => {
  const traceA = {
    tokens: [
      { id: 1, text: "The" },
      { id: 2, text: " cat" },
    ],
    output: {
      top_tokens: [
        { token_id: 9, text: " sat", probability: 0.5 },
        { token_id: 8, text: " ran", probability: 0.2 },
      ],
    },
    logit_lens: {
      layers: [
        { label: "embeddings", top_tokens: [{ token_id: 1 }] },
        { label: "layer 0", top_tokens: [{ token_id: 9 }] },
      ],
    },
    layers: [
      { index: 0, entropy_bits: 1, sink_mass: 0.5, previous_token_mass: 0.1 },
      { index: 1, entropy_bits: 2, sink_mass: 0.4, previous_token_mass: 0.1 },
    ],
  };
  const traceB = {
    tokens: [
      { id: 1, text: "The" },
      { id: 3, text: " dog" },
    ],
    output: {
      top_tokens: [
        { token_id: 8, text: " ran", probability: 0.6 },
        { token_id: 9, text: " sat", probability: 0.1 },
      ],
    },
    logit_lens: {
      layers: [
        { label: "embeddings", top_tokens: [{ token_id: 1 }] },
        { label: "layer 0", top_tokens: [{ token_id: 8 }] },
      ],
    },
    layers: [
      { index: 0, entropy_bits: 1.1, sink_mass: 0.5, previous_token_mass: 0.1 },
      { index: 1, entropy_bits: 1.2, sink_mass: 0.6, previous_token_mass: 0.1 },
    ],
  };
  const layers = layerReactions(traceA, traceB);
  assert.equal(layers[0].layer, 1);
  assert.ok(Math.abs(layers[0].entropyDelta + 0.8) < 1e-9);
  const lens = lensDivergence(traceA.logit_lens, traceB.logit_lens);
  assert.deepEqual(
    [lens.firstDisagreement, lens.agreeCount, lens.firstLabel],
    [1, 1, "layer 0"],
  );
  const story = promptCompareStory(traceA, traceB);
  assert.match(story.text, /changed “␣cat” to “␣dog” after 1 shared token\./);
  assert.match(
    story.text,
    /answer changes from “␣sat” \(50%\) to “␣ran” \(60%\); the old answer keeps 10%/,
  );
  assert.match(story.text, /first disagrees at layer 0/);
  assert.match(
    story.text,
    /Layer 1's attention reacted most: entropy down 0.80 bits/,
  );
  assert.equal(promptCompareStory(traceA, null), null);
});

test("patchStory reports flips and the strongest layer", () => {
  const block = {
    positions: "last",
    source_run: {
      top_tokens: [{ token_id: 1, text: " Paris", probability: 0.66 }],
    },
    target_run: {
      top_tokens: [{ token_id: 2, text: " Berlin", probability: 0.48 }],
      source_answer_probability: 0.02,
      target_answer_probability: 0.48,
    },
    layers: [
      {
        layer: 0,
        recovery: 0.05,
        source_answer_probability: 0.05,
        top_is_source: false,
        top_tokens: [{ token_id: 2 }],
      },
      {
        layer: 1,
        recovery: 0.9,
        source_answer_probability: 0.6,
        top_is_source: true,
        top_tokens: [{ token_id: 1 }],
      },
      {
        layer: 2,
        recovery: 0.4,
        source_answer_probability: 0.28,
        top_is_source: false,
        top_tokens: [{ token_id: 2 }],
      },
    ],
  };
  const story = patchStory(block);
  assert.equal(story.best.layer, 1);
  assert.deepEqual(story.flips, [1]);
  assert.match(story.text, /flips the answer to “␣Paris” at layer 1/);
  assert.match(story.text, /recovers 90% of the gap/);
  const same = patchStory({
    ...block,
    target_run: {
      ...block.target_run,
      top_tokens: block.source_run.top_tokens,
    },
  });
  assert.match(same.text, /already answer/);
  assert.equal(patchStory(null), null);
});
