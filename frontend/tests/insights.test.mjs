import {
  comparableFrames,
  frameDifferenceStory,
  pixelDifference,
} from "../src/insights.js";
import { eventForLayer, stopSeries } from "../src/insights.js";
import { layerCaptured } from "../src/insights.js";
import { attentionPastAudio } from "../src/insights.js";
import { runParts, validIndex } from "../src/insights.js";
import { speechTokenStory } from "../src/insights.js";
import test from "node:test";
import assert from "node:assert/strict";
import {
  imageAttentionStory,
  layerLabels,
  layerStops,
  moduleInStop,
} from "../src/insights.js";
import {
  ablationStory,
  responseSegments,
  responseStory,
  layerNodePath,
  diffusionCompareStory,
  stepDeltas,
  tokenMassShift,
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

test("stepDeltas, tokenMassShift and diffusionCompareStory read the two runs", () => {
  const a = [
    { step: 0, values: [1, 1, 1, 1] },
    { step: 1, values: [2, 2, 2, 2] },
  ];
  const b = [
    { step: 0, values: [1, 1, 1, 1] },
    { step: 1, values: [2, 2, 0, 0] },
  ];
  const deltas = stepDeltas(a, b);
  assert.equal(deltas.length, 2);
  assert.equal(deltas[0].relative, 0);
  assert.ok(Math.abs(deltas[1].relative - Math.sqrt(2) / 2) < 1e-9);
  const shift = tokenMassShift(
    {
      text_labels: ["a", " cat", "<|endoftext|>", "<|endoftext|>"],
      events: [{ token_mass: [0.1, 0.3, 0.01, 0.01] }],
    },
    { events: [{ token_mass: [0.2, 0.05, 0.01, 0.01] }] },
  );
  assert.equal(shift.ranked[0].index, 1);
  assert.equal(shift.ranked.length, 2); // padding is left out of the ranking
  const story = diffusionCompareStory({
    ablation: { labels: [" cat"], applied_attention_calls: 12 },
    pixels: {
      mean: 0.12,
      grid: [0, 0, 0, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    deltas,
    shift,
  });
  assert.match(story.text, /“␣cat” masked out/);
  assert.match(story.text, /most at the top right/);
  assert.match(story.text, /diverge from step 1/);
  assert.match(story.text, /Attention moved most for “␣cat”/);
  assert.equal(diffusionCompareStory({}), null);
});

test("layerNodePath finds the layer node across families", () => {
  const nodes = [
    { path: "model" },
    { path: "model.layers.0" },
    { path: "model.layers.12" },
    { path: "model.layers.12.self_attn" },
  ];
  assert.equal(layerNodePath(nodes, 12), "model.layers.12");
  assert.equal(
    layerNodePath([{ path: "transformer.h.3" }], 3),
    "transformer.h.3",
  );
  assert.equal(layerNodePath(nodes, 99), null);
  assert.equal(layerNodePath(null, 0), null);
});

test("responseStory summarises confidence and where tokens settled", () => {
  const trace = {
    response: " Paris, the city",
    stopped: "eos",
    steps: [
      {
        index: 0,
        text: " Paris",
        probability: 0.66,
        entropy_bits: 2.1,
        lens: { chosen_probability_by_layer: [0, 0.1, 0.6], settled_layer: 1 },
      },
      {
        index: 1,
        text: ",",
        probability: 0.3,
        entropy_bits: 3.3,
        lens: { chosen_probability_by_layer: [0, 0.2, 0.3], settled_layer: 2 },
      },
      {
        index: 2,
        text: " the",
        probability: 0.9,
        entropy_bits: 0.5,
        lens: { chosen_probability_by_layer: [0, 0.8, 0.9], settled_layer: 1 },
      },
    ],
  };
  const story = responseStory(trace);
  assert.match(story.text, /wrote “Paris,␣the␣city”/);
  assert.match(story.text, /surest of “␣the” \(90%\)/);
  assert.match(story.text, /least sure of “,” \(30%, 3.3 bits\)/);
  assert.match(
    story.text,
    /Every token settled by the output; the typical token was decided at layer 0 of 2, and 1 only at the final layer/,
  );
  assert.match(story.text, /chose to stop/);
  assert.equal(story.leastConfident.index, 1);
  assert.equal(responseStory({ steps: [] }), null);
});

test("responseSegments finds the thinking span and the story reports it", () => {
  const mk = (text, p = 0.5) => ({
    text,
    probability: p,
    entropy_bits: 1,
    top_tokens: [
      { text, probability: p },
      { text: "x", probability: p - 0.05 },
    ],
  });
  const steps = [
    mk("<think>"),
    mk("hmm", 0.4),
    mk("ok", 0.6),
    mk("</think>"),
    mk(" Paris", 0.9),
  ].map((s, i) => ({ ...s, index: i }));
  const seg = responseSegments(steps);
  assert.deepEqual(seg, { thinking: [1, 3], closed: true, answerStart: 4 });
  assert.equal(responseSegments([mk("a")]).thinking, null);
  const open = responseSegments([mk("<think>"), mk("a")]);
  assert.deepEqual([open.thinking, open.closed], [[1, 2], false]);
  const story = responseStory({
    steps,
    stopped: "eos",
    response: "<think>hmmok</think> Paris",
  });
  assert.match(story.text, /thought for 2 tokens before answering with 1/);
  assert.equal(story.closeCalls.length, 4);
});

test("crossAttentionStory ranks words above padding and reads the chosen frame", () => {
  const event = {
    grid: [1, 2],
    frames: 2,
    token_mass: [0.5, 0.1, 0.05],
    // frame 0 then frame 1, two regions each
    maps: [
      [0.5, 0.5, 0.5, 0.5],
      [0.2, 0, 0, 0.2],
      [0.1, 0, 0, 0.1],
    ],
  };
  const labels = ["<pad>", " cat", " sat"];
  const first = crossAttentionStory(event, labels);
  assert.equal(first.index, 1);
  assert.deepEqual(
    first.ranked.map((r) => r.index),
    [1, 2],
  );
  assert.deepEqual(first.peakCell, [0, 0]);
  assert.match(first.text, /ranks 1 of 2 tokens/);
  assert.match(
    first.text,
    /77% of this block's attention to text lands on padding/,
  );
  assert.deepEqual(crossAttentionStory(event, labels, 1, 1).peakCell, [0, 1]);
  assert.equal(crossAttentionStory(event, labels, 0).index, 0);
});

test("stepDeltas lines up ComfyUI runs by module and denoiser call", () => {
  const event = (module, call, values) => ({
    module,
    denoiser_call: call,
    values,
  });
  const a = [
    event("x", 0, [1, 1]),
    event("y", 0, [5, 5]),
    event("x", 1, [2, 2]),
  ];
  const b = [
    event("x", 0, [1, 1]),
    event("y", 0, [9, 9]),
    event("x", 1, [2, 0]),
  ];
  const deltas = stepDeltas(a, b);
  assert.deepEqual(
    deltas.map((d) => [d.step, d.module]),
    [
      [0, "x"],
      [1, "x"],
    ],
  );
  assert.equal(deltas[0].relative, 0);
  const story = diffusionCompareStory({ deltas });
  assert.match(story.text, /The observed output diverges from call 1/);
});

test("imageAttentionStory finds the strongest patch and its concentration", () => {
  const story = imageAttentionStory([0.1, 0.1, 0.1, 0.5, null, 0.2], [2, 3]);
  assert.equal(story.peak, 0.5);
  assert.deepEqual([story.row, story.col], [1, 0]);
  assert.equal(story.count, 6);
  assert.ok(
    Math.abs(story.top4Share - 0.9 / 1) < 1e-9 || story.top4Share > 0.8,
  );
  assert.equal(imageAttentionStory([], [1, 1]), null);
});

test("layer stops tell encoder and decoder apart", () => {
  const whisper = [
    "model.encoder.layers.0",
    "model.encoder.layers.1",
    "model.decoder.layers.0",
    "model.decoder.layers.1",
  ];
  assert.deepEqual(layerLabels(whisper), ["E0", "E1", "D0", "D1"]);
  const stops = layerStops([...whisper, "model.encoder.layer_norm"]);
  assert.equal(stops.length, 4);
  assert.equal(stops[2].name, "decoder layer 0");
  assert.equal(stops[2].repeated, true);
  assert.equal(moduleInStop("model.decoder.layers.0", stops[0]), false);
  assert.equal(moduleInStop("model.decoder.layers.0.fc1", stops[2]), true);
  const flux = layerStops(["double_blocks.0", "single_blocks.0"]);
  assert.equal(flux[1].name, "single blocks layer 0");
  const single = layerStops(["model.layers.0", "model.layers.1"]);
  assert.deepEqual(
    single.map((s) => s.label),
    ["L0", "L1"],
  );
  assert.equal(single[1].name, "Layer 1");
  assert.equal(single[0].repeated, false);
  // Native traces record a module path relative to the denoiser.
  assert.equal(
    moduleInStop("transformer_blocks.0", {
      path: "model.diffusion_model.transformer_blocks.0",
    }),
    true,
  );
});

test("speechTokenStory places the strongest slice in seconds", () => {
  const story = speechTokenStory([0.1, null, 0.4, 0.1], 0.2, 8);
  assert.equal(story.peak, 0.4);
  assert.equal(story.peakSeconds, 5);
  assert.equal(story.outside, 0.2);
  assert.equal(speechTokenStory([], null, 8).peak, 0);
});

test("validIndex drops a remembered position the new list no longer has", () => {
  assert.equal(validIndex(15, 4, 1), 1);
  assert.equal(validIndex(2, 4, 1), 2);
  assert.equal(validIndex(0, 4, 1), 0);
  assert.equal(validIndex(null, 4, 1), 1);
  assert.equal(validIndex(0, 0, -1), -1);
});

test("runParts finds a trace's run, with later parts replacing earlier ones", () => {
  const traces = [
    { id: "a", run: "r", step: "module" },
    { id: "b", run: "r", step: "attention" },
    { id: "c", run: "r", step: "attention" },
    { id: "d", run: "other", step: null },
  ];
  assert.deepEqual(runParts(traces, "a"), {
    run: "r",
    ids: { module: "a", attention: "c" },
  });
  assert.equal(runParts(traces, "d"), null);
  assert.equal(runParts(traces, "gone"), null);
});

test("attentionPastAudio warns from the final layer only", () => {
  const speech = [{ outside: [0.9, 0.9] }, { outside: [0.1, 0.2] }];
  assert.equal(attentionPastAudio(speech).doubtful, false);
  assert.ok(Math.abs(attentionPastAudio(speech).share - 0.15) < 1e-9);
  assert.equal(attentionPastAudio([{ outside: [0.7, 0.8] }]).doubtful, true);
  assert.equal(attentionPastAudio([]), null);
});

test("layerCaptured only accepts layers this run recorded", () => {
  const events = [
    { module: "double_blocks.0" },
    { module: "model.encoder.layers.1.fc" },
  ];
  assert.equal(layerCaptured("double_blocks.0", events), true);
  assert.equal(layerCaptured("double_blocks.1", events), false);
  assert.equal(layerCaptured("model.encoder.layers.1", events), true);
  assert.equal(layerCaptured("x", [{ step: 1 }]), false);
  assert.equal(layerCaptured("x", null), false);
});

test("moving between layers stays on the same denoiser call", () => {
  const events = [
    { module: "a", denoiser_call: 0 },
    { module: "b", denoiser_call: 0 },
    { module: "a", denoiser_call: 1 },
    { module: "b", denoiser_call: 1 },
  ];
  const isB = (e) => e.module === "b";
  assert.equal(eventForLayer(events, isB, events[2]), 3);
  assert.equal(eventForLayer(events, isB, { denoiser_call: 9 }), 1);
  assert.equal(
    eventForLayer(events, (e) => e.module === "z", events[2]),
    -1,
  );
});

test("stopSeries plots by timeline position, not repeated layer numbers", () => {
  const stops = [
    { path: "double_blocks.0" },
    { path: "single_blocks.0" },
    { path: "single_blocks.1" },
  ];
  const events = [
    { module: "double_blocks.0", values: [3, 4] },
    { module: "single_blocks.0", values: [1, 1] },
  ];
  const series = stopSeries(stops, events);
  assert.deepEqual(
    series.points.map((p) => p.layer),
    [0, 1],
  );
  assert.ok(Math.abs(series.points[1].value - 1) < 1e-9);
  assert.equal(stopSeries(stops, []), null);
});

test("pixelDifference measures only colour, on the right cell", () => {
  const a = new Uint8ClampedArray([0, 0, 0, 255, 0, 0, 0, 255]);
  const b = new Uint8ClampedArray([255, 255, 255, 0, 0, 0, 0, 0]);
  const result = pixelDifference(a, b, 2, 1);
  assert.equal(result.mean, 0.5);
  assert.equal(result.image.length, 8);
  assert.equal(result.image[7], 255);
  assert.equal(result.grid[0], 1);
});

test("frames are only compared when both runs kept the same ones", () => {
  const run = (indices) => ({
    preview: { base64: "first" },
    preview_frames: {
      frames: indices.map((i) => `f${i}`),
      frame_indices: indices,
    },
  });
  const same = comparableFrames(run([0, 4]), run([0, 4]));
  assert.deepEqual(same.b, ["f0", "f4"]);
  assert.equal(same.note, null);
  const different = comparableFrames(run([0, 4]), run([0, 8]));
  assert.deepEqual(different.a, ["first"]);
  assert.match(different.note, /first frames/);
  const mixed = comparableFrames(run([0, 4]), { preview: { base64: "x" } });
  assert.deepEqual(mixed.b, ["x"]);
  assert.match(mixed.note, /first frames/);
  assert.equal(comparableFrames({}, {}).a.length, 0);
});

test("frameDifferenceStory names the frames that differ most and least", () => {
  const story = frameDifferenceStory([0.1, 0.4, 0.2], [0, 4, 8]);
  assert.equal(story.peak, 1);
  assert.match(story.text, /largest at frame 4/);
  assert.match(story.text, /smallest at frame 0/);
  assert.equal(frameDifferenceStory([0.1], [0]), null);
});

test("compare wording says nothing it cannot support", () => {
  assert.match(
    frameDifferenceStory([0.08, 0.081, 0.079], [0, 4, 8]).text,
    /about the same across all 3 frames/,
  );
  const story = diffusionCompareStory({
    ablation: { labels: [" red"] },
    ablatedSide: "A",
    shift: { ranked: [{ index: 0, label: "a", delta: -1e-6 }], peak: 1e-6 },
  });
  assert.match(story.text, /A ran with/);
  assert.match(story.text, /barely moved/);
  assert.doesNotMatch(story.text, /-0\.00/);
  assert.match(
    diffusionCompareStory({ ablation: { labels: ["<pad>", "<pad>", " red"] } })
      .text,
    /\u201c<pad>\u201d \u00d7 2, \u201c.?red\u201d/,
  );
});

test("an audio map is described over time, not as an image region", () => {
  const story = crossAttentionStory(
    {
      stream: "audio",
      grid: [1, 4],
      token_mass: [0.2, 0.1],
      maps: [
        [0.1, 0.1, 0.4, 0.2],
        [0, 0, 0, 0.1],
      ],
    },
    [" cat", " sat"],
  );
  assert.match(story.text, /63% of the way through the audio/);
  assert.doesNotMatch(story.text, /top|bottom|centre/);
  const flat = crossAttentionStory(
    {
      stream: "audio",
      grid: [1, 3],
      token_mass: [0.1],
      maps: [[0.1, 0.11, 0.1]],
    },
    [" cat"],
  );
  assert.match(flat.text, /about evenly across the audio/);
});

test("registers are described as registers, and their share is not called padding", () => {
  const labels = ["<bos>", " cube", "<register>", "<register>"];
  const story = crossAttentionStory(
    {
      grid: [1, 2],
      token_mass: [0.01, 0.05, 0.4, 0.5],
      maps: [
        [0.01, 0.01],
        [0.05, 0.05],
        [0.4, 0.4],
        [0.5, 0.5],
      ],
    },
    labels,
  );
  assert.equal(story.index, 1);
  assert.deepEqual(
    story.ranked.map((r) => r.index),
    [1, 0],
  );
  assert.match(story.text, /learned registers/);
  assert.doesNotMatch(story.text, /padding/);
});
