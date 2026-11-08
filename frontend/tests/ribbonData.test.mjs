import test from "node:test";
import assert from "node:assert/strict";
import {
  answerAttention,
  attentionReceived,
  clampWindow,
  normalize,
  segmentShares,
  sentenceSegments,
  tokenNorms,
  windowShowing,
} from "../src/ribbonData.js";

const tokens = (text) =>
  text.split(" ").map((t, i) => ({ text: (i ? " " : "") + t, id: i }));

test("a window stays inside the prompt and a short prompt is one window", () => {
  assert.deepEqual(clampWindow(400, 24, 480), { start: 400, size: 24 });
  assert.deepEqual(clampWindow(470, 24, 480), { start: 456, size: 24 });
  assert.deepEqual(clampWindow(-5, 24, 480), { start: 0, size: 24 });
  assert.deepEqual(clampWindow(3, 24, 9), { start: 0, size: 9 });
});

test("showing a token keeps the window still when it already holds it", () => {
  const current = { start: 100, size: 24 };
  assert.deepEqual(windowShowing(110, current, 480), current);
  assert.deepEqual(windowShowing(300, current, 480), { start: 288, size: 24 });
  assert.deepEqual(windowShowing(2, null, 9), { start: 0, size: 9 });
});

test("token norms come from a token grid, raw or derived", () => {
  assert.deepEqual(tokenNorms({ slices: { token_rms: [1, 2, NaN] } }), [
    1,
    2,
    null,
  ]);
  assert.deepEqual(tokenNorms({ token_rms: [0.5] }), [0.5]);
  assert.equal(tokenNorms({ values: [] }), null);
});

test("the answer's attention onto the prompt averages the generated steps' rows", () => {
  const steps = [
    { attention: [0.5, 0.25, 0.25] },
    { attention: [0.1, 0.1, 0.4, 0.4] },
    { text: "no attention recorded" },
  ];
  assert.deepEqual(answerAttention(steps, 3), [0.3, 0.175, 0.325]);
  assert.equal(answerAttention([], 3), null);
});

test("normalize maps the robust range onto 0…1 and keeps gaps", () => {
  const out = normalize([0, 1, 2, 3, null, 100]);
  assert.equal(out[4], null);
  assert.equal(out[0], 0);
  assert.equal(out[5], 1);
  assert.ok(out[2] > out[1]);
});

test("a prompt folds into sentences, and a run-on sentence breaks at a clause", () => {
  const segments = sentenceSegments(
    tokens("Explain how it works . Then stop ! Really ? yes"),
  );
  assert.deepEqual(
    segments.map((s) => [s.start, s.end]),
    [
      [0, 5],
      [5, 8],
      [8, 10],
      [10, 11],
    ],
  );
  assert.equal(segments[0].text, "Explain how it works .");
  const long = sentenceSegments(
    tokens(
      Array.from({ length: 60 }, (_, i) => (i === 49 ? "so;" : "w")).join(" "),
    ),
    48,
  );
  assert.deepEqual(
    long.map((s) => [s.start, s.end]),
    [
      [0, 50],
      [50, 60],
    ],
  );
});

test("segment shares split a measure by sentence", () => {
  const segments = [
    { start: 0, end: 2 },
    { start: 2, end: 4 },
  ];
  assert.deepEqual(segmentShares(segments, [1, 1, 3, 3]), [0.25, 0.75]);
  assert.deepEqual(segmentShares(segments, [0, 0, 0, 0]), [0, 0]);
});

test("attention received is the causal column mean, spread over pooled blocks", () => {
  const matrix = [
    [1, 0, 0],
    [0.5, 0.5, 0],
    [0.2, 0.3, 0.5],
  ];
  const plain = attentionReceived(matrix, null, 3);
  assert.deepEqual(
    plain.map((v) => Number(v.toFixed(3))),
    [0.567, 0.4, 0.5],
  );
  const pooled = attentionReceived(
    matrix,
    {
      token_count: 6,
      ranges: [
        [0, 2],
        [2, 4],
        [4, 6],
      ],
    },
    6,
  );
  assert.equal(pooled.length, 6);
  assert.equal(pooled[0], pooled[1]);
  assert.equal(pooled[5], 0.5);
});
