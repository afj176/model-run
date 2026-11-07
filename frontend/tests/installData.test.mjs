import test from "node:test";
import assert from "node:assert/strict";
import {
  groupJobs,
  formatBytes,
  installProgress,
  planTotals,
} from "../src/installData.js";

test("sizes read like a file manager", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(1536), "1.5 KB");
  assert.equal(formatBytes(2132696762), "2.0 GB");
  assert.equal(formatBytes(150 * 1024 ** 2), "150 MB");
  assert.equal(formatBytes(undefined), "unknown size");
});

test("a download says what it is doing and whether the hash was checked", () => {
  const job = { status: "downloading", bytes: 512, total_bytes: 1024 };
  assert.equal(installProgress(job).fraction, 0.5);
  assert.equal(installProgress(job).label, "Downloading 512 B of 1.0 KB");
  assert.match(
    installProgress({ status: "installed", hash_checked: true }).label,
    /SHA-256 matched/,
  );
  assert.match(
    installProgress({ status: "installed", hash_checked: false }).label,
    /no published/,
  );
  assert.equal(installProgress({ status: "queued" }).fraction, 0);
});

test("a plan only counts what can go ahead, minus what is already downloaded", () => {
  const totals = planTotals({
    items: [
      { ok: true, size_bytes: 100, resume_bytes: 40, free_bytes: 9000 },
      { ok: false, size_bytes: 500, free_bytes: 8000 },
    ],
  });
  assert.deepEqual(totals, { ready: 1, bytes: 60, free: 8000 });
});

test("a package's files fold into one row that is as stuck as its worst file", () => {
  const jobs = [
    {
      id: "a",
      package: "org/m",
      filename: "model.safetensors",
      status: "installed",
      bytes: 90,
      total_bytes: 90,
      hash_checked: true,
    },
    {
      id: "b",
      package: "org/m",
      filename: "tokenizer.json",
      status: "downloading",
      bytes: 5,
      total_bytes: 10,
      hash_checked: false,
    },
    {
      id: "c",
      package: "org/m",
      filename: "config.json",
      status: "queued",
      bytes: 0,
      total_bytes: 1,
    },
    { id: "d", filename: "vae.safetensors", status: "failed", error: "gone" },
  ];
  const rows = groupJobs(jobs);
  assert.equal(rows.length, 2);
  const [pkg, single] = rows;
  assert.equal(pkg.filename, "org/m");
  assert.equal(pkg.status, "downloading");
  assert.equal(pkg.bytes, 95);
  assert.equal(pkg.total_bytes, 101);
  assert.deepEqual(pkg.files, { done: 1, total: 3 });
  assert.match(installProgress(pkg).label, /1 of 3 files$/);
  assert.equal(single, jobs[3]);
  const failed = groupJobs([
    { ...jobs[0] },
    { ...jobs[1], status: "failed", error: "hash" },
  ]);
  assert.equal(failed[0].status, "failed");
  assert.equal(failed[0].error, "hash");
  const done = groupJobs([{ ...jobs[0] }]);
  assert.equal(installProgress(done[0]).label, "Installed · 1 file checked");
});
