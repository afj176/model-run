import test from "node:test";
import assert from "node:assert/strict";
import {
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
