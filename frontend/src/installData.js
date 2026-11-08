import { leastOf } from "./extent.js";
/** Bytes in the largest whole unit, as a file manager would show them. */
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "unknown size";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${unit && value < 100 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

/** What one download is doing, in words, and how far along it is. */
export function installProgress(job) {
  const total = job.total_bytes || 0;
  const fraction = total ? Math.min(1, (job.bytes || 0) / total) : 0;
  const amount = `${formatBytes(job.bytes || 0)} of ${formatBytes(total)}`;
  const label = {
    queued: "Waiting",
    downloading: `Downloading ${amount}`,
    verifying: "Checking the file",
    installed: job.files
      ? `Installed · ${job.files.total} file${job.files.total === 1 ? "" : "s"} checked`
      : job.hash_checked
        ? "Installed · SHA-256 matched"
        : "Installed · no published SHA-256 to check",
    failed: "Failed",
    cancelled: "Stopped",
  }[job.status];
  const files =
    job.files && job.status !== "installed"
      ? ` · ${job.files.done} of ${job.files.total} files`
      : "";
  return { fraction, label: (label ?? job.status) + files };
}

/** How much of a checked plan can go ahead. Free space is per disk, so the smallest is the safe figure. */
export function planTotals(plan) {
  const ready = plan.items.filter((i) => i.ok);
  const free = plan.items
    .map((i) => i.free_bytes)
    .filter((v) => Number.isFinite(v));
  return {
    ready: ready.length,
    bytes: ready.reduce(
      (sum, i) => sum + (i.size_bytes || 0) - (i.resume_bytes || 0),
      0,
    ),
    free: free.length ? leastOf(free) : NaN,
  };
}

const ORDER = ["downloading", "verifying", "queued", "failed", "cancelled"];

/**
 * Downloads as rows: each package's files folded into one row, other files as they are.
 * A package row is as far along as all its files together, and as stuck as its worst one.
 */
export function groupJobs(jobs) {
  const rows = [];
  const packages = new Map();
  for (const job of jobs) {
    if (!job.package) {
      rows.push(job);
      continue;
    }
    if (!packages.has(job.package)) {
      const row = {
        id: `package:${job.package}`,
        package: job.package,
        filename: job.package,
        members: [],
      };
      packages.set(job.package, row);
      rows.push(row);
    }
    packages.get(job.package).members.push(job);
  }
  for (const row of packages.values()) {
    const { members } = row;
    row.status =
      ORDER.find((s) => members.some((j) => j.status === s)) || "installed";
    row.bytes = members.reduce((sum, j) => sum + (j.bytes || 0), 0);
    row.total_bytes = members.reduce((sum, j) => sum + (j.total_bytes || 0), 0);
    row.hash_checked = members.every((j) => j.hash_checked);
    row.error = members.find((j) => j.error)?.error || null;
    row.retrying = members.find((j) => j.retrying)?.retrying || null;
    row.files = {
      done: members.filter((j) => j.status === "installed").length,
      total: members.length,
    };
  }
  return rows;
}
