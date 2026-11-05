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
    installed: job.hash_checked
      ? "Installed · SHA-256 matched"
      : "Installed · no published SHA-256 to check",
    failed: "Failed",
    cancelled: "Stopped",
  }[job.status];
  return { fraction, label: label ?? job.status };
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
    free: free.length ? Math.min(...free) : NaN,
  };
}
