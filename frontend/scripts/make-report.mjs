// Generate an insight report from a running Model Run service (verification helper).
// Usage: node scripts/make-report.mjs <service base url> <output dir> [trace index from end]
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { buildReport } from "../src/report.js";

const [base = "http://127.0.0.1:8766/", outDir = ".", pick = "1"] =
  process.argv.slice(2);
const api = new URL("api/", base);
const token = (await (await fetch(new URL("session", api))).json()).token;
const get = async (route) => {
  const r = await fetch(new URL(route, api), {
    headers: { "X-Model-Run-Token": token },
  });
  if (!r.ok) throw new Error(`${route}: ${r.status}`);
  return r.json();
};
const state = await get("state");
mkdirSync(outDir, { recursive: true });
const records =
  Number(pick) === 0 ? state.traces : [state.traces.at(-Number(pick))];
for (const record of records) {
  const trace = await get(`traces/${record.id}`);
  const html = buildReport(trace, { name: record.name });
  const file = join(
    outDir,
    `report-${trace.kind}-${record.id.slice(0, 6)}.html`,
  );
  writeFileSync(file, html);
  console.log(file, `${(html.length / 1024).toFixed(0)} KiB`);
}
