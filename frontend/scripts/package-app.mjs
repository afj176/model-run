import { cp, mkdir, readdir, unlink, readFile, writeFile } from "node:fs/promises";
import { resolve, join } from "node:path";
await mkdir("../model_run/web", { recursive: true });
await cp("dist/client", "../model_run/web", { recursive: true });
const assets = resolve("../model_run/web/assets");
const current = new Set(await readdir("dist/client/assets"));
// Remove only obsolete hashed build assets in this explicit output directory.
for (const file of await readdir(assets)) {
  if (/^index-[\w-]+\.(js|css)$/.test(file) && !current.has(file))
    await unlink(join(assets, file));
}
console.log("Packaged frontend into model_run/web");
let notices = "Third-party frontend licenses\n\n";
for (const name of ["react", "react-dom", "scheduler", "three", "@phosphor-icons/react"]) {
  const pkg = JSON.parse(await readFile(`node_modules/${name}/package.json`, "utf8"));
  notices += `${name} ${pkg.version}\n${await readFile(`node_modules/${name}/LICENSE`, "utf8")}\n\n`;
}
await writeFile("../model_run/web/THIRD_PARTY_NOTICES.txt", notices);
