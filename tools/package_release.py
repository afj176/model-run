"""Build a custom-node ZIP from an explicit allowlist; never bundle models, caches or local reports."""
import re
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
version = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
destination = root / ".tmp" / "dist" / f"model-run-comfyui-{version}.zip"
destination.parent.mkdir(parents=True, exist_ok=True)
files = [root / "__init__.py", root / "pyproject.toml", root / "README.md", root / "COMPATIBILITY_MATRIX.md", root / "Start-ModelRun.ps1"]
files.extend((root / "model_run").glob("*.py"))
files.extend(p for p in (root / "model_run" / "web").rglob("*") if p.is_file())
files.extend((root / "comfy_web").glob("*.js"))
with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, "model-run/" + path.relative_to(root).as_posix())
print(destination)
