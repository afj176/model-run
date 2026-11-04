"""Build a custom-node ZIP from an explicit allowlist; never bundle models or caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
destination = root / ".tmp" / "dist" / "model-run-comfyui-0.1.0.zip"
destination.parent.mkdir(parents=True, exist_ok=True)
files = [root / "__init__.py", root / "pyproject.toml", root / "README.md", root / "Start-ModelRun.ps1",
         root / "RUNTIME_ADAPTERS.md", root / "VALIDATION.md", root / "COMPATIBILITY_MATRIX.md"]
files.extend((root / "model_run").glob("*.py"))
files.extend(p for p in (root / "model_run" / "web").rglob("*") if p.is_file())
files.extend((root / "comfy_web").glob("*.js"))
files.extend((root / "reports").glob("*.md"))
with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, "model-run/" + path.relative_to(root).as_posix())
print(destination)
