"""Manual read-only artifact campaign. Does not depend on personal paths or run in unittest."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_run.inventory import inspect_file

parser = argparse.ArgumentParser()
parser.add_argument("roots", nargs="+", type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
from safetensors import safe_open
results = []
seen = set()
for root in args.roots:
    if not root.is_dir():
        results.append({"path": str(root), "status": "unavailable", "reason": "Root directory not found"})
        continue
    for artifact in sorted(root.rglob("*.safetensors")):
        path = artifact.resolve()
        if path in seen:
            continue
        seen.add(path)
        record = {"path": str(path), "size_bytes": path.stat().st_size}
        try:
            info = inspect_file(path)
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                assert set(keys) == {item["name"] for item in info["tensors"]}
                for item in info["tensors"]:
                    assert list(handle.get_slice(item["name"]).get_shape()) == item["shape"]
            record.update({"status": "passed", "stage": "metadata_cross_validation", "tensor_count": len(keys),
                           "dtypes": sorted({item["dtype"] for item in info["tensors"]}),
                           "inference": "not_run"})
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        results.append(record)
report = {"kind": "local_artifact_campaign", "results": results,
          "limitations": ["Metadata cross-validation does not read all tensor values or establish inference support.",
                          "This manual campaign is independent of automated fixture tests."]}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"artifacts": len(results), "passed": sum(r["status"] == "passed" for r in results),
                  "failed": sum(r["status"] == "failed" for r in results)}))
