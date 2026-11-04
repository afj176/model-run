"""Read-only ComfyUI model catalog command."""
import argparse
import json
import sys
from pathlib import Path
from .catalog import discover_models
from .comfy_paths import read_extra_paths, standard_model_paths


def main():
    parser = argparse.ArgumentParser(description="Inventory explicitly configured ComfyUI model paths.")
    parser.add_argument("--models-dir", type=Path, help="ComfyUI model root")
    parser.add_argument("--extra-paths", type=Path, action="append", default=[], help="Extra paths YAML; may be repeated")
    parser.add_argument("--max-files", type=int, default=10000)
    args = parser.parse_args()
    if not args.models_dir and not args.extra_paths:
        parser.error("Specify --models-dir, --extra-paths, or both")
    try:
        reports = ([standard_model_paths(args.models_dir)] if args.models_dir else []) + [read_extra_paths(path) for path in args.extra_paths]
        entries = [entry for report in reports for entry in report["entries"] if entry["exists"]]
        result = discover_models(entries, max_files=args.max_files)
        result["path_sources"] = reports
        print(json.dumps(result, ensure_ascii=True, indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
