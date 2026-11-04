import argparse
import json
import sys
from pathlib import Path
from .inventory import InventoryError, inspect


def main():
    parser = argparse.ArgumentParser(description="Inspect supported local model artifacts without executing them.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        if args.path.suffix.lower() == ".onnx":
            from .onnx_graph import inspect_onnx
            result = inspect_onnx(args.path)
        else:
            result = inspect(args.path)
        print(json.dumps(result, indent=2, ensure_ascii=True))
    except (InventoryError, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
