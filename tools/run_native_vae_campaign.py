"""Opt-in native VAE campaign; explicit paths, isolated processes, bounded duration."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--python", required=True, type=Path)
parser.add_argument("--comfy-root", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--timeout", type=int, default=180)
parser.add_argument("roots", nargs="+", type=Path)
args = parser.parse_args()
results = []
for root in args.roots:
    for artifact in sorted(root.glob("*.safetensors")):
        command = [str(args.python), str(Path(__file__).with_name("probe_comfy_vae.py")), str(args.comfy_root), str(artifact)]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, errors="replace")
            lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
            if process.returncode == 0 and lines:
                result = json.loads(lines[-1])
            else:
                result = {"artifact": artifact.name, "status": "failed", "exit_code": process.returncode,
                          "error": process.stderr[-4000:], "stdout": process.stdout[-2000:]}
        except subprocess.TimeoutExpired:
            result = {"artifact": artifact.name, "status": "timeout", "timeout_seconds": args.timeout}
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
        print(json.dumps({"artifact": artifact.name, "status": result["status"]}), flush=True)
