"""Opt-in serial native checkpoint loading campaign."""
import argparse
import json
import subprocess
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--python", required=True, type=Path)
parser.add_argument("--comfy-root", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--max-bytes", type=int, default=9 * 1024**3)
parser.add_argument("--timeout", type=int, default=180)
parser.add_argument("roots", nargs="+", type=Path)
args = parser.parse_args()
results = []
for root in args.roots:
    for artifact in sorted(root.glob("*.safetensors")):
        try:
            process = subprocess.run([str(args.python), str(Path(__file__).with_name("probe_comfy_diffusion_load.py")),
                str(args.comfy_root), str(artifact), "--max-bytes", str(args.max_bytes)], capture_output=True, text=True,
                errors="replace", timeout=args.timeout)
            lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
            result = json.loads(lines[-1]) if process.returncode == 0 and lines else {
                "artifact": artifact.name, "status": "failed", "error": process.stderr[-4000:], "exit_code": process.returncode}
        except subprocess.TimeoutExpired:
            result = {"artifact": artifact.name, "status": "timeout", "timeout_seconds": args.timeout}
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
        print(json.dumps({"artifact": artifact.name, "status": result["status"]}), flush=True)
