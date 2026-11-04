"""Manual encoder load campaign driven by explicit local cases."""
import argparse
import json
import subprocess
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--python", required=True, type=Path)
parser.add_argument("--comfy-root", required=True, type=Path)
parser.add_argument("--cases", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--forward", action="store_true")
args = parser.parse_args()
results = []
for case in json.loads(args.cases.read_text(encoding="utf-8-sig")):
    try:
        process = subprocess.run([str(args.python), str(Path(__file__).with_name("probe_comfy_encoder.py")),
            str(args.comfy_root), case["path"], case["clip_type"]] + (["--forward"] if args.forward else []), capture_output=True, text=True, errors="replace", timeout=300)
        lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
        result = json.loads(lines[-1]) if process.returncode == 0 and lines else {"artifact": Path(case["path"]).name,
            "status": "failed", "exit_code": process.returncode, "error": process.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        result = {"artifact": Path(case["path"]).name, "status": "timeout"}
    results.append(result)
    args.output.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": result["artifact"], "status": result["status"]}), flush=True)
