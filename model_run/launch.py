"""Start ComfyUI with Model Run loaded, then open Model Run beside the ComfyUI canvas.

ComfyUI runs with its own Python. Model Run is registered through a generated extra-paths file
that points at this checkout, so the ComfyUI installation itself is never changed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

from .comfy_paths import comfy_folder_paths

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / ".runtime"
DEFAULT_PORT = 8188
WAIT_SECONDS = 300
# ComfyUI's own folders are found by ComfyUI itself; these come from Comfy Desktop's shared folders.
SKIP_CATEGORIES = {"custom_nodes"}


def comfy_python(comfy_dir):
    """The Python ComfyUI runs on: its virtual environment, or the portable build's embedded Python."""
    root = Path(comfy_dir)
    for candidate in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python",
                      root / "venv" / "Scripts" / "python.exe", root / "venv" / "bin" / "python",
                      root.parent / "python_embeded" / "python.exe"):
        if candidate.is_file():
            return candidate
    return None


def _yaml_text(text):
    return json.dumps(str(text))  # a JSON string is a valid YAML scalar, with any backslashes escaped


def extra_paths_config(comfy_dir, runtime=RUNTIME):
    """The extra-paths file ComfyUI starts with: Model Run as a custom node, plus Comfy Desktop's shared model folders."""
    folders = {}
    for entry in comfy_folder_paths(comfy_dir)["entries"]:
        if entry.get("section") == "log" and entry["category"] not in SKIP_CATEGORIES:
            folders.setdefault(entry["category"], [])
            if entry["path"] not in folders[entry["category"]]:
                folders[entry["category"]].append(entry["path"])
    lines = ["# Written by Model Run's launcher on each start; edits are replaced.", "model_run:",
             f"  base_path: {_yaml_text(runtime)}", "  custom_nodes: custom_nodes"]
    if folders:
        lines.append("comfy_desktop_shared:")
        for category, paths in sorted(folders.items()):
            lines.append(f"  {category}: |")
            lines.extend(f"    {path}" for path in paths)
    return "\n".join(lines) + "\n"


def link_node_pack(runtime=RUNTIME, root=ROOT):
    """runtime/custom_nodes/model-run, a link to this checkout, so ComfyUI loads it as a node pack."""
    folder = runtime / "custom_nodes"
    folder.mkdir(parents=True, exist_ok=True)
    link = folder / "model-run"
    if link.exists():
        if link.resolve() == root.resolve():
            return link
        raise RuntimeError(f"{link} points somewhere else; remove it and start again")
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(root), str(link))  # a junction needs no administrator rights
    else:
        link.symlink_to(root, target_is_directory=True)
    return link


def _get(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (OSError, ValueError):
        return None


def running(port):
    """model_run when a ComfyUI with Model Run answers on the port, comfyui for one without it, None when nothing does."""
    base = f"http://127.0.0.1:{port}"
    if _get(f"{base}/model-run/api/session") == 200:
        return "model_run"
    if _get(f"{base}/system_stats") == 200:
        return "comfyui"
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Start ComfyUI with Model Run and open both")
    parser.add_argument("--comfy-dir", required=True, help="The ComfyUI folder (the one holding main.py)")
    parser.add_argument("--comfy-python", help="ComfyUI's Python, when it is not the folder's .venv or a portable build's")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    comfy_dir = Path(args.comfy_dir).resolve()
    if not (comfy_dir / "main.py").is_file():
        sys.exit(f"{comfy_dir} has no main.py; pass the ComfyUI folder itself")
    base = f"http://127.0.0.1:{args.port}"
    state = running(args.port)
    if state == "comfyui":
        sys.exit(f"A ComfyUI without Model Run already answers on port {args.port}. Pick another with --port, or stop it first.")
    if state is None:
        python = Path(args.comfy_python) if args.comfy_python else comfy_python(comfy_dir)
        if python is None or not python.is_file():
            sys.exit("ComfyUI's Python was not found; pass it with --comfy-python")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        link_node_pack()
        config = RUNTIME / "extra_model_paths.yaml"
        config.write_text(extra_paths_config(comfy_dir), encoding="utf-8")
        log = open(RUNTIME / "comfyui.log", "ab")
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        process = subprocess.Popen([str(python), "main.py", "--port", str(args.port), "--extra-model-paths-config", str(config)],
                                   cwd=comfy_dir, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **flags)
        print(f"Starting ComfyUI with Model Run (PID {process.pid}); its log is .runtime/comfyui.log", flush=True)
        deadline = time.monotonic() + WAIT_SECONDS
        while running(args.port) != "model_run":
            if process.poll() is not None:
                sys.exit("ComfyUI exited while starting. Check .runtime/comfyui.log.")
            if time.monotonic() > deadline:
                sys.exit(f"ComfyUI did not answer within {WAIT_SECONDS} s. It is still starting (PID {process.pid}); check .runtime/comfyui.log.")
            time.sleep(1)
    print(f"ComfyUI: {base}/\nModel Run: {base}/model-run/", flush=True)
    if not args.no_browser:
        webbrowser.open(f"{base}/")
        webbrowser.open(f"{base}/model-run/")


if __name__ == "__main__":
    main()
