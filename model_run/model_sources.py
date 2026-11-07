"""Model lists ComfyUI already ships, read as data. Nothing here touches the network."""
import glob
import importlib.util
import json
import os
import re
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

MAX_LIST_BYTES = 16 * 1024 * 1024
MAX_TEMPLATES = 5000
INSTALLABLE_SUFFIXES = (".safetensors", ".sft", ".gguf")  # tensors and metadata only; pickle formats can run code
GGUF_NOTE = "Loads through the ComfyUI-GGUF custom node, from the diffusion model or text encoder folder"
TYPE_FOLDERS = {"checkpoint": "checkpoints", "checkpoints": "checkpoints", "lora": "loras", "vae": "vae", "taesd": "vae_approx",
                "upscale": "upscale_models", "controlnet": "controlnet", "t2i-adapter": "controlnet", "clip": "text_encoders",
                "unet": "diffusion_models", "diffusion_model": "diffusion_models", "clip_vision": "clip_vision", "embedding": "embeddings",
                "gligen": "gligen", "style_model": "style_models", "photomaker": "photomaker", "hypernetwork": "hypernetworks"}
FILE_FOLDERS = ((r"(^|[/_.-])(vae|ae)([/_.-]|$)", "vae"), (r"lora", "loras"), (r"controlnet|control_", "controlnet"),
                (r"(^|[/_.-])(clip|t5|umt5|gemma|qwen|text_encoders?|llm)([/_.-]|\d)", "text_encoders"), (r"esrgan|upscale", "upscale_models"),
                (r"(^|[/_.-])(transformer|unet|diffusion_models?|dit)([/_.-]|$)", "diffusion_models"))


def huggingface_parts(url):
    """(repo, revision, path) when a URL is a Hugging Face file; None for anything else."""
    if not isinstance(url, str) or len(url) > 2048:
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "https" or parts.hostname not in ("huggingface.co", "hf.co") or port is not None or parts.username or parts.password:
        return None
    match = re.fullmatch(r"/([\w.-]+/[\w.-]+)/resolve/([^/]+)/(.+)", parts.path)
    if not match or ".." in unquote(match.group(3)).split("/"):
        return None
    return match.group(1), unquote(match.group(2)), unquote(match.group(3))


def file_url(repo, path, revision="main"):
    return f"https://huggingface.co/{repo}/resolve/{quote(revision, safe='')}/{quote(path)}"


def guess_folder(path):
    """A guess from a file's name; the person installing confirms it."""
    lowered = path.lower().replace("\\", "/")
    # ComfyUI-GGUF reads GGUF models from the diffusion model folders, never from checkpoints.
    fallback = "diffusion_models" if lowered.endswith(".gguf") else "checkpoints"
    return next((folder for pattern, folder in FILE_FOLDERS if re.search(pattern, lowered)), fallback)


def parse_size(text):
    match = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?)B?\s*", text or "", re.I)
    return int(float(match.group(1)) * 1024 ** "_KMGT".index(match.group(2).upper() or "_")) if match else None


def blocked_reason(url, filename):
    if huggingface_parts(url) is None:
        return "Only https files from Hugging Face can be installed"
    if not str(filename).lower().endswith(INSTALLABLE_SUFFIXES):
        return "Only .safetensors, .sft and .gguf can be installed: pickle formats such as .ckpt, .pt, .pth and .bin can run code when loaded"
    return None


def install_note(filename):
    """What a person should know before installing this file, or None."""
    return GGUF_NOTE if str(filename).lower().endswith(".gguf") else None


def _read(path):
    if os.path.getsize(path) > MAX_LIST_BYTES:
        raise ValueError("list exceeds 16 MiB")
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _roots(comfy_dir):
    """One ComfyUI folder, or several: --base-directory keeps user data away from the code."""
    if not comfy_dir:
        return []
    return [comfy_dir] if isinstance(comfy_dir, (str, os.PathLike)) else list(dict.fromkeys(comfy_dir))


def _package_dirs(package, comfy_dir):
    found = []
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        found.extend(spec.submodule_search_locations)
    for root in map(Path, _roots(comfy_dir)):
        for pattern in (".venv/Lib/site-packages", ".venv/lib/python*/site-packages", "../standalone-env/Lib/site-packages", "../standalone-env/lib/python*/site-packages"):
            found.extend(glob.glob(str(root / pattern / package)))
    return list(dict.fromkeys(os.path.normpath(p) for p in found if os.path.isdir(p)))


def _model_entries(value, into):
    if isinstance(value, dict):
        models = value.get("models")
        if isinstance(models, list):
            into.extend(m for m in models if isinstance(m, dict) and m.get("url") and m.get("directory") and m.get("name"))
        for item in value.values():
            _model_entries(item, into)
    elif isinstance(value, list):
        for item in value:
            _model_entries(item, into)


def _number(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def read_templates(comfy_dir, notes):
    """{template name: {title, description, kind, media, rank, size_bytes, models}} from ComfyUI's workflow templates.

    The index groups templates into modules (Image, Video, Audio, 3D Model...) and gives each one a search
    rank and the size of everything it downloads; a template listed in several modules keeps the first.
    """
    templates = {}
    for package in _package_dirs("comfyui_workflow_templates_json", comfy_dir):
        meta = {}
        try:
            for module in _read(os.path.join(package, "templates", "index.json")):
                for item in module.get("templates", []):
                    if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"] not in meta:
                        meta[item["name"]] = {"title": str(item.get("title") or ""), "description": str(item.get("description") or ""),
                                              "kind": str(module.get("title") or "")[:40], "media": str(module.get("type") or "")[:20],
                                              "rank": _number(item.get("searchRank")) or 0, "size_bytes": _number(item.get("size"))}
        except (OSError, ValueError, AttributeError) as exc:
            notes.append(f"Workflow titles unavailable: {exc}"[:200])
        for path in sorted(glob.glob(os.path.join(package, "templates", "*.json")))[:MAX_TEMPLATES]:
            name = Path(path).stem
            if name.startswith("index"):
                continue
            entries = []
            try:
                _model_entries(_read(path), entries)
            except (OSError, ValueError):
                continue
            if entries:
                info = meta.get(name) or {"title": "", "description": "", "kind": "", "media": "", "rank": 0, "size_bytes": None}
                templates[name] = {**info, "title": info["title"] or name.replace("_", " "), "models": entries}
    return templates


TEMPLATE_NAME = re.compile(r"[A-Za-z0-9][\w.-]{0,199}")


def template_thumbnail(name, comfy_dir=None):
    """The installed preview image for a workflow template, or None. ComfyUI ships them in its template media packages."""
    if not isinstance(name, str) or not TEMPLATE_NAME.fullmatch(name):
        return None
    for package in _package_dirs("comfyui_workflow_templates_json", comfy_dir):
        for media in sorted(glob.glob(os.path.join(os.path.dirname(package), "comfyui_workflow_templates_media_*"))):
            candidate = os.path.join(media, "templates", f"{name}-1.webp")
            if os.path.isfile(candidate):
                return candidate
    return None


def read_manager(comfy_dir, notes):
    paths = [os.path.join(p, "model-list.json") for p in _package_dirs("comfyui_manager", comfy_dir)]
    for root in map(Path, _roots(comfy_dir)):
        paths += glob.glob(str(root / "user" / "__manager" / "cache" / "*model-list.json"))
        paths += glob.glob(str(root / "custom_nodes" / "*" / "model-list.json"))
    models = []
    for path in dict.fromkeys(paths):
        try:
            value = _read(path)
            models.extend(m for m in value.get("models", []) if isinstance(m, dict))
        except (OSError, ValueError, AttributeError) as exc:
            notes.append(f"ComfyUI Manager list unavailable: {exc}"[:200])
    return models


def _split_target(directory, name):
    parts = [p for p in str(directory).replace("\\", "/").split("/") if p] + [p for p in str(name).replace("\\", "/").split("/") if p]
    return (parts[0], "/".join(parts[1:-1]), parts[-1]) if len(parts) >= 2 else ("", "", str(name))


def load_index(comfy_dir=None):
    """Merge both lists by URL; each entry keeps every workflow that needs it."""
    notes, items = [], {}
    def entry(url, category, subfolder, filename):
        key = url.split("?")[0]
        if key not in items:
            items[key] = {"id": sha256(key.encode()).hexdigest()[:16], "name": filename, "filename": filename, "url": key, "category": category,
                          "subfolder": subfolder, "size_bytes": None, "sha256": None, "description": "", "base": "", "sources": [], "workflows": []}
        return items[key]
    templates = read_templates(comfy_dir, notes)
    workflows = []
    for name, template in sorted(templates.items()):
        title, ids = template["title"], []
        for model in template["models"]:
            category, subfolder, filename = _split_target(model["directory"], model["name"])
            item = entry(str(model["url"]), category, subfolder, filename)
            if "templates" not in item["sources"]:
                item["sources"].append("templates")
            if str(model.get("hash_type", "")).upper() == "SHA256" and re.fullmatch(r"[0-9a-fA-F]{64}", str(model.get("hash", ""))):
                item["sha256"] = str(model["hash"]).lower()
            if name not in [w["name"] for w in item["workflows"]]:
                item["workflows"].append({"name": name, "title": title})
            if item["id"] not in ids:
                ids.append(item["id"])
        workflows.append({"name": name, "title": title, "description": template["description"][:400], "kind": template["kind"],
                          "media": template["media"], "rank": template["rank"], "size_bytes": template["size_bytes"], "models": ids})
    for model in read_manager(comfy_dir, notes):
        url, filename = model.get("url"), model.get("filename")
        if not isinstance(url, str) or not isinstance(filename, str) or not url.startswith("http"):
            continue
        save_path = str(model.get("save_path") or "default")
        directory = TYPE_FOLDERS.get(str(model.get("type", "")).lower(), "") if save_path == "default" else save_path
        category, subfolder, name = _split_target(directory or "unknown", filename)
        item = entry(url, category, subfolder, name)
        if "manager" not in item["sources"]:
            item["sources"].append("manager")
        item["description"] = item["description"] or str(model.get("description") or "")[:400]
        item["base"] = item["base"] or str(model.get("base") or "")[:64]
        item["display"] = str(model.get("name") or "")[:200]
        item["size_bytes"] = item["size_bytes"] or parse_size(str(model.get("size") or ""))
    for item in items.values():
        item["blocked"] = blocked_reason(item["url"], item["filename"])
        item["note"] = install_note(item["filename"])
    if not items:
        notes.append("No ComfyUI model lists were found here. Hugging Face search still works.")
    return {"models": items, "by_id": {item["id"]: item for item in items.values()}, "workflows": workflows, "notes": notes}
