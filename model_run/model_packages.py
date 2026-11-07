"""Hugging Face repositories as complete local packages: whether one runs here, and which of its files that takes."""
import json
from pathlib import Path, PurePosixPath
import re

from .inventory import check_gguf, inspect_file
from .media_models import task_for_config

# Models checked end to end in the standalone app (see the README). Sizes and access come from Hugging Face.
STARTERS = (
    ("Qwen/Qwen3-0.6B", "A small chat model and the quickest first run: layers, the response and attention."),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "A tiny chat model that runs anywhere, even on the CPU."),
    ("openai/whisper-tiny", "Speech to text: the spectrogram it heard and where each word looked."),
    ("google/vit-base-patch16-224", "An image classifier: top labels and where the class token looks."),
    ("google/embeddinggemma-300m", "Sentence embeddings: each layer and the final sentence vector."),
    ("google/gemma-3-4b-it", "A larger chat model; it needs a GPU with room for 8 GB."),
    ("google/gemma-4-E4B-it", "Gemma 4 as a language model; it needs a GPU with room for 15 GB."),
    ("Qwen/Qwen3-8B", "A larger Qwen3; it needs a GPU with room for 16 GB."),
)
KINDS = {"text": "Language model", "embedding": "Embedding model", "image": "Image classifier",
         "audio": "Speech recognition", "diffusion": "Diffusers pipeline"}
TEXT_SUFFIXES = (".json", ".txt", ".model", ".jinja")
MAX_TEXT_BYTES = 64 * 1024 * 1024
MAX_PACKAGE_FILES = 160
# Folders holding the same model in formats the app does not load.
SKIP_FOLDERS = {"onnx", "openvino", "original", "coreml", "flax", "tf", "tflite", "mlx", "gguf", "metal", "runs", "logs"}
# The files that make a folder a package; written last, so a half-downloaded package is never listed.
MARKERS = ("config.json", "model_index.json")
# A file still downloading; the installer names its partial files this way.
PART_SUFFIX = ".modelrun-part"


def _repo_parts(repo):
    if not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or ".." in repo:
        raise ValueError("Invalid repository name")
    return repo.split("/")


def package_folder(root, repo):
    """Where a repository's package goes: <root>/<owner>/<name>, so the folder reads as the model's name."""
    owner, name = _repo_parts(repo)
    return Path(root) / owner / name


def is_installed(root, repo):
    """A package is here when its marker is, and no file of it is still part-downloaded."""
    if root is None:
        return False
    folder = package_folder(root, repo)
    return any((folder / marker).is_file() for marker in MARKERS) and not any(folder.rglob(f"*{PART_SUFFIX}"))


def _names(info):
    siblings = info.get("siblings") if isinstance(info, dict) else None
    return [s["rfilename"] for s in siblings or [] if isinstance(s, dict) and isinstance(s.get("rfilename"), str)]


VARIANT = re.compile(r"\.(fp16|fp32|bf16|fp8\w*|int8\w*|int4\w*|non_ema|ema_only|ema)(-\d+-of-\d+)?\.safetensors$", re.I)


def _variant(name):
    """model.fp16.safetensors and the like: copies the loader does not ask for. Shards such as
    model.safetensors-00001-of-00004.safetensors are not variants."""
    return bool(VARIANT.search(name))


def _weights(names):
    """The safetensors files a loader reads from one folder: model*.safetensors when there are any."""
    plain = [n for n in names if n.lower().endswith(".safetensors") and not _variant(PurePosixPath(n).name)]
    usual = [n for n in plain if PurePosixPath(n).name.startswith(("model", "diffusion_pytorch_model"))]
    return usual or plain


def classify(info, causal_types=None):
    """Whether the standalone app can run a repository as published.

    verdict is runs_here, comfyui (single weight files or GGUF, as ComfyUI uses them) or inspect,
    with the reason in words."""
    tags = set(info.get("tags") or [])
    config = info.get("config") if isinstance(info.get("config"), dict) else {}
    names = _names(info)
    have = set(names)
    gguf = any(n.lower().endswith(".gguf") for n in names)
    loose = any(n.lower().endswith((".safetensors", ".sft")) for n in names)

    def verdict(value, reason, kind=None):
        return {"verdict": value, "reason": reason, "kind": kind, "kind_label": KINDS.get(kind)}

    if "custom_code" in tags or config.get("auto_map"):
        return verdict("inspect", "It needs the repository's own Python code, which Model Run never runs")
    if "model_index.json" in have:
        folders = {PurePosixPath(n).parent.as_posix() for n in names if PurePosixPath(n).name == "config.json" and "/" in n}
        bare = sorted(f for f in folders if not _weights([n for n in names if PurePosixPath(n).parent.as_posix() == f]))
        if bare:
            return verdict("inspect", f"{bare[0]} has no safetensors weights the loader reads")
        return verdict("runs_here", "A complete Diffusers pipeline", "diffusion")
    if "config.json" in have:
        if not _weights([n for n in names if "/" not in n]):
            return verdict("comfyui" if gguf else "inspect", "Its weights are GGUF, for ComfyUI's GGUF loaders" if gguf
                           else "It has no safetensors weights, only formats Model Run does not load")
        if "modules.json" in have:
            return verdict("runs_here", "A Sentence Transformers package", "embedding")
        task = task_for_config(config)
        if task == "text":
            causal = causal_types if causal_types is not None else _causal_types()
            if causal and config.get("model_type") not in causal:
                return verdict("inspect", f"Transformers here does not know the {config.get('model_type') or 'unnamed'} architecture")
            return verdict("runs_here", "A Transformers language model", "text")
        if task in ("image", "audio"):
            return verdict("runs_here", f"A Transformers {KINDS[task].lower()}", task)
        architecture = (config.get("architectures") or [config.get("model_type") or "This model"])[0]
        return verdict("inspect", f"{architecture} is not a kind of model this app runs yet")
    if gguf:
        return verdict("comfyui", "GGUF files, for ComfyUI's GGUF loaders")
    if loose:
        return verdict("comfyui", "Single weight files without a config, as ComfyUI uses them")
    return verdict("inspect", "No model files Model Run reads")


def _causal_types():
    from .media_models import _causal_lm_types
    return _causal_lm_types()


def package_files(info):
    """The files a package needs, each with its size and published hash, or ValueError saying why it cannot be one."""
    entries = {s["rfilename"]: s for s in info.get("siblings") or [] if isinstance(s, dict) and isinstance(s.get("rfilename"), str)}
    names = list(entries)
    diffusers = "model_index.json" in entries
    chosen = []
    by_folder = {}
    for name in names:
        path = PurePosixPath(name)
        parts = path.parts
        if any(p.startswith(".") or p.lower() in SKIP_FOLDERS for p in parts):
            continue
        folder = path.parent.as_posix()
        # Transformers reads the top folder; Sentence Transformers adds numbered steps; Diffusers has a folder per part.
        if folder != "." and not diffusers and not re.fullmatch(r"\d+_\w+", folder):
            continue
        by_folder.setdefault(folder, []).append(name)
    for folder, members in by_folder.items():
        # A pipeline keeps its weights in its part folders; a file at the top is a single-file copy for ComfyUI.
        keep = set() if diffusers and folder == "." else set(_weights(members))
        chosen += [n for n in members if n in keep or (n.lower().endswith(TEXT_SUFFIXES) and not n.lower().endswith(("readme.md",)))]
    files = []
    for name in chosen:
        entry = entries[name]
        lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
        size = lfs.get("size", entry.get("size"))
        if not isinstance(size, int):
            raise ValueError(f"Hugging Face did not report a size for {name}")
        if not name.lower().endswith(".safetensors") and size > MAX_TEXT_BYTES:
            raise ValueError(f"{name} is larger than a settings file should be")
        sha = lfs.get("sha256") if isinstance(lfs.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", lfs.get("sha256")) else None
        blob = entry.get("blobId") if not sha and isinstance(entry.get("blobId"), str) and re.fullmatch(r"[0-9a-f]{40}", entry.get("blobId")) else None
        files.append({"path": name, "size_bytes": size, "sha256": sha, "git_oid": blob})
    if not any(f["path"].lower().endswith(".safetensors") for f in files):
        raise ValueError("This repository has no safetensors weights Model Run loads")
    if len(files) > MAX_PACKAGE_FILES:
        raise ValueError("This repository has more files than one package should")
    # Weights first and the marker files last: the folder is listed only once everything has arrived.
    files.sort(key=lambda f: (PurePosixPath(f["path"]).name in MARKERS and "/" not in f["path"], not f["path"].lower().endswith(".safetensors"), f["path"]))
    return files


def check_file(name, path):
    """Refuse a downloaded file whose contents are not what its name says. Nothing is executed or unpickled."""
    lower = name.lower()
    if lower.endswith((".safetensors", ".sft")):
        inspect_file(path)
        return "safetensors"
    if lower.endswith(".gguf"):
        check_gguf(path)
        return "GGUF"
    if Path(path).stat().st_size > MAX_TEXT_BYTES:
        raise ValueError("too large for a settings file")
    data = Path(path).read_bytes()
    if lower.endswith(".json"):
        json.loads(data)
        return "JSON"
    if lower.endswith((".txt", ".jinja")):
        data.decode("utf-8")
        return "text"
    if lower.endswith(".model"):
        # A SentencePiece model is protocol buffers; a zip or pickle here would be something else entirely.
        if data[:4] == b"PK\x03\x04" or data[:1] == b"\x80":
            raise ValueError("not a SentencePiece model")
        return "SentencePiece"
    raise ValueError("not a file type Model Run downloads")
