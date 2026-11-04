"""Read explicit ComfyUI extra-model-path configuration without loading ComfyUI."""
import os
from pathlib import Path

MAX_CONFIG_BYTES = 1024 * 1024

# Snapshot of common ComfyUI model categories. It is not a live host registry.
STANDARD_MODEL_DIRS = {
    "checkpoints": ("checkpoints",), "configs": ("configs",), "loras": ("loras",),
    "vae": ("vae",), "text_encoders": ("text_encoders", "clip"),
    "diffusion_models": ("unet", "diffusion_models"), "clip_vision": ("clip_vision",),
    "style_models": ("style_models",), "embeddings": ("embeddings",), "diffusers": ("diffusers",),
    "vae_approx": ("vae_approx",), "controlnet": ("controlnet", "t2i_adapter"),
    "gligen": ("gligen",), "upscale_models": ("upscale_models",),
    "latent_upscale_models": ("latent_upscale_models",), "hypernetworks": ("hypernetworks",),
    "photomaker": ("photomaker",), "classifiers": ("classifiers",),
    "model_patches": ("model_patches",), "audio_encoders": ("audio_encoders",),
    "background_removal": ("background_removal",), "frame_interpolation": ("frame_interpolation",),
    "geometry_estimation": ("geometry_estimation",), "optical_flow": ("optical_flow",),
    "detection": ("detection",),
}


def standard_model_paths(models_dir):
    """Resolve common category locations from an explicitly supplied model root."""
    root = Path(models_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("models_dir must be a directory")
    entries = []
    for category, directories in STANDARD_MODEL_DIRS.items():
        for directory in directories:
            location = root / directory
            entries.append({"section": "standard", "category": category, "path": str(location),
                            "exists": location.is_dir(), "default": True})
    return {"schema_version": "1.0", "kind": "comfy_standard_model_paths", "source": str(root),
            "entries": entries, "limitations": ["Categories are a static compatibility snapshot, not the live ComfyUI registry.",
            "The caller supplies the actual model root; custom and runtime-added paths are not discovered."]}


def read_extra_paths(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("ComfyUI path configuration inspection requires PyYAML") from exc
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("Configuration exceeds size limit")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeError) as exc:
        raise ValueError(f"Invalid ComfyUI path configuration: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    entries = []
    for name, section in config.items():
        if section is None:
            continue
        if not isinstance(name, str) or not isinstance(section, dict):
            raise ValueError("Configuration sections must be named mappings")
        base = section.get("base_path")
        if base is not None and not isinstance(base, str):
            raise ValueError("base_path must be a string")
        if base is not None:
            base = os.path.expandvars(os.path.expanduser(base))
            base = os.path.abspath(os.path.join(path.parent, base))
        for category, value in section.items():
            if category in ("base_path", "is_default"):
                continue
            if not isinstance(category, str) or not isinstance(value, str):
                raise ValueError("Model categories must contain path strings")
            if category == "custom_nodes":
                continue
            for item in value.split("\n"):
                if not item:
                    continue
                resolved = os.path.normpath(os.path.join(base or str(path.parent), item))
                entries.append({"section": name, "category": category, "path": resolved,
                                "exists": os.path.isdir(resolved), "default": bool(section.get("is_default", False))})
    return {"schema_version": "1.0", "source": str(path), "kind": "comfy_extra_model_paths",
            "entries": entries, "limitations": ["Only the supplied extra-path file is inspected; built-in and runtime-added paths are not included.",
                                                    "Paths are configuration data; model files are not scanned or loaded."]}
