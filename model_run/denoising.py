"""What the sampler is doing at each step: the latent, the predicted noise and the running estimate."""
import base64
import io
import logging
import math

MAX_STEPS = 32
PANEL_SIDE = 128
PUBLISH_EVERY = 4
LIMITATIONS = [
    "Images are a quick linear projection of the latent, or its first channels where the model has none. They are not VAE decodes.",
    "Predicted noise and the guidance push are scaled per image so they stay visible, so compare their strength with the numbers, not the brightness.",
    "Recorded after guidance is combined, one entry per denoiser step. Samplers that call the model several times per step have an entry per call.",
    "Only the first image in a batch, and the first frame of a video, is shown.",
]


def _first_item(tensor):
    """The image a batch starts with, as [channels, ...]; packed or nested streams give their first stream."""
    if hasattr(tensor, "tensors"):
        tensor = tensor.tensors[0]
    return tensor.detach().float()[:1]


def _rms(tensor):
    return float(tensor.pow(2).mean().sqrt().cpu())


def _clean(value):
    return value if math.isfinite(value) else None


class DenoisingRecorder:
    """A post-guidance function. It returns the denoised result untouched and reports nothing back to sampling."""
    def __init__(self, latent_format, *, on_update=None, max_steps=MAX_STEPS, side=PANEL_SIDE):
        self.latent_format = latent_format
        self.on_update = on_update
        self.max_steps = max_steps
        self.side = side
        self.previous = None
        self.seen = 0
        self.failed = False
        self.section = {"method": "sampler_post_cfg", "projection": "first_channels", "steps": [], "total_steps": None, "limitations": list(LIMITATIONS)}

    def __deepcopy__(self, memo):
        # ComfyUI clones model options; the record belongs to this run, not to each clone.
        memo[id(self)] = self
        return self

    def __call__(self, args):
        try:
            if not self.failed:
                self._record(args)
        except Exception as exc:  # noqa: BLE001 - observing must never break sampling
            self.failed = True
            self.section["skipped"] = f"{type(exc).__name__}: {exc}"[:300]
            logging.getLogger(__name__).warning("Model Run stopped recording denoising steps: %s", exc)
        return args["denoised"]

    def _stride(self, args):
        options = (args.get("model_options") or {}).get("transformer_options") or {}
        sigmas = options.get("sample_sigmas")
        if sigmas is None or not hasattr(sigmas, "numel") or sigmas.numel() < 2:
            return 1, None
        total = int(sigmas.numel()) - 1
        self.section["total_steps"] = total
        return max(1, math.ceil(total / self.max_steps)), total

    def _record(self, args):
        import torch
        call = self.seen
        self.seen += 1
        stride, total = self._stride(args)
        final = total is not None and call >= total - 1
        if call % stride and not final or len(self.section["steps"]) >= self.max_steps + 1:
            return
        with torch.no_grad():
            latent, estimate = _first_item(args["input"]), _first_item(args["denoised"])
            sigma_tensor = args.get("sigma")
            sigma = float(sigma_tensor.reshape(-1)[0].cpu()) if hasattr(sigma_tensor, "reshape") else float(sigma_tensor)
            noise = (latent - estimate) / sigma if sigma > 0 else None
            record = {"index": call, "sigma": _clean(sigma), "rms": {"latent": _clean(_rms(latent)), "estimate": _clean(_rms(estimate)), "noise": _clean(_rms(noise)) if noise is not None else None},
                      "estimate_change": _clean(_rms(estimate - self.previous)) if self.previous is not None and self.previous.shape == estimate.shape else None}
            self.previous = estimate
            images = {"latent": self._image(latent), "estimate": self._image(estimate)}
            if noise is not None:
                images["noise"] = self._image(noise, scaled=True)
            scale = args.get("cond_scale")
            cond, uncond = args.get("cond_denoised"), args.get("uncond_denoised")
            # At guidance 1 ComfyUI skips the negative prompt, so there is nothing to compare.
            if cond is not None and uncond is not None and scale is not None and not math.isclose(float(scale), 1.0):
                positive, negative = _first_item(cond), _first_item(uncond)
                push = (positive - negative) * float(scale)
                record["guidance_scale"] = float(scale)
                record["guidance_rms"] = _clean(_rms(push))
                record["guidance_cosine"] = _clean(float(torch.nn.functional.cosine_similarity(positive.reshape(1, -1), negative.reshape(1, -1)).cpu()))
                images.update(positive=self._image(positive), negative=self._image(negative), guidance=self._image(push, scaled=True))
            record["images"] = images
        self.section["steps"].append(record)
        if self.on_update is not None and (final or len(self.section["steps"]) % PUBLISH_EVERY == 1):
            self.on_update(self.section)

    def _project(self, tensor, scaled):
        """To [height, width, 3] in 0..1, the way ComfyUI's quick preview does it."""
        import torch
        format_ = self.latent_format
        factors = getattr(format_, "latent_rgb_factors", None)
        reshape = getattr(format_, "latent_rgb_factors_reshape", None)
        if reshape is not None:
            tensor = reshape(tensor)
        tensor = tensor[0, :, 0] if tensor.ndim == 5 else tensor[0]
        channels = int(tensor.shape[0])
        if factors is not None and len(factors) == channels:
            self.section["projection"] = "latent_rgb_factors"
            matrix = torch.tensor(factors, dtype=tensor.dtype, device=tensor.device)
            bias = getattr(format_, "latent_rgb_factors_bias", None)
            image = tensor.movedim(0, -1) @ matrix
            if bias is not None and not scaled:
                image = image + torch.tensor(bias, dtype=tensor.dtype, device=tensor.device)
            if not scaled:
                return ((image + 1) / 2).clamp(0, 1)
        else:
            picked = tensor[[min(i, channels - 1) for i in range(3)]]
            image = picked.movedim(0, -1)
            scaled = True
        spread = image.std().clamp_min(1e-6)
        return ((image - image.mean()) / (3 * spread) + 1).div(2).clamp(0, 1)

    def _image(self, tensor, *, scaled=False):
        from PIL import Image
        image = self._project(tensor, scaled)
        pixels = (image * 255).round().byte().cpu().numpy()
        picture = Image.fromarray(pixels)
        ratio = self.side / max(picture.size)
        picture = picture.resize((max(1, round(picture.width * ratio)), max(1, round(picture.height * ratio))), Image.BILINEAR)
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=80)
        return base64.b64encode(buffer.getvalue()).decode()
