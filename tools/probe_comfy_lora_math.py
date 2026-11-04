"""Manual numerical native LoRA application check for every ordinary 2D pair."""
import argparse
import json
import sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("comfy_root", type=Path)
parser.add_argument("artifact", type=Path)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(args.comfy_root))
import torch
import comfy.lora
import comfy.utils
state = comfy.utils.load_torch_file(str(args.artifact), safe_load=True)
stems = sorted(key[:-len(".lora_up.weight")] for key in state if key.endswith(".lora_up.weight"))
if not stems:
    raise ValueError("No ordinary LoRA pairs found")
mapping = {stem: stem + ".weight" for stem in stems}
diff_stems = sorted(key[:-5] for key in state if key.endswith(".diff"))
mapping.update({stem: stem + ".weight" for stem in diff_stems})
patches = comfy.lora.load_lora(state, mapping)
checked = 0
for stem in stems:
    up = state[stem + ".lora_up.weight"].to(device="cuda", dtype=torch.float32)
    down = state[stem + ".lora_down.weight"].to(device="cuda", dtype=torch.float32)
    if up.ndim != 2 or down.ndim != 2 or up.shape[1] != down.shape[0]:
        raise ValueError(f"Unsupported pair shape: {stem}")
    alpha = state.get(stem + ".alpha")
    scale = float(alpha.item()) / down.shape[0] if alpha is not None else 1.0
    key = stem + ".weight"
    base = torch.full((up.shape[0], down.shape[1]), 0.01, device="cuda")
    with torch.inference_mode():
        reference = base + (up @ down) * (0.7 * scale)
        observed = comfy.lora.calculate_weight([(0.7, patches[key], 1.0, None, None)], base.clone(), key)
    torch.testing.assert_close(observed, reference, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(observed).all().item()
    checked += 1
    del base, reference, observed, up, down
for stem in diff_stems:
    delta = state[stem + ".diff"].to(device="cuda", dtype=torch.float32)
    base = torch.full_like(delta, 0.01)
    key = stem + ".weight"
    observed = comfy.lora.calculate_weight([(0.7, patches[key], 1.0, None, None)], base.clone(), key)
    torch.testing.assert_close(observed, base + 0.7 * delta, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(observed).all().item()
print(json.dumps({"artifact": args.artifact.name, "status": "passed", "stage": "all_2d_lora_pair_numeric_application",
                  "pairs_checked": checked, "direct_diffs_checked": len(diff_stems), "device": "cuda", "dtype": "float32", "strength": 0.7,
                  "limitations": ["Compared with synthetic base weights; quantized base weight repacking and generated output are not covered."]}))
