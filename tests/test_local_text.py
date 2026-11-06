import json
import tempfile
import unittest
from pathlib import Path

import torch

from model_run.local_text import fitting_dtype


class FittingDtypeTests(unittest.TestCase):
    def package(self, root, dtype, weight_bytes):
        (root / "config.json").write_text(json.dumps({"torch_dtype": dtype}), encoding="utf-8")
        with open(root / "model.safetensors", "wb") as handle:
            handle.truncate(weight_bytes)

    def test_float32_when_it_fits_or_off_the_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.package(root, "bfloat16", 1000)
            self.assertIs(fitting_dtype(root, "cuda", free_bytes=10_000), torch.float32)
            self.assertIs(fitting_dtype(root, "cpu", free_bytes=10), torch.float32)

    def test_a_16_bit_checkpoint_too_big_for_float32_keeps_its_own_type(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.package(root, "bfloat16", 1000)
            self.assertIs(fitting_dtype(root, "cuda", free_bytes=1500), torch.bfloat16)
            self.package(root, "float16", 1000)
            self.assertIs(fitting_dtype(root, "cuda", free_bytes=1500), torch.float16)

    def test_float32_checkpoints_are_never_narrowed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.package(root, "float32", 1000)
            self.assertIs(fitting_dtype(root, "cuda", free_bytes=100), torch.float32)


class PromptTokenTests(unittest.TestCase):
    def test_a_missing_start_token_is_added_once(self):
        from model_run.application import with_start_token
        gemma = type("Tok", (), {"bos_token_id": 2, "eos_token_id": [1, 106]})()
        ids = torch.tensor([[818, 5279]])
        self.assertEqual(with_start_token(gemma, ids).tolist(), [[2, 818, 5279]])
        self.assertEqual(with_start_token(gemma, torch.tensor([[2, 818]])).tolist(), [[2, 818]])

    def test_tokenizers_without_a_distinct_start_token_are_unchanged(self):
        from model_run.application import with_start_token
        ids = torch.tensor([[464, 3139]])
        qwen = type("Tok", (), {"bos_token_id": None, "eos_token_id": 151645})()
        gpt2 = type("Tok", (), {"bos_token_id": 50256, "eos_token_id": 50256})()
        self.assertEqual(with_start_token(qwen, ids).tolist(), ids.tolist())
        self.assertEqual(with_start_token(gpt2, ids).tolist(), ids.tolist())


class TextLayerPathTests(unittest.TestCase):
    def test_language_layers_come_before_other_towers(self):
        from model_run.application import text_layer_paths
        class Tower(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
        class Multimodal(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower, self.language_model = Tower(), Tower()
            def get_decoder(self):
                return self.language_model
        model = Multimodal()
        tree = {"nodes": [{"path": name} for name, _ in model.named_modules() if name]}
        self.assertEqual(text_layer_paths(model, tree), ["language_model.layers.0", "language_model.layers.1", "language_model.layers.2"])
        del Multimodal.get_decoder
        self.assertEqual(text_layer_paths(model, tree, limit=2), ["vision_tower.layers.0", "vision_tower.layers.1"])
