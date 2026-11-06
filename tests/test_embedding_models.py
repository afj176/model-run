import json
import tempfile
import unittest
from pathlib import Path

import torch

from model_run.embedding_models import EmbeddingAdapter, is_embedding_package, load_local_embedding_model
from model_run.trace_store import encode_trace


def tiny_package(root):
    from safetensors.torch import save_file
    from transformers import BertConfig, BertModel, BertTokenizerFast
    torch.manual_seed(0)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "red", "cube", "on", "table"]
    (root / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    BertTokenizerFast(vocab_file=str(root / "vocab.txt")).save_pretrained(str(root))
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=2, num_attention_heads=2, intermediate_size=16)).save_pretrained(str(root))
    (root / "modules.json").write_text(json.dumps([
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
        {"idx": 3, "name": "3", "path": "3_Normalize", "type": "sentence_transformers.models.Normalize"}]), encoding="utf-8")
    (root / "1_Pooling").mkdir()
    (root / "1_Pooling" / "config.json").write_text(json.dumps({"word_embedding_dimension": 8, "pooling_mode_mean_tokens": True}), encoding="utf-8")
    (root / "2_Dense").mkdir()
    (root / "2_Dense" / "config.json").write_text(json.dumps({"in_features": 8, "out_features": 4, "bias": False,
                                                               "activation_function": "torch.nn.modules.linear.Identity"}), encoding="utf-8")
    dense = torch.randn(4, 8)
    save_file({"linear.weight": dense}, str(root / "2_Dense" / "model.safetensors"))
    return dense


class EmbeddingModelTests(unittest.TestCase):
    def test_the_package_defines_the_vector(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dense = tiny_package(root)
            self.assertTrue(is_embedding_package(root))
            self.assertFalse(is_embedding_package(root / "2_Dense"))
            model, tokenizer, head = load_local_embedding_model(root)
            self.assertEqual((head["pooling"], len(head["dense"]), head["normalize"]), ("mean_tokens", 1, True))
            encoded = tokenizer("a red cube", return_tensors="pt")
            adapter = EmbeddingAdapter(model, head)
            output, trace = adapter.embed(dict(encoded), ["encoder.layer.0", "encoder.layer.1"])
            with torch.no_grad():
                states = model(**encoded).last_hidden_state[0]
            expected = dense @ states.mean(0)
            expected = expected / expected.norm()
            record = trace["embedding"]
            self.assertEqual((record["dimension"], record["token_count"]), (4, encoded["input_ids"].shape[1]))
            self.assertTrue(torch.allclose(torch.tensor(record["values"]), expected, atol=1e-5))
            self.assertEqual(len(record["token_alignment"]), record["token_count"])
            self.assertEqual(len(trace["events"]), 2)
            encode_trace(trace)

    def test_unsupported_steps_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tiny_package(root)
            config = root / "2_Dense" / "config.json"
            config.write_text(json.dumps({**json.loads(config.read_text()), "activation_function": "torch.nn.modules.activation.Tanh"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Tanh"):
                load_local_embedding_model(root)
