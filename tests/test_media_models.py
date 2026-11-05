import base64
import io
import tempfile
import unittest
from pathlib import Path

import torch

from model_run.media_models import (ImageClassifierAdapter, SpeechAdapter, declared_name, decode_audio, decode_image,
                                    load_local_media_model, task_for_config, validate_image_classification)
from model_run.tensor_slices import SliceRecorder
from model_run.trace_store import encode_trace

WHISPER = Path(__file__).parent.parent / ".models" / "compat-1"


def media(data):
    return {"base64": base64.b64encode(data).decode()}


def tiny_vit():
    from transformers import ViTConfig, ViTForImageClassification
    torch.manual_seed(0)
    config = ViTConfig(image_size=32, patch_size=8, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                       num_labels=3, id2label={0: "cat", 1: "dog", 2: "cube"}, label2id={"cat": 0, "dog": 1, "cube": 2}, attn_implementation="eager")
    return ViTForImageClassification(config).eval()


class TaskTests(unittest.TestCase):
    def test_tasks_come_from_the_declared_architecture(self):
        self.assertEqual(task_for_config({"architectures": ["LlamaForCausalLM"]}), "text")
        self.assertEqual(task_for_config({"architectures": ["ViTForImageClassification"]}), "image")
        self.assertEqual(task_for_config({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}), "audio")
        self.assertEqual(task_for_config({"architectures": ["SomethingElse"]}), "inventory")
        self.assertEqual(task_for_config(None), "inventory")

    def test_names_come_from_the_config_only_when_published(self):
        self.assertEqual(declared_name({"_name_or_path": "openai/whisper-tiny"}, "compat-1"), "whisper-tiny")
        self.assertEqual(declared_name({"_name_or_path": "C:/private/models/x"}, "folder"), "folder")
        self.assertEqual(declared_name({}, "folder"), "folder")


class DecodeTests(unittest.TestCase):
    def test_image_formats_are_restricted(self):
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGBA", (4, 3)).save(buffer, format="PNG")
        self.assertEqual(decode_image(media(buffer.getvalue())).mode, "RGB")
        gif = io.BytesIO()
        Image.new("P", (4, 3)).save(gif, format="GIF")
        with self.assertRaises(ValueError):
            decode_image(media(gif.getvalue()))
        with self.assertRaises(ValueError):
            decode_image(media(b"not an image"))
        with self.assertRaises(ValueError):
            decode_image({"base64": "!!!"})
        with self.assertRaises(ValueError):
            decode_image(None)

    def test_audio_is_resampled_mono_and_clipped_with_a_record(self):
        import numpy as np
        import soundfile
        buffer = io.BytesIO()
        rate = 8000
        soundfile.write(buffer, np.zeros((rate * 31, 2), dtype="float32"), rate, format="WAV")
        samples, info = decode_audio(media(buffer.getvalue()))
        self.assertEqual(samples.ndim, 1)
        self.assertEqual(len(samples), 30 * 16000)
        self.assertEqual((info["source_rate"], info["source_channels"], info["rate"], info["clipped"], info["used_seconds"]), (8000, 2, 16000, True, 30.0))
        self.assertEqual(info["source_seconds"], 31.0)
        with self.assertRaises(ValueError):
            decode_audio(media(b"not audio"))


class ImageAdapterTests(unittest.TestCase):
    def test_classify_measures_labels_attention_and_patch_slices(self):
        adapter = ImageClassifierAdapter(tiny_vit())
        self.assertEqual(adapter.grid, (4, 4))
        output, trace = adapter.classify({"pixel_values": torch.zeros(1, 3, 32, 32)}, ["vit.layers.0", "vit.layers.1"], max_tensor_elements=8)
        record = trace["image_classification"]
        expected = torch.softmax(output.logits[0], dim=-1)
        self.assertAlmostEqual(sum(e["probability"] for e in record["top_labels"]), 1.0, places=5)
        self.assertAlmostEqual(record["top_labels"][0]["probability"], float(expected.max()), places=5)
        self.assertEqual(record["top_labels"][0]["label"], ["cat", "dog", "cube"][int(expected.argmax())])
        self.assertEqual([entry["layer"] for entry in record["attention_layers"]], [0, 1])
        weights = output.attentions[1][0].mean(0)[0]
        self.assertAlmostEqual(record["attention_layers"][1]["patches"][0], float(weights[1]), places=3)
        self.assertAlmostEqual(record["attention_layers"][1]["self"] + sum(record["attention_layers"][1]["patches"]), 1.0, places=2)
        slices = trace["events"][0]["slices"]
        self.assertEqual(slices["layout"], "patch_tokens")
        self.assertEqual((slices["rows"], slices["cols"], slices["axes"][0]["name"]), (4, 4, "feature"))
        self.assertIn("class token", slices["assumption"])
        encode_trace(trace)
        record["attention_layers"][0]["patches"].pop()
        with self.assertRaises(ValueError):
            encode_trace(trace)

    def test_section_validator_rejects_bad_probability(self):
        with self.assertRaises(ValueError):
            validate_image_classification({"method": "class_token_attention_row", "entropy_bits": 1, "attention_layers": [],
                                           "top_labels": [{"index": 0, "label": "x", "probability": 2}]})

    def test_patch_tokens_need_the_exact_count(self):
        recorder = SliceRecorder(token_grid=(1, 2, 2))
        self.assertEqual(recorder.record(torch.ones(1, 5, 3))["layout"], "patch_tokens")
        self.assertIsNone(SliceRecorder(token_grid=(1, 2, 2)).record(torch.ones(1, 6, 3)))

    def test_local_loader_reads_a_saved_package(self):
        from transformers import ViTImageProcessor
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            tiny_vit().save_pretrained(directory, safe_serialization=True)
            ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(directory)
            model, processor = load_local_media_model(directory, "image")
            self.assertFalse(model.training)
            self.assertEqual(processor.size["height"], 32)


@unittest.skipUnless((WHISPER / "model.safetensors").is_file(), "Local Whisper package not present")
class SpeechAdapterTests(unittest.TestCase):
    def test_transcribe_records_steps_that_match_the_sequence(self):
        model, processor = load_local_media_model(WHISPER, "audio")
        features = processor(torch.zeros(16000).numpy(), sampling_rate=16000, return_tensors="pt")["input_features"]
        output, trace = SpeechAdapter(model).transcribe({"input_features": features}, ["model.encoder.layers.0", "model.decoder.layers.0"], max_new_tokens=8, max_tensor_elements=8)
        generation = trace.pop("transcript_generation")
        self.assertEqual([e["module"] for e in trace["events"]], ["model.encoder.layers.0", "model.decoder.layers.0"])
        self.assertEqual(trace["events"][0]["shape"], [1, 1500, 384])
        self.assertGreaterEqual(generation["input"]["token_count"], 1)
        self.assertEqual(generation["steps"][0]["position"], generation["input"]["token_count"])
        encode_trace(generation)
        encode_trace(trace)


if __name__ == "__main__":
    unittest.main()
