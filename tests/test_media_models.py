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
        # Image-and-text models load as causal language models when Transformers maps their type that way.
        self.assertEqual(task_for_config({"model_type": "gemma3", "architectures": ["Gemma3ForConditionalGeneration"]}), "text")
        self.assertEqual(task_for_config({"model_type": "t5", "architectures": ["T5ForConditionalGeneration"]}), "inventory")
        self.assertEqual(task_for_config({"model_type": "bart", "is_encoder_decoder": True, "architectures": ["BartForConditionalGeneration"]}), "inventory")
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


class ImportValidationTests(unittest.TestCase):
    def audio(self):
        return {"method": "whisper_cross_attention", "seconds": 1.0, "time_bins": 2, "mel": {"bins": 1, "cols": 1, "values": [0.0]},
                "tokens": [{"step": 0, "token_id": 1, "probability": 0.5, "text": "a"}], "limitations": ["ok"],
                "layers": [{"layer": 0, "maps": [[0.1, 0.2]], "outside": [0.1]}]}

    def test_a_good_section_passes_and_wrong_types_are_refused(self):
        from model_run.media_models import validate_audio_view, validate_input_media
        validate_audio_view(self.audio())
        for change in (lambda v: v.update(limitations="not a list"), lambda v: v.update(attention_unavailable=3),
                       lambda v: v["tokens"][0].update(text=4), lambda v: v["tokens"][0].pop("step")):
            view = self.audio()
            change(view)
            with self.assertRaises(ValueError):
                validate_audio_view(view)
        validate_input_media({"kind": "image", "name": "x.png", "source_width": 1, "source_height": 2})
        for bad in ({"kind": "image", "name": "x", "source_width": "1", "source_height": 2}, {"kind": "other", "name": "x"}, "text",
                    {"kind": "audio", "name": 4, "source_rate": 1, "source_seconds": 1, "used_seconds": 1, "rate": 1}):
            with self.assertRaises(ValueError):
                validate_input_media(bad)

    def test_image_sections_refuse_wrong_descriptions(self):
        good = {"method": "class_token_attention_row", "entropy_bits": 1, "attention_layers": [], "top_labels": [{"index": 0, "label": "x", "probability": 1}]}
        validate_image_classification(good)
        for change in ({"limitations": "x"}, {"class_count": "9"}, {"attention_unavailable": 1}):
            with self.assertRaises(ValueError):
                validate_image_classification({**good, **change})

    def test_slices_refuse_wrong_statistics_and_descriptions(self):
        from model_run.tensor_slices import validate_slices
        record = SliceRecorder().record(torch.ones(1, 4, 8, 8))
        validate_slices(record)
        record["axes"][0]["mean_abs"][0] = "high"
        with self.assertRaises(ValueError):
            validate_slices(record)
        record = SliceRecorder().record(torch.ones(1, 4, 8, 8))
        record["pooled"] = "no"
        with self.assertRaises(ValueError):
            validate_slices(record)


class SequenceTests(unittest.TestCase):
    def test_a_long_sequence_keeps_only_the_kept_steps_and_stays_bounded(self):
        recorder = SliceRecorder(sequence=lambda length: (100, "time", "note") if length == 1500 else None)
        tensor = torch.zeros(1, 1500, 384)
        tensor[0, :100] = 1
        record = recorder.record(tensor)
        self.assertEqual((record["layout"], record["rows"], record["cols"], record["source_cols"], record["pooled"]), ("sequence", 16, 64, 100, True))
        self.assertEqual(set(record["values"]), {1.0})  # the padding beyond step 100 is not averaged in
        from model_run.tensor_slices import validate_slices
        validate_slices(record)
        self.assertIsNone(recorder.record(torch.zeros(1, 77, 8)))
        record["values"].pop()
        with self.assertRaises(ValueError):
            validate_slices(record)


class BatchedSequenceTests(unittest.TestCase):
    def test_guidance_batches_keep_a_map_per_entry(self):
        from model_run.tensor_slices import validate_slices
        recorder = SliceRecorder(sequence=lambda length: (length, "time", "note") if length == 9 else None)
        tensor = torch.ones(2, 9, 4)
        tensor[1] = 3
        record = recorder.record(tensor)
        validate_slices(record)
        self.assertEqual(record["axes"][0]["name"], "batch")
        self.assertEqual(len(record["values"]), 2 * 4 * 9)
        self.assertEqual((record["values"][0], record["values"][-1]), (1.0, 3.0))
        self.assertEqual(record["axes"][0]["mean_abs"], [1.0, 3.0])
        self.assertEqual(SliceRecorder(sequence=lambda _: (9, "time", None)).record(torch.ones(1, 9, 4))["axes"], [])


class LongAudioTests(unittest.TestCase):
    def test_only_what_is_used_is_decoded(self):
        import numpy as np
        import soundfile
        from unittest import mock
        buffer = io.BytesIO()
        soundfile.write(buffer, np.zeros(16000 * 120, dtype="float32"), 16000, format="FLAC")
        real = soundfile.SoundFile.read
        asked = []
        def spy(self, frames=-1, **kwargs):
            asked.append(frames)
            return real(self, frames=frames, **kwargs)
        with mock.patch.object(soundfile.SoundFile, "read", spy):
            samples, info = decode_audio(media(buffer.getvalue()))
        self.assertEqual(asked, [31 * 16000])
        self.assertEqual(len(samples), 30 * 16000)
        self.assertEqual((info["source_seconds"], info["used_seconds"], info["clipped"]), (120.0, 30.0, True))


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
    def test_english_only_models_run_without_a_task(self):
        model, processor = load_local_media_model(WHISPER, "audio")
        model.generation_config.is_multilingual = False
        model.generation_config.eos_token_id = [model.generation_config.eos_token_id]
        features = processor(torch.zeros(16000).numpy(), sampling_rate=16000, return_tensors="pt")["input_features"]
        _, trace = SpeechAdapter(model).transcribe({"input_features": features}, ["model.encoder.layers.0"], max_new_tokens=4, max_tensor_elements=8,
            audio_info={"used_seconds": 1.0, "hop_length": 160, "rate": 16000})
        generation = trace["transcript_generation"]
        self.assertIn(generation["stopped"], ("eos", "max_new_tokens"))
        self.assertIn("only transcribes English", " ".join(generation["limitations"]))

    def test_transcribe_records_steps_that_match_the_sequence(self):
        model, processor = load_local_media_model(WHISPER, "audio")
        features = processor(torch.zeros(16000).numpy(), sampling_rate=16000, return_tensors="pt")["input_features"]
        output, trace = SpeechAdapter(model).transcribe({"input_features": features}, ["model.encoder.layers.0", "model.decoder.layers.0"], max_new_tokens=8, max_tensor_elements=8,
            audio_info={"used_seconds": 1.0, "hop_length": 160, "rate": 16000})
        generation = trace.pop("transcript_generation")
        self.assertEqual([e["module"] for e in trace["events"]], ["model.encoder.layers.0", "model.decoder.layers.0"])
        self.assertEqual(trace["events"][0]["shape"], [1, 1500, 384])
        self.assertGreaterEqual(generation["input"]["token_count"], 1)
        self.assertEqual(generation["steps"][0]["position"], generation["input"]["token_count"])
        encode_trace(generation)
        view = trace["audio_view"]
        steps = generation["steps"]
        self.assertEqual(len(view["tokens"]), len(steps))
        self.assertEqual(view["mel"]["bins"], 80)
        self.assertEqual(view["encoder_length"], 1500)
        # One second of audio covers 50 encoder frames.
        self.assertEqual(view["encoder_frames"], 50)
        layer = view["layers"][-1]
        self.assertEqual(layer["layer"], 3)
        expected = output.cross_attentions[3][0].mean(0)[steps[0]["position"] - 1]
        self.assertAlmostEqual(sum(layer["maps"][0]), float(expected[:50].sum()), places=2)
        self.assertAlmostEqual(layer["outside"][0], float(expected[50:].sum()), places=3)
        self.assertEqual(trace["events"][0]["slices"]["layout"], "sequence")
        self.assertEqual((trace["events"][0]["slices"]["col_axis"], trace["events"][0]["slices"]["cols"]), ("time", 50))
        self.assertEqual(trace["events"][1]["slices"]["col_axis"], "token")
        encode_trace(trace)
        view["layers"][-1]["maps"][0].pop()
        with self.assertRaises(ValueError):
            encode_trace(trace)


if __name__ == "__main__":
    unittest.main()
