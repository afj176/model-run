import unittest

try:
    import torch
except ImportError:
    torch = None

from model_run.cross_attention import image_grid, pooled_grid, measure, stable_audio_labels, text_key_range, MAX_MAP_VALUES


class GridTests(unittest.TestCase):
    def test_image_grid_and_pooling(self):
        self.assertEqual(image_grid(16, [1, 4, 4, 4]), (4, 4, 1, 1))
        self.assertEqual(image_grid(4, [1, 4, 4, 4]), (2, 2, 2, 1))
        self.assertEqual(image_grid(1024, [1, 16, 64, 64]), (32, 32, 2, 1))
        self.assertEqual(image_grid(3 * 16, [1, 16, 3, 8, 8]), (4, 4, 2, 3))  # video: 3 frames of 4×4 tokens
        self.assertEqual(image_grid(2 * 16, [1, 16, 4, 8, 8]), (4, 4, 2, 2))  # temporal patch 2
        self.assertIsNone(image_grid(17, [1, 4, 4, 4]))
        self.assertIsNone(image_grid(16, [1, 4, 16]))
        self.assertEqual(pooled_grid(64, 64, 77, 32), (32, 32))
        self.assertEqual(pooled_grid(16, 8, 10, 32), (16, 8))
        rows, cols = pooled_grid(64, 64, 512, 64)
        self.assertLessEqual(512 * rows * cols, MAX_MAP_VALUES)


@unittest.skipIf(torch is None, "optional torch dependency unavailable")
class MeasureTests(unittest.TestCase):
    def test_cross_attention_matches_softmax(self):
        torch.manual_seed(0)
        q = torch.randn(2, 2, 16, 8)
        k = torch.randn(2, 2, 5, 8)
        result = measure(q, k, heads=2, text_tokens=5, latent_shape=[2, 4, 4, 4], batch_index=1)
        self.assertEqual(result["attention_kind"], "cross")
        self.assertEqual(result["grid"], [4, 4])
        expected = torch.softmax((q[1] @ k[1].transpose(1, 2)) * 8 ** -0.5, dim=-1).mean(0)  # [16, 5]
        torch.testing.assert_close(torch.tensor(result["maps"]), expected.transpose(0, 1), atol=1e-4, rtol=0)
        torch.testing.assert_close(torch.tensor(result["text_mass"]), torch.ones(16), atol=1e-3, rtol=0)
        self.assertEqual(len(result["token_mass"]), 5)
        # The flat [B, L, heads*D] layout gives the same answer.
        flat = measure(q.transpose(1, 2).reshape(2, 16, 16), k.transpose(1, 2).reshape(2, 5, 16), heads=2, text_tokens=5, latent_shape=[2, 4, 4, 4], batch_index=1)
        torch.testing.assert_close(torch.tensor(flat["maps"]), torch.tensor(result["maps"]), atol=1e-5, rtol=0)

    def test_joint_attention_mask_and_pooling(self):
        torch.manual_seed(1)
        q = torch.randn(1, 3, 5 + 64, 4)
        k = torch.randn(1, 3, 5 + 64, 4)
        mask = torch.ones(1, 1, 69, dtype=torch.bool)
        mask[..., 4] = False  # a padded text token
        result = measure(q, k, heads=3, text_tokens=5, latent_shape=[1, 4, 8, 8], mask=mask, max_grid=4)
        self.assertEqual(result["attention_kind"], "joint")
        self.assertEqual(result["source_grid"], [8, 8])
        self.assertEqual(result["grid"], [4, 4])
        self.assertTrue(result["mask_applied"])
        self.assertEqual(result["maps"][4], [0.0] * 16)
        scores = (q[0, :, 5:] @ k[0].transpose(1, 2)) * 0.5
        scores[..., 4] = float("-inf")
        expected = torch.softmax(scores, dim=-1).mean(0)[:, :5].transpose(0, 1).reshape(5, 1, 8, 8)
        expected = torch.nn.functional.adaptive_avg_pool2d(expected, (4, 4)).reshape(5, 16)
        torch.testing.assert_close(torch.tensor(result["maps"]), expected, atol=1e-4, rtol=0)
        self.assertTrue(all(0 <= v <= 1 for v in result["text_mass"]))

    def test_grouped_query_keys_are_expanded(self):
        torch.manual_seed(2)
        q = torch.randn(1, 4, 16, 8)
        k = torch.randn(1, 2, 5, 8)
        result = measure(q, k, heads=4, text_tokens=5, latent_shape=[1, 4, 4, 4])
        expected = measure(q, k.repeat_interleave(2, dim=1), heads=4, text_tokens=5, latent_shape=[1, 4, 4, 4])
        self.assertEqual(result["maps"], expected["maps"])
        flat = measure(q.transpose(1, 2).reshape(1, 16, 32), k.transpose(1, 2).reshape(1, 5, 16), heads=4, text_tokens=5, latent_shape=[1, 4, 4, 4])
        self.assertEqual(flat["maps"], expected["maps"])
        self.assertIn("unavailable", measure(q, torch.randn(1, 3, 5, 8), heads=4, text_tokens=5, latent_shape=[1, 4, 4, 4]))

    def test_image_queries_over_text_and_image_keys(self):
        torch.manual_seed(5)
        q = torch.randn(1, 2, 16, 8)
        k = torch.randn(1, 2, 5 + 16, 8)
        result = measure(q, k, heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4])
        self.assertEqual((result["attention_kind"], result["image_tokens"]), ("joint", 16))
        expected = torch.softmax((q[0] @ k[0].transpose(1, 2)) * 8 ** -0.5, dim=-1).mean(0)[:, :5].transpose(0, 1)
        torch.testing.assert_close(torch.tensor(result["maps"]), expected, atol=1e-4, rtol=0)

    def test_queries_with_a_text_token_before_the_image(self):
        # HiDream-O1: 16 causal text tokens run apart, so queries are the last text token and the image.
        torch.manual_seed(6)
        q = torch.randn(1, 2, 1 + 16, 8)
        k = torch.randn(1, 2, 5 + 16, 8)
        result = measure(q, k, heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4])
        self.assertEqual((result["attention_kind"], result["image_tokens"]), ("joint", 16))
        self.assertTrue(result["text_spreads"])
        expected = torch.softmax((q[0, :, 1:] @ k[0].transpose(1, 2)) * 8 ** -0.5, dim=-1).mean(0)[:, :5].transpose(0, 1)
        torch.testing.assert_close(torch.tensor(result["maps"]), expected, atol=1e-4, rtol=0)

    def test_only_block_causal_layouts_warn_that_meaning_spreads(self):
        torch.manual_seed(8)
        joint = measure(torch.randn(1, 2, 5 + 16, 8), torch.randn(1, 2, 5 + 16, 8), heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4])
        self.assertEqual(joint["attention_kind"], "joint")
        self.assertNotIn("text_spreads", joint)
        qwen = measure(torch.randn(1, 2, 16, 8), torch.randn(1, 2, 5 + 16, 8), heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4])
        self.assertTrue(qwen["text_spreads"])

    def test_text_can_sit_after_the_image_or_after_registers(self):
        torch.manual_seed(9)
        q = torch.randn(1, 2, 16 + 5, 8)
        k = torch.randn(1, 2, 16 + 5, 8)
        last = measure(q, k, heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4], text_start=-1)
        self.assertEqual((last["attention_kind"], last["image_tokens"]), ("joint", 16))
        expected = torch.softmax((q[0, :, :16] @ k[0].transpose(1, 2)) * 8 ** -0.5, dim=-1).mean(0)[:, 16:21].transpose(0, 1)
        torch.testing.assert_close(torch.tensor(last["maps"]), expected, atol=1e-4, rtol=0)
        q = torch.randn(1, 2, 3 + 5 + 16, 8)
        k = torch.randn(1, 2, 3 + 5 + 16, 8)
        lead = measure(q, k, heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4], text_start=3)
        expected = torch.softmax((q[0, :, 8:] @ k[0].transpose(1, 2)) * 8 ** -0.5, dim=-1).mean(0)[:, 3:8].transpose(0, 1)
        torch.testing.assert_close(torch.tensor(lead["maps"]), expected, atol=1e-4, rtol=0)

    def test_masks_follow_the_text_wherever_it_sits(self):
        from model_run.cross_attention import text_key_range
        self.assertEqual(text_key_range(21, 21, 5, allow_joint=True, text_start=-1), (16, 21))

    def test_keys_can_run_image_text_then_other_conditioning(self):
        # HiDream I1 and Stable Cascade: 16 image, 5 text, 3 other, and only the text is read.
        keys = torch.zeros(1, 2, 24, 8)
        keys[:, :, 16:21] = 1.0
        hidream = measure(keys, keys, heads=2, text_tokens=5, latent_shape=[1, 4, 8, 8], image_first=True, known_image_tokens=16)
        self.assertEqual((hidream["attention_kind"], hidream["source_grid"]), ("joint", [4, 4]))
        self.assertEqual(hidream["image_tokens"], 16)
        # Stable Cascade's queries are the image alone.
        cascade = measure(torch.zeros(1, 2, 16, 8), keys, heads=2, text_tokens=5, latent_shape=[1, 4, 8, 8], image_first=True)
        self.assertEqual(cascade["source_grid"], [4, 4])
        self.assertAlmostEqual(sum(cascade["token_mass"]), 5 / 24, places=4)

    def test_stable_audio_labels_name_its_two_number_embeddings(self):
        self.assertEqual(stable_audio_labels(["a", "</s>"], 4), ["a", "</s>", "<seconds start>", "<seconds total>"])
        self.assertIsNone(stable_audio_labels(["a", "</s>"], 3))

    def test_audio_queries_end_with_time_steps(self):
        # ACE-Step: 8 time steps, keys are a speaker token, 5 words and 2 lyric tokens.
        found = measure(torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8)[:, :, :8], heads=2, text_tokens=5, latent_shape=[1, 8, 16, 8], stream="audio", text_start=1, known_image_tokens=8)
        self.assertIn("unavailable", found)  # equal lengths are self-attention
        found = measure(torch.randn(1, 2, 9, 8), torch.randn(1, 2, 8, 8), heads=2, text_tokens=5, latent_shape=[1, 8, 16, 8], stream="audio", text_start=1, known_image_tokens=8)
        self.assertEqual((found["attention_kind"], found["source_grid"], found["stream"]), ("cross", [1, 8], "audio"))

    def test_summary_tokens_before_the_image_are_not_mapped(self):
        # Fibo 2: 16 summary tokens that happen to fit the grid must not be read as the image.
        q, k = torch.randn(1, 2, 16, 8), torch.randn(1, 2, 5, 8)
        self.assertIn("summary tokens", measure(q, k, heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4], known_image_tokens=16, query_lead=16)["unavailable"])
        same = torch.randn(1, 2, 32, 8)
        self.assertIn("no text keys", measure(same, same, heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4], known_image_tokens=16, query_lead=16)["unavailable"])
        joined = torch.randn(1, 2, 32, 8)
        found = measure(joined, k, heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4], known_image_tokens=16, query_lead=16)
        self.assertEqual((found["attention_kind"], found["source_grid"]), ("cross", [4, 4]))
        self.assertEqual(text_key_range(24, 24, 5, allow_joint=True, text_start=3), (3, 8))
        self.assertEqual(text_key_range(21, 21, 5, allow_joint=True), (0, 5))

    def test_geometry_identifies_joint_attention_but_not_self_attention(self):
        from model_run.cross_attention import joint_by_geometry
        self.assertTrue(joint_by_geometry(16 + 5, 16 + 5, 5, [1, 4, 4, 4]))
        self.assertFalse(joint_by_geometry(16, 16, 5, [1, 4, 4, 4]))  # self-attention on the image alone
        self.assertFalse(joint_by_geometry(16 + 5, 16, 5, [1, 4, 4, 4]))
        self.assertTrue(joint_by_geometry(3 + 5 + 16, 3 + 5 + 16, 5, [1, 4, 4, 4], extra=3))
        # A count that fits an image by itself stays self-attention, even if text would also fit.
        self.assertFalse(joint_by_geometry(64, 64, 48, [1, 4, 8, 8]))

    def test_reference_image_tokens_are_left_out_of_the_map(self):
        torch.manual_seed(10)
        q = torch.randn(1, 2, 5 + 16 + 4, 8)
        k = torch.randn(1, 2, 5 + 16 + 4, 8)
        result = measure(q, k, heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4], reference_tokens=4)
        self.assertEqual((result["image_tokens"], result["grid"]), (16, [4, 4]))
        expected = torch.softmax((q[0, :, 5:21] @ k[0].transpose(1, 2)) * 8 ** -0.5, dim=-1).mean(0)[:, :5].transpose(0, 1)
        torch.testing.assert_close(torch.tensor(result["maps"]), expected, atol=1e-4, rtol=0)
        self.assertIn("unavailable", measure(q, k, heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4]))

    def test_an_audio_stream_is_a_strip_over_time(self):
        torch.manual_seed(7)
        q = torch.randn(1, 2, 9, 8)
        k = torch.randn(1, 2, 5, 8)
        result = measure(q, k, heads=2, text_tokens=5, latent_shape=[1, 4, 2, 8, 8], stream="audio")
        self.assertEqual((result["stream"], result["grid"], result["frames"], result["attention_kind"]), ("audio", [1, 9], 1, "cross"))
        expected = torch.softmax((q[0] @ k[0].transpose(1, 2)) * 8 ** -0.5, dim=-1).mean(0).transpose(0, 1)
        torch.testing.assert_close(torch.tensor(result["maps"]), expected, atol=1e-4, rtol=0)
        self.assertIn("unavailable", measure(q, k, heads=2, text_tokens=5, latent_shape=[1, 4, 2, 8, 8]))

    def test_unsupported_geometry_is_reported(self):
        q = torch.randn(1, 2, 17, 8)
        k = torch.randn(1, 2, 5, 8)
        self.assertIn("unavailable", measure(q, k, heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4]))
        self.assertIn("unavailable", measure(q, k, heads=3, text_tokens=5, latent_shape=[1, 4, 4, 4]))
        self.assertIn("unavailable", measure(q, q, heads=2, text_tokens=20, latent_shape=[1, 4, 4, 4]))
        with self.assertRaises(ValueError): measure(q, k, heads=2, text_tokens=5, latent_shape=[1, 4, 4, 4], max_grid=0)


@unittest.skipIf(torch is None, "optional torch dependency unavailable")
class VideoMapTests(unittest.TestCase):
    def test_video_latents_record_per_frame_maps(self):
        torch.manual_seed(3)
        q = torch.randn(1, 2, 3 * 16, 8)
        k = torch.randn(1, 2, 5, 8)
        result = measure(q, k, heads=2, text_tokens=5, latent_shape=[1, 16, 3, 8, 8], max_grid=4)
        self.assertEqual((result["frames"], result["grid"], result["source_grid"]), (3, [4, 4], [4, 4]))
        self.assertEqual(len(result["maps"][0]), 3 * 16)
        self.assertEqual(result["frame_indices"], [0, 1, 2])
        from model_run.trace_store import _validate_cross_attention
        _validate_cross_attention({"method": "measured_attention_probabilities", "source": "fixture", "selection": "x", "grid_limit": 4, "limitations": [],
                                   "events": [{"denoiser_call": 0, "block": "x", "batch": "b", **result}]})


class LabelAlignmentTests(unittest.TestCase):
    def test_padding_with_an_ordinary_token_is_collapsed(self):
        from model_run.cross_attention import collapse_padding
        self.assertEqual(collapse_padding(["<|startoftext|>", "a", "<|endoftext|>", "!", "!"]), ["<|startoftext|>", "a", "<|endoftext|>", "<|endoftext|>", "<|endoftext|>"])
        self.assertEqual(collapse_padding(["a", "!", "!"]), ["a", "!", "!"])  # no end token: these may be words
        self.assertEqual(collapse_padding(["<|endoftext|>", "a", "b"]), ["<|endoftext|>", "a", "b"])

    def test_registers_follow_the_real_tokens(self):
        from model_run.cross_attention import place_registers
        labels = ["<pad>"] * 6 + ["<bos>", "a", " cube"]
        self.assertEqual(place_registers(labels, 5), ["<bos>", "a", " cube", "<register>", "<register>"])
        self.assertEqual(place_registers(["", "", "<bos>", " cube"], 3), ["<bos>", " cube", "<register>"])
        self.assertIsNone(place_registers(labels, 2))
        self.assertIsNone(place_registers(["<pad>"], 3))
        self.assertIsNone(place_registers(None, 3))

    def test_labels_only_name_keys_they_line_up_with(self):
        from model_run.cross_attention import align_labels, text_key_range
        self.assertEqual(align_labels(["a", "b"], 2), ["a", "b"])
        template = ["<|im_start|>", "system", "x", "<|im_end|>", "<|im_start|>", "user", "cat"]
        self.assertEqual(align_labels(template, 3), ["<|im_start|>", "user", "cat"])
        self.assertIsNone(align_labels(["a", "b", "c"], 2))
        self.assertIsNone(align_labels(["a"], 2))
        self.assertIsNone(align_labels(None, 2))
        self.assertEqual(text_key_range(1024, 1039, 15, allow_joint=True), (0, 15))
        self.assertIsNone(text_key_range(1024, 1039, 15))
