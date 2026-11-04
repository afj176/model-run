import unittest

try:
    import torch
except ImportError:
    torch = None

from model_run.cross_attention import image_grid, pooled_grid, measure, MAX_MAP_VALUES


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
