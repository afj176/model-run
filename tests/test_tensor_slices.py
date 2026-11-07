import unittest
try:
    import torch
except ImportError:  # these tests build PyTorch modules at import, so the whole module skips
    raise unittest.SkipTest("optional PyTorch dependency unavailable")
from model_run.capture import capture_forward
from model_run.tensor_slices import SliceRecorder, MAX_SLICES, SIDE, validate_slices
from model_run.trace_store import encode_trace


def distinct_latent():
    # Every value encodes its own channel, frame, row and column.
    channel, frame, row, col = torch.meshgrid(torch.arange(4.), torch.arange(3.), torch.arange(8.), torch.arange(8.), indexing="ij")
    return (1000 * channel + 100 * frame + 10 * row + col).unsqueeze(0)


def slice_of(record, positions):
    """The values of one slice, in the order the frontend reads them."""
    offset = 0
    for axis, position in zip(record["axes"], positions):
        offset = offset * len(axis["indices"]) + position
    size = record["rows"] * record["cols"]
    return record["values"][offset * size:(offset + 1) * size]


class TensorSliceTests(unittest.TestCase):
    def test_latent_slices_are_exact_and_named_when_the_shape_matches(self):
        latent = distinct_latent()
        record = SliceRecorder().record(latent, list(latent.shape))
        validate_slices(record)
        self.assertEqual([axis["name"] for axis in record["axes"]], ["channel", "frame"])
        self.assertEqual((record["row_axis"], record["col_axis"], record["pooled"]), ("height", "width", False))
        self.assertEqual(slice_of(record, [2, 1])[:3], [2100.0, 2101.0, 2102.0])
        self.assertEqual(slice_of(record, [3, 2])[-1], 3277.0)
        self.assertEqual(len(record["aggregate"]), 3 * 8 * 8)
        # The all-channels map of frame 1, row 0, col 0 is the mean magnitude across channels.
        self.assertAlmostEqual(record["aggregate"][64], (100 + 1100 + 2100 + 3100) / 4, places=2)

    def test_axes_are_not_named_without_a_known_latent(self):
        record = SliceRecorder().record(distinct_latent())
        self.assertEqual([axis["name"] for axis in record["axes"]], [None, None])
        self.assertEqual(record["layout"], "axes")
        self.assertEqual(record["aggregate_axis"], 0)

    def test_packed_tokens_are_restored_to_frames_only_when_the_count_fits(self):
        frames, side, features = 2, 8, 6
        tokens = torch.arange(frames * side * side, dtype=torch.float32).reshape(1, -1, 1).repeat(1, 1, features)
        record = SliceRecorder().record(tokens, [1, 16, frames, side, side])
        validate_slices(record)
        self.assertEqual(record["layout"], "packed_tokens")
        self.assertEqual([axis["name"] for axis in record["axes"]], ["feature", "frame"])
        self.assertTrue(record["assumption"])
        self.assertEqual(slice_of(record, [0, 1])[0], 64.0)
        self.assertEqual(record["aggregate"][64], 64.0)
        self.assertIsNone(SliceRecorder().record(torch.ones(1, 77, 6), [1, 16, frames, side, side]))
        self.assertIsNone(SliceRecorder().record(torch.ones(1, 77, 6)))

    def test_large_tensors_stay_bounded_and_say_they_were_averaged(self):
        record = SliceRecorder().record(torch.ones(1, 128, 40, 64, 48), [1, 128, 40, 64, 48])
        validate_slices(record)
        frame = record["axes"][1]
        self.assertEqual(len(frame["indices"]), 16)
        self.assertEqual(frame["indices"][0], 0)
        self.assertLessEqual(len(record["axes"][0]["indices"]) * 16, MAX_SLICES)
        self.assertEqual((record["rows"], record["cols"], record["source_rows"], record["pooled"]), (SIDE, SIDE, 64, True))

    def test_guidance_batches_and_reference_tokens_keep_their_reading(self):
        record = SliceRecorder().record(torch.ones(2, 4, 3, 8, 8), [1, 4, 3, 8, 8])
        self.assertEqual([a["name"] for a in record["axes"]], ["batch", "channel", "frame"])
        recorder = SliceRecorder()
        recorder.reference_tokens = 4
        tokens = torch.zeros(1, 16 + 4, 3)
        tokens[0, 16:] = 9  # reference tokens must not reach the map
        record = recorder.record(tokens, [1, 3, 4, 4])
        self.assertEqual((record["layout"], max(record["values"])), ("packed_tokens", 0.0))
        self.assertIn("Reference image", record["assumption"])

    def test_a_block_returning_text_first_is_read_by_its_image(self):
        class Block(torch.nn.Module):
            def forward(self, x):
                return torch.ones(1, 6, 8), torch.ones(1, 16, 8) * 2
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = Block()
            def forward(self, x):
                return self.layer(x)
        _, trace = capture_forward(Model().eval(), {"x": torch.zeros(1)}, ["layer"], latent_shape=[1, 3, 4, 4], max_tensor_elements=4)
        self.assertEqual((trace["events"][0]["shape"], trace["events"][0]["output_path"]), ([1, 16, 8], [1]))
        _, plain = capture_forward(Model().eval(), {"x": torch.zeros(1)}, ["layer"], max_tensor_elements=4)
        self.assertEqual(plain["events"][0]["shape"], [1, 6, 8])

    def test_unet_features_are_named_and_every_layer_keeps_its_view(self):
        recorder = SliceRecorder(max_values=524288, per_event=524288 // 208)
        recorder.conditional_batch = 1
        for _ in range(208):
            record = recorder.record(torch.ones(2, 1280, 8, 8), [2, 4, 64, 64])
            self.assertNotIn("unavailable", record)
            validate_slices(record)
        self.assertEqual([a["name"] for a in record["axes"]], ["batch", "channel"])
        self.assertEqual(record["layout"], "features")
        self.assertEqual(record["axes"][0]["conditional"], 1)
        self.assertEqual(SliceRecorder().record(torch.ones(1, 8, 3, 5), [1, 4, 64, 64])["layout"], "axes")  # not the latent's shape

    def test_vae_axes_are_only_named_when_the_shape_fits_the_image(self):
        recorder = SliceRecorder(convolutional=True)
        fits = recorder.record(torch.ones(1, 8, 2, 16, 16), [1, 4, 2, 8, 8])
        self.assertEqual([a["name"] for a in fits["axes"]], ["channel", "frame"])
        channels_last = recorder.record(torch.ones(1, 9, 64, 64, 48), [1, 128, 2, 8, 8])
        self.assertEqual(channels_last["layout"], "axes")

    def test_nothing_to_navigate_records_nothing(self):
        self.assertIsNone(SliceRecorder().record(torch.ones(1, 1, 8, 8)))
        self.assertIsNone(SliceRecorder().record(torch.ones(4, 4)))

    def test_budget_and_failures_report_instead_of_raising(self):
        recorder = SliceRecorder(max_values=10)
        self.assertIn("budget", recorder.record(distinct_latent())["unavailable"])
        class Broken:
            shape = (1, 2, 3, 4)
        self.assertIn("unavailable", SliceRecorder().record(Broken(), [1, 2, 3, 4]))

    def test_non_finite_values_become_null(self):
        latent = distinct_latent()
        latent[0, 0, 0, 0, 0] = float("nan")
        record = SliceRecorder().record(latent, list(latent.shape))
        self.assertIsNone(record["values"][0])
        validate_slices(record)

    def test_capture_records_slices_and_the_trace_validates(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Identity()
            def forward(self, x):
                return self.layer(x)
        latent = distinct_latent()
        _, trace = capture_forward(Model().eval(), {"x": latent}, ["layer"], latent_shape=list(latent.shape), max_tensor_elements=16)
        self.assertTrue(trace["events"][0]["truncated"])
        self.assertEqual(trace["events"][0]["slices"]["axes"][1]["name"], "frame")
        encode_trace(trace)
        trace["events"][0]["slices"]["values"].pop()
        with self.assertRaises(ValueError):
            encode_trace(trace)


if __name__ == "__main__":
    unittest.main()
