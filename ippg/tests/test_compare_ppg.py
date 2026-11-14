"""Tests for the waveform comparison; no video extraction required."""

from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_ppg as comparison


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(0.0, 10.0, 1 / 30)
        self.pulse = np.sin(2 * np.pi * 1.2 * self.time)

    def test_zscore_rejects_constant_and_invalid_arrays(self):
        for values in (np.ones(10), [1, np.nan], [[1, 2]], [1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                comparison.zscore(values)

    def test_identical_shape_ignores_amplitude_scale_and_offset(self):
        data = comparison.compare_filtered_signals(
            self.time, self.pulse, self.time, 10 + 3 * self.pulse
        )
        self.assertAlmostEqual(data.pearson_r, 1.0)
        self.assertAlmostEqual(data.sample_mae_z, 0.0)
        np.testing.assert_allclose(data.ippg_z, data.cppg_z, atol=1e-14)
        self.assertEqual(len(data.metrics()["samples"]), 5)

    def test_inverted_signal_is_not_silently_flipped(self):
        data = comparison.compare_filtered_signals(
            self.time, self.pulse, self.time, -self.pulse
        )
        self.assertAlmostEqual(data.pearson_r, -1.0)
        self.assertGreater(data.sample_mae_z, 0)

    def test_interpolation_uses_only_overlap(self):
        contact_time = np.arange(1.0, 9.0, 0.05)
        data = comparison.compare_filtered_signals(
            self.time, self.time, contact_time, contact_time
        )
        self.assertGreaterEqual(data.time_seconds[0], contact_time[0])
        self.assertLessEqual(data.time_seconds[-1], contact_time[-1])
        self.assertAlmostEqual(data.pearson_r, 1.0)

    def test_sample_validation_prevents_silent_extrapolation(self):
        for samples in ([5, 5.2], [5, 5, 5.4, 5.6, 5.8],
                        [5, 5.2, 5.4, 5.6, 20], [5, 5.2, 5.4, 5.6, np.nan]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                comparison.compare_filtered_signals(
                    self.time, self.pulse, self.time, self.pulse, samples
                )

    def test_no_overlap_and_nonmonotonic_time_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "common"):
            comparison.compare_filtered_signals(
                self.time, self.pulse, self.time + 20, self.pulse
            )
        with self.assertRaisesRegex(ValueError, "increasing"):
            comparison.compare_filtered_signals(
                self.time[::-1], self.pulse, self.time, self.pulse
            )

    def test_output_cannot_overwrite_input_even_with_flag(self):
        source = Path("example.json")
        with self.assertRaisesRegex(ValueError, "input"):
            comparison.validate_output(source, ".json", [source], True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(FileExistsError):
                comparison.validate_output(directory, "", [], False)

    def test_plot_returns_three_panels_and_rejects_bad_zoom(self):
        data = comparison.compare_filtered_signals(
            self.time, self.pulse, self.time, self.pulse
        )
        metadata = {"bandpass_hz": [0.6, 2.0], "pulse_method": "pos",
                    "video": "11.avi", "contact": "11.xlsx"}
        figure = comparison.plot_comparison(data, metadata)
        try:
            self.assertEqual(len(figure.axes), 3)
            figure.canvas.draw()
        finally:
            plt.close(figure)
        with self.assertRaisesRegex(ValueError, "Zoom"):
            comparison.plot_comparison(data, metadata, (5, 5.5))

    def test_cli_parses_paths_with_spaces_and_five_times(self):
        args = comparison.parse_arguments([
            "folder with spaces/a.avi", "reference.xlsx", "--no-show",
            "--sample-times", "6", "6.2", "6.4", "6.6", "6.8",
        ])
        self.assertEqual(args.video, Path("folder with spaces/a.avi"))
        self.assertTrue(args.no_show)
        self.assertEqual(args.sample_times, [6, 6.2, 6.4, 6.6, 6.8])


if __name__ == "__main__":
    unittest.main()
