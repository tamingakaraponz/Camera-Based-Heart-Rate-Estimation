"""Synthetic checks for the shared-RGB, three-panel comparison."""

from pathlib import Path
from contextlib import chdir
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_gr_pos_cppg as comparison
from ippg_v2_refactored import ExtractedSignals, IPPGConfig, calculate_pos


class DataPathTests(unittest.TestCase):
    def test_id_maps_to_project_paths_independent_of_working_directory(self):
        root = Path(comparison.__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            video, contact = comparison.dataset_paths("21")
        self.assertEqual(video, root / "DATA/temp/21.avi")
        self.assertEqual(contact, root / "DATAcppg/21.xlsx")

    def test_cli_accepts_id_without_prompt(self):
        args = comparison.parse_arguments(["62", "--no-show"])
        with patch("builtins.input", side_effect=AssertionError("Unexpected prompt")):
            self.assertEqual(comparison.resolve_input_paths(args.video, args.cppg),
                             comparison.dataset_paths("62"))

    def test_interactive_id_needs_only_one_answer(self):
        with patch("builtins.input", return_value=" 21 ") as prompt:
            self.assertEqual(comparison.resolve_input_paths(None, None), comparison.dataset_paths("21"))
        prompt.assert_called_once()

    def test_explicit_paths_still_work(self):
        video, contact = Path("videos/my video.avi"), Path("reference.xlsx")
        with patch("builtins.input", side_effect=AssertionError("Unexpected prompt")):
            self.assertEqual(comparison.resolve_input_paths(video, contact),
                             (video.resolve(), contact.resolve()))

    def test_single_video_path_still_prompts_for_contact(self):
        with patch("builtins.input", return_value="reference.xlsx") as prompt:
            result = comparison.resolve_input_paths(Path("my video.avi"), None)
        self.assertEqual(result, (Path("my video.avi").resolve(), Path("reference.xlsx").resolve()))
        prompt.assert_called_once()

    def test_ambiguous_id_and_custom_contact_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "alone"):
            comparison.resolve_input_paths(Path("21"), Path("62.xlsx"))

    def test_invalid_id_rejected_and_leading_zero_preserved(self):
        for value in ("", "../21", "-1", "21.avi", "2.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                comparison.dataset_paths(value)
        self.assertEqual(comparison.dataset_paths("021")[0].name, "021.avi")


class ThreeMethodTests(unittest.TestCase):
    def setUp(self):
        self.config = IPPGConfig(show_preview=False)
        self.t = np.arange(360) / 30 + 2
        wave = np.sin(2 * np.pi * 1.2 * self.t)
        rgb = np.array([160, 120, 90])[None, :] * (
            1 + 0.01 * wave[:, None] * np.array([0.2, 1.0, 0.1])[None, :]
        )
        self.signals = ExtractedSignals(
            self.t, rgb, rgb, rgb, calculate_pos(rgb, 30), np.zeros(360),
            30.0, "pos", (0, 0, 32, 32),
        )
        self.ct = np.arange(300) / 20
        self.ca = 800 + 10 * np.sin(2 * np.pi * 1.2 * self.ct)
        self.metadata = {"video": "11.avi", "contact": "11.xlsx", "bandpass_hz": [0.6, 2.0]}

    def prepare(self):
        return comparison.prepare_comparison(self.signals, self.ct, self.ca, 20, self.config)

    def test_shared_grid_and_normalization(self):
        data = self.prepare()
        self.assertEqual(data.cppg_z.shape, data.gr_z.shape)
        self.assertEqual(data.cppg_z.shape, data.pos_z.shape)
        for pulse in (data.cppg_z, data.gr_z, data.pos_z, data.pos_dwt_z):
            self.assertAlmostEqual(float(np.mean(pulse)), 0)
            self.assertAlmostEqual(float(np.std(pulse)), 1)
        self.assertEqual(set(data.metrics()["methods"]), {"g-r", "pos", "pos-dwt"})

    def test_cppg_bpm_comes_from_full_contact_series(self):
        expected = comparison.analyze_contact_samples(
            self.ct,
            self.ca,
            20,
            self.config.forehead_low_hz,
            self.config.forehead_high_hz,
            filter_order=3,
        ).heart_rate_bpm
        with patch.object(
            comparison,
            "analyze_contact_samples",
            wraps=comparison.analyze_contact_samples,
        ) as analyze_contact:
            data = self.prepare()
        self.assertEqual(analyze_contact.call_count, 1)
        self.assertAlmostEqual(data.bpm["cppg"], expected)
        self.assertIn("full workbook", data.metrics()["fft_scope"])

    def test_pos_is_reused_and_gr_is_computed_from_same_rgb(self):
        with patch.object(comparison, "calculate_pos", side_effect=AssertionError("POS recomputed")):
            data = self.prepare()
        expected = comparison.bandpass_filter(
            self.signals.combined_rgb[:, 1] - self.signals.combined_rgb[:, 0], 30, 0.6, 2, 6
        )
        indices = np.searchsorted(self.t, data.time_seconds)
        np.testing.assert_allclose(data.gr_z, comparison.zscore(expected[indices]))

    def test_video_is_extracted_once(self):
        with patch.object(comparison, "extract_video_signals", return_value=self.signals) as extract:
            with patch.object(comparison, "load_contact_ppg", return_value=(self.ct, self.ca, 20)):
                comparison.analyze_pair(Path("video.avi"), Path("contact.xlsx"), self.config)
        self.assertEqual(extract.call_count, 1)

    def test_exactly_three_axes_same_reference_and_scales_and_zoom_invariance(self):
        data = self.prepare()
        metrics_before = data.metrics()
        figure = comparison.plot_comparison(data, self.metadata, (5, 8))
        try:
            self.assertEqual(len(figure.axes), 3)
            first = figure.axes[0]
            for other in figure.axes[1:]:
                np.testing.assert_array_equal(first.lines[1].get_ydata(), other.lines[1].get_ydata())
                self.assertEqual(first.get_xlim(), other.get_xlim())
                self.assertEqual(first.get_ylim(), other.get_ylim())
            self.assertEqual(metrics_before, data.metrics())
            figure.canvas.draw()
        finally:
            plt.close(figure)

    def test_out_of_range_zoom_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Zoom"):
            comparison.plot_comparison(self.prepare(), self.metadata, (0, 4))

    def test_three_panel_cli_keeps_both_pos_branches(self):
        args = comparison.parse_arguments(["21", "--wavelet-level", "3"])
        self.assertEqual(args.wavelet_level, 3)
        self.assertFalse(hasattr(args, "denoise"))


if __name__ == "__main__":
    unittest.main()
