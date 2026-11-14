"""Canonical full-workbook contact PPG processing tests."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_background_methods import analyze_contact_samples


class ContactPPGTests(unittest.TestCase):
    def test_bandpass_then_fft_recovers_known_bpm(self):
        fps = 20.0
        time = np.arange(600, dtype=float) / fps + 1_703_000_000.0
        amplitude = 870 + 5 * np.sin(2 * np.pi * 1.2 * (time - time[0]))
        result = analyze_contact_samples(time, amplitude, fps, 0.6, 2.0)
        self.assertAlmostEqual(result.heart_rate_bpm, 72.0, delta=2.1)
        self.assertAlmostEqual(result.time_seconds[0], 0.0)
        self.assertEqual(result.filtered_amplitude.shape, result.time_seconds.shape)

    def test_full_series_bpm_is_not_defined_by_video_crop(self):
        fps = 20.0
        time = np.arange(800, dtype=float) / fps
        amplitude = 870 + 5 * np.sin(2 * np.pi * 1.1 * time)
        full = analyze_contact_samples(time, amplitude, fps, 0.6, 2.0)
        # A video may begin after calibration, but the canonical cPPG result
        # remains the full-series analysis above rather than a new cropped FFT.
        self.assertAlmostEqual(full.heart_rate_bpm, 66.0, delta=1.6)
        self.assertGreater(full.time_seconds.size, np.count_nonzero(time >= 2.0))


if __name__ == "__main__":
    unittest.main()
