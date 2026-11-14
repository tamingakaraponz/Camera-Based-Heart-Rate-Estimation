"""Tests for the focused iPPG/cPPG comparison."""

from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_ippg_cppg as comparison


class FocusedComparisonTests(unittest.TestCase):
    def test_normalized_fft_recovers_known_frequency(self):
        fps = 30.0
        time = np.arange(900, dtype=float) / fps
        signal = np.sin(2 * np.pi * 1.2 * time)
        frequency, magnitude = comparison.normalized_fft(time, signal, fps, 0.6, 2.0)
        self.assertAlmostEqual(float(frequency[np.argmax(magnitude)] * 60), 72.0, delta=2.1)
        self.assertAlmostEqual(float(np.max(magnitude)), 1.0)

    def test_plot_contains_waveform_and_fft_panels(self):
        time = np.arange(300, dtype=float) / 30.0
        pulse = np.sin(2 * np.pi * 1.2 * time)
        result = {
            "time_seconds": time,
            "ippg_z": pulse,
            "cppg_z": pulse,
            "ippg_frequency_hz": np.array([0.6, 1.2, 2.0]),
            "ippg_fft_normalized": np.array([0.1, 1.0, 0.1]),
            "cppg_frequency_hz": np.array([0.6, 1.2, 2.0]),
            "cppg_fft_normalized": np.array([0.1, 1.0, 0.1]),
            "ippg_bpm": 72.0,
            "cppg_bpm": 72.0,
            "bpm_error": 0.0,
            "absolute_bpm_error": 0.0,
            "pearson_r": 1.0,
            "mae_z": 0.0,
            "rmse_z": 0.0,
        }
        figure = comparison.plot_ippg_cppg(result, {"bandpass_hz": [0.6, 2.0]})
        try:
            self.assertEqual(len(figure.axes), 2)
            self.assertEqual(len(figure.axes[0].lines), 2)
            self.assertEqual(len(figure.axes[1].lines), 2)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
