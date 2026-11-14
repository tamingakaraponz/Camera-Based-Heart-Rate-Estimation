"""Analytic-signal checks for the diagnostic, not physiological validation."""
import unittest

import numpy as np

from diagnose_time_windows import analyze_window, differences


class TimeWindowDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(0, 40, 1 / 30)

    def test_known_stationary_pulse_and_resolution(self):
        y = np.sin(2 * np.pi * 1.2 * self.t)
        result, _, _ = analyze_window(self.t, y, 2.1, 12.1)
        self.assertAlmostEqual(result["bin_bpm"], 72)
        self.assertAlmostEqual(result["bin_spacing_bpm"], 6)
        # Log interpolation of roundoff-sized side bins need not be exactly zero.
        self.assertAlmostEqual(result["interpolated_bpm"], 72, delta=0.01)

    def test_known_two_competing_peaks(self):
        y = np.sin(2 * np.pi * 1.2 * self.t) + 0.7 * np.sin(2 * np.pi * 1.6 * self.t)
        result, _, _ = analyze_window(self.t, y, 2.1, 12.1)
        self.assertAlmostEqual(result["bin_bpm"], 72)
        self.assertAlmostEqual(result["separated_peaks"][1]["bpm"], 96)
        self.assertAlmostEqual(result["runner_up_to_main_power"], 0.49, places=5)

    def test_does_not_extrapolate_missing_window(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            analyze_window(self.t[:300], np.ones(300), 2.1, 12.1)

    def test_flat_signal_is_not_reported_as_pulse(self):
        with self.assertRaisesRegex(ValueError, "No usable"):
            analyze_window(self.t, np.ones(len(self.t)), 2.1, 12.1)

    def test_multiplier_can_make_agreement_worse(self):
        result = differences(60, 60)
        self.assertEqual(result["outcome"], "farther")
        self.assertAlmostEqual(result["before_bpm"], 0)
        self.assertAlmostEqual(result["after_bpm"], 3.75)


if __name__ == "__main__":
    unittest.main()
