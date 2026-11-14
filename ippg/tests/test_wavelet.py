"""Deterministic DWT contract, safety and bypass regression tests."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import pywt
import ippg_v2_refactored as ippg
import compare_ppg
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class WaveletTests(unittest.TestCase):
    def setUp(self):
        self.config = ippg.IPPGConfig(show_preview=False)
        self.t = np.arange(901) / 30
        self.clean = np.sin(2 * np.pi * 1.2 * self.t)
        self.noisy = self.clean + np.random.default_rng(20260830).normal(0, 0.2, self.t.size)

    def test_roundtrip_without_threshold_even_and_odd(self):
        for n in (900, 901):
            raw = self.noisy[:n]
            coefficients = pywt.wavedec(raw, "db4", mode="symmetric", level=4)
            reconstructed = pywt.waverec(coefficients, "db4", mode="symmetric")[:n]
            np.testing.assert_allclose(reconstructed, raw, atol=1e-12)
            output, meta = ippg.wavelet_denoise(raw, replace(self.config, wavelet_strength=0), 30)
            np.testing.assert_array_equal(output, raw)
            self.assertFalse(np.shares_memory(raw, output))
            self.assertFalse(meta["applied"])

    def test_threshold_formula_and_untouched_approximation(self):
        raw = self.noisy.copy()
        coefficients = pywt.wavedec(raw, "db4", mode="symmetric", level=4)
        sigma = np.median(abs(coefficients[-1] - np.median(coefficients[-1]))) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(raw)))
        expected = pywt.waverec([coefficients[0]] + [pywt.threshold(d, threshold, mode="soft")
                                                   for d in coefficients[1:]], "db4", mode="symmetric")[:len(raw)]
        output, meta = ippg.wavelet_denoise(raw, replace(self.config, wavelet_strategy="all-details"), 30)
        np.testing.assert_array_equal(raw, self.noisy)
        np.testing.assert_allclose(output, expected)
        self.assertAlmostEqual(meta["threshold"], threshold)
        self.assertEqual(meta["actual_level"], 4)

    def test_protected_strategy_leaves_pulse_overlapping_detail_coefficients_unchanged(self):
        raw = self.noisy.copy()
        coefficients = pywt.wavedec(raw, "db4", mode="symmetric", level=4)
        output, meta = ippg.wavelet_denoise(raw, self.config, 30)
        expected = pywt.waverec(
            [coefficients[0], coefficients[1], coefficients[2],
             pywt.threshold(coefficients[3], meta["threshold"], mode="soft"),
             pywt.threshold(coefficients[4], meta["threshold"], mode="soft")],
            "db4", mode="symmetric",
        )[:len(raw)]
        np.testing.assert_allclose(output, expected)
        actions = {row["coefficient"]: row["action"] for row in meta["coefficients"]}
        self.assertEqual(actions, {"cA4": "preserved_approximation", "cD4": "preserved_pulse_overlap",
                                   "cD3": "preserved_pulse_overlap", "cD2": "soft_threshold",
                                   "cD1": "soft_threshold"})
        self.assertEqual(meta["pulse_band_hz"], [0.6, 2.0])

    def test_known_bpm_clean_and_seeded_noise(self):
        for pulse in (self.clean, self.noisy):
            processed = ippg.process_pulse(pulse, 30, "pos", self.config)
            for stage in (processed.raw, processed.reconstructed, processed.filtered):
                self.assertEqual(stage.shape, pulse.shape)
                self.assertTrue(np.isfinite(stage).all())
            f, spectrum = ippg.calculate_spectrum(self.t, processed.filtered, 30)
            self.assertAlmostEqual(ippg.estimate_heart_rate(f, spectrum, .6, 2), 72, delta=2.1)

    def test_strength_grid_preserves_known_bpm_when_pulse_bands_are_protected(self):
        for strength in (0.1, 0.25, 0.5, 1.0):
            with self.subTest(strength=strength):
                processed = ippg.process_pulse(
                    self.noisy, 30, "pos", replace(self.config, wavelet_strength=strength),
                )
                frequency, spectrum = ippg.calculate_spectrum(self.t, processed.filtered, 30)
                self.assertAlmostEqual(
                    ippg.estimate_heart_rate(frequency, spectrum, .6, 2), 72, delta=2.1,
                )

    def test_bypass_matches_old_filter_spectrum_and_bpm_exactly(self):
        config = replace(self.config, denoise="none")
        expected = ippg.bandpass_filter(self.noisy, 30, .6, 2, 6)
        output = ippg.process_pulse(self.noisy, 30, "pos", config)
        np.testing.assert_array_equal(output.filtered, expected)
        f, old = ippg.calculate_spectrum(self.t, expected, 30)
        _, new = ippg.calculate_spectrum(self.t, output.filtered, 30)
        np.testing.assert_array_equal(old, new)
        self.assertEqual(ippg.estimate_heart_rate(f, old, .6, 2), ippg.estimate_heart_rate(f, new, .6, 2))

    def test_short_signal_auto_warns_explicit_level_errors(self):
        with self.assertWarns(RuntimeWarning):
            output, meta = ippg.wavelet_denoise(np.arange(5.), self.config, 30)
        np.testing.assert_array_equal(output, np.arange(5.))
        self.assertEqual(meta["status"], "skipped_short_signal")
        with self.assertRaisesRegex(ValueError, "exceeds maximum"):
            ippg.wavelet_denoise(np.arange(5.), replace(self.config, wavelet_level=1), 30)
        _, meta = ippg.wavelet_denoise(np.arange(28.), self.config, 30)
        self.assertEqual(meta["actual_level"], 2)

    def test_bad_settings_and_nonfinite_input(self):
        for settings in ({"wavelet_level": 99}, {"wavelet_level": 0}, {"wavelet_level": 1.5},
                         {"wavelet_strength": -1}, {"wavelet_strength": np.inf},
                         {"wavelet_strength": np.nan}, {"wavelet": "invalid"}, {"denoise": "invalid"},
                         {"wavelet_strategy": "invalid"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ippg.wavelet_denoise(self.clean, replace(self.config, **settings), 30)
        for bad in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                ippg.wavelet_denoise(np.array([1., bad]), self.config, 30)

    def test_flat_signal_and_zero_spectrum_do_not_report_bpm(self):
        for mode in ("none", "dwt"):
            with self.assertRaisesRegex(ValueError, "flat"):
                ippg.process_pulse(np.ones(900), 30, "pos", replace(self.config, denoise=mode))
        with self.assertRaisesRegex(ValueError, "No usable spectral peak"):
            ippg.estimate_heart_rate(np.array([0., 1.]), np.zeros(2), .6, 2)
        output, meta = ippg.wavelet_denoise(np.zeros(900), self.config, 30)
        np.testing.assert_array_equal(output, np.zeros(900))
        self.assertEqual(meta["status"], "skipped_zero_threshold")

    def test_missing_dependency_is_explicit_bypass_and_gr_still_work(self):
        with patch.object(ippg, "pywt", None):
            with self.assertRaisesRegex(RuntimeError, "pip install PyWavelets"):
                ippg.process_pulse(self.clean, 30, "pos", self.config)
            ippg.process_pulse(self.clean, 30, "pos", replace(self.config, denoise="none"))
            gr = ippg.process_pulse(self.clean, 30, "g-r", self.config)
            self.assertEqual(gr.denoise_metadata["status"], "not_applicable")

    def test_main_analysis_keeps_stages_and_bypass_bpm(self):
        rgb = np.ones((len(self.t), 3)) * [160, 120, 90]
        signals = ippg.ExtractedSignals(self.t, rgb, rgb, rgb, self.noisy,
                                        None, 30., "pos", None)
        for denoise in ("none", "dwt"):
            config = replace(self.config, denoise=denoise)
            result = ippg.analyze_signals(signals, config)
            expected = ippg.process_pulse(self.noisy, 30, "pos", config)
            np.testing.assert_array_equal(result.signals.pulse_signal, self.noisy)
            np.testing.assert_array_equal(result.reconstructed_pulse_signal, expected.reconstructed)
            np.testing.assert_array_equal(result.filtered_pulse_signal, expected.filtered)
            self.assertEqual(result.background_method, "none")
            self.assertIsNone(result.filtered_background)
            f, spectrum = ippg.calculate_spectrum(self.t, expected.filtered, 30)
            self.assertEqual(result.heart_rate_bpm, ippg.estimate_heart_rate(f, spectrum, .6, 2))

    def test_single_comparison_uses_shared_processor_and_records_dwt(self):
        rgb = np.ones((len(self.t), 3)) * [160, 120, 90]
        signals = ippg.ExtractedSignals(self.t, rgb, rgb, rgb, self.noisy,
                                        np.zeros(len(self.t)), 30., "pos", (0, 0, 32, 32))
        contact_time = np.arange(650) / 20
        contact = np.sin(2 * np.pi * 1.2 * contact_time)
        with patch.object(compare_ppg, "extract_video_signals", return_value=signals), \
             patch.object(compare_ppg, "load_contact_ppg", return_value=(contact_time, contact, 20)), \
             patch.object(compare_ppg, "process_pulse", wraps=ippg.process_pulse) as process:
            _, metadata = compare_ppg.analyze_pair(Path("11.avi"), Path("11.xlsx"), self.config,
                                                  compare_ppg.DEFAULT_SAMPLE_TIMES)
        process.assert_called_once()
        self.assertEqual(metadata["denoise"]["method"], "dwt")
        self.assertTrue(metadata["denoise"]["applied"])


if __name__ == "__main__":
    unittest.main()
