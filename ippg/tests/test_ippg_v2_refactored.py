"""Unit tests for the iPPG v2 signal and background-processing steps."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parent.parent / "ippg_v2_refactored.py"
SPEC = importlib.util.spec_from_file_location("ippg_v2_refactored", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {MODULE_PATH}")
IPPG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IPPG
SPEC.loader.exec_module(IPPG)


class ResamplingTests(unittest.TestCase):
    def test_resampling_preserves_channels_and_uniform_time(self) -> None:
        time = np.array([0.0, 0.1, 0.3, 0.4])
        values = np.column_stack((time, 2.0 * time, 3.0 * time))

        uniform_time, uniform_values = IPPG.resample_to_fps(time, values, 10.0)

        np.testing.assert_allclose(np.diff(uniform_time), 0.1)
        np.testing.assert_allclose(uniform_values[:, 0], uniform_time)
        np.testing.assert_allclose(uniform_values[:, 1], 2.0 * uniform_time)
        np.testing.assert_allclose(uniform_values[:, 2], 3.0 * uniform_time)

    def test_relative_luminance_has_zero_mean(self) -> None:
        luminance = np.array([90.0, 100.0, 110.0])

        relative_y = IPPG.relative_luminance(luminance)

        np.testing.assert_allclose(relative_y, [-0.1, 0.0, 0.1])
        self.assertAlmostEqual(float(np.mean(relative_y)), 0.0)

    def test_relative_luminance_rejects_zero_mean(self) -> None:
        with self.assertRaisesRegex(ValueError, "undefined"):
            IPPG.relative_luminance(np.zeros(10))


class POSTests(unittest.TestCase):
    def test_common_brightness_change_is_rejected(self) -> None:
        fps = 30.0
        time = np.arange(8.0 * fps) / fps
        illumination = 1.0 + 0.20 * np.sin(2.0 * np.pi * 0.8 * time)
        skin_rgb = np.array([160.0, 120.0, 90.0])
        rgb = illumination[:, None] * skin_rgb[None, :]

        pulse = IPPG.calculate_pos(rgb, fps)

        self.assertEqual(pulse.shape, (time.size,))
        self.assertTrue(np.all(np.isfinite(pulse)))
        self.assertLess(float(np.max(np.abs(pulse))), 1e-8)

    def test_synthetic_pulse_frequency_is_recovered(self) -> None:
        fps = 30.0
        target_hz = 1.2
        time = np.arange(12.0 * fps) / fps
        cardiac_wave = np.sin(2.0 * np.pi * target_hz * time)
        illumination = 1.0 + 0.08 * np.sin(2.0 * np.pi * 0.25 * time)
        skin_rgb = np.array([160.0, 120.0, 90.0])
        channel_response = np.array([0.2, 1.0, 0.1])
        rgb = (
            skin_rgb[None, :]
            * illumination[:, None]
            * (1.0 + 0.01 * cardiac_wave[:, None] * channel_response[None, :])
        )

        pulse = IPPG.calculate_pos(rgb, fps)
        filtered = IPPG.bandpass_filter(pulse, fps, 0.6, 2.0, order=3)
        frequencies, spectrum = IPPG.calculate_spectrum(time, filtered, fps)
        valid = (frequencies >= 0.6) & (frequencies <= 2.0)
        detected_hz = frequencies[valid][np.argmax(np.abs(spectrum[valid]))]

        self.assertAlmostEqual(float(detected_hz), target_hz, delta=0.1)

    def test_invalid_shape_and_short_signal_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            IPPG.calculate_pos(np.ones((100, 2)), 30.0)
        with self.assertRaisesRegex(ValueError, "at least"):
            IPPG.calculate_pos(np.ones((10, 3)), 30.0)


class BackgroundMethodTests(unittest.TestCase):
    def test_background_is_disabled_by_default_for_every_pulse_method(self) -> None:
        automatic = IPPG.IPPGConfig(background_method=None)
        explicit = IPPG.IPPGConfig(background_method="legacy")

        self.assertEqual(IPPG.resolve_background_method(automatic, "pos"), "none")
        self.assertEqual(IPPG.resolve_background_method(automatic, "g-r"), "none")
        self.assertEqual(IPPG.resolve_background_method(automatic, "pos-ica"), "none")
        self.assertEqual(IPPG.resolve_background_method(explicit, "pos"), "legacy")

    def test_explicit_background_method_requires_extracted_background(self) -> None:
        time = np.arange(300, dtype=float) / 30.0
        pulse = np.sin(2 * np.pi * 1.2 * time)
        rgb = np.ones((time.size, 3), dtype=float)
        signals = IPPG.ExtractedSignals(
            time, rgb, rgb, rgb, pulse, None, 30.0, "pos", None
        )
        config = IPPG.IPPGConfig(background_method="power", denoise="none")
        with self.assertRaisesRegex(ValueError, "requires background extraction"):
            IPPG.analyze_signals(signals, config)

    def test_power_method_penalizes_frequency_shared_with_background(self) -> None:
        frequencies = np.array([0.0, 0.8, 1.2, 1.6])
        pulse = np.array([0.0, 1.0, 10.0, 5.0], dtype=complex)
        background = np.array([0.0, 1.0, 10.0, 0.0], dtype=complex)
        config = IPPG.IPPGConfig(background_power_weight=1.0)

        no_penalty = IPPG.calculate_background_score(
            frequencies, pulse, background, config, "none"
        )
        legacy = IPPG.calculate_background_score(
            frequencies, pulse, background, config, "legacy"
        )
        power_penalty = IPPG.calculate_background_score(
            frequencies, pulse, background, config, "power"
        )

        np.testing.assert_allclose(no_penalty, np.abs(pulse) ** 2)
        self.assertTrue(np.all(np.isfinite(legacy)))
        self.assertTrue(np.all(power_penalty >= 0.0))
        self.assertLess(power_penalty[2], power_penalty[3])

    def test_stable_background_patch_is_fixed_after_calibration(self) -> None:
        config = IPPG.IPPGConfig(
            background_patch_size=16,
            background_calibration_frames=3,
            background_exclusion_margin_pixels=0,
            background_min_luminance=1.0,
            background_max_luminance=254.0,
        )
        tracker = IPPG.StableBackgroundPatch(config)
        face_box = (16, 16, 48, 48)

        output = None
        for frame_index in range(3):
            frame = np.full((64, 64, 3), 100 + frame_index, dtype=np.uint8)
            frame[0:16, 0:16] = 120
            output = tracker.update(frame, face_box)

        self.assertIsNotNone(output)
        self.assertIsNotNone(tracker.box)
        assert tracker.box is not None
        self.assertFalse(IPPG._boxes_intersect(tracker.box, face_box))

        selected_box = tracker.box
        next_frame = np.full((64, 64, 3), 140, dtype=np.uint8)
        next_value = tracker.update(next_frame, face_box)
        self.assertEqual(tracker.box, selected_box)
        self.assertTrue(np.isfinite(next_value))


if __name__ == "__main__":
    unittest.main()
