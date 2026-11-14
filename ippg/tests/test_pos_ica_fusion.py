"""Deterministic ICA, fusion, fallback and comparator contract tests."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sawtooth, square

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare_pos_ica_cppg as comparison
import benchmark_pos_ica_fusion as benchmark
import ippg_v2_refactored as ippg


class PosIcaFusionTests(unittest.TestCase):
    def setUp(self):
        self.fps = 30.0
        self.time = np.arange(900, dtype=float) / self.fps
        pulse = np.sin(2 * np.pi * 1.2 * self.time) + 0.2 * np.sin(
            2 * np.pi * 2.4 * self.time
        )
        sources = np.column_stack(
            (
                pulse,
                sawtooth(2 * np.pi * 0.31 * self.time),
                square(2 * np.pi * 0.17 * self.time),
            )
        )
        mixing = np.array(((0.8, 0.2, 0.1), (1.0, -0.3, 0.2), (0.4, 0.7, -0.2)))
        self.rgb = 120.0 + sources @ mixing.T
        self.signals = ippg.ExtractedSignals(
            self.time,
            self.rgb,
            self.rgb,
            self.rgb,
            None,
            np.sin(2 * np.pi * 0.2 * self.time),
            self.fps,
            "pos-ica",
            (0, 0, 16, 16),
        )
        self.config = ippg.IPPGConfig(
            pulse_method="pos-ica", denoise="none", background_method="none"
        )

    def test_fastica_is_deterministic_and_finds_known_bpm(self):
        first = ippg.calculate_pos_ica_fusion(self.signals, self.config)
        second = ippg.calculate_pos_ica_fusion(self.signals, self.config)
        self.assertEqual(first.selected_ica.component_index, second.selected_ica.component_index)
        np.testing.assert_allclose(
            first.selected_ica.processed.raw, second.selected_ica.processed.raw
        )
        self.assertAlmostEqual(first.selected_ica.heart_rate_bpm, 72.0, delta=2.5)
        self.assertAlmostEqual(first.spectral_bpm, 72.0, delta=2.5)

    def test_weights_polarity_and_outputs_are_valid(self):
        fusion = ippg.calculate_pos_ica_fusion(self.signals, self.config)
        self.assertAlmostEqual(sum(fusion.weights.values()), 1.0)
        self.assertIn(fusion.ica_polarity, (-1, 1))
        self.assertEqual(fusion.time_fused_signal.shape, self.time.shape)
        self.assertTrue(np.all(np.isfinite(fusion.time_fused_signal)))
        self.assertTrue(np.all(np.isfinite(fusion.spectral_score)))
        self.assertFalse(fusion.metadata["uses_contact_ppg"])

    def test_ica_sign_flip_does_not_change_bpm_or_weight(self):
        components, metadata = ippg.calculate_ica_components(self.rgb, self.config)
        with patch.object(
            ippg, "calculate_ica_components", return_value=(-components, metadata)
        ):
            flipped = ippg.calculate_pos_ica_fusion(self.signals, self.config)
        normal = ippg.calculate_pos_ica_fusion(self.signals, self.config)
        self.assertAlmostEqual(flipped.spectral_bpm, normal.spectral_bpm)
        self.assertAlmostEqual(flipped.weights["ica"], normal.weights["ica"])
        np.testing.assert_allclose(flipped.time_fused_signal, normal.time_fused_signal)

    def test_ica_failure_falls_back_to_pos(self):
        with patch.object(ippg, "calculate_ica_components", side_effect=RuntimeError("no convergence")):
            fusion = ippg.calculate_pos_ica_fusion(self.signals, self.config)
        self.assertEqual(fusion.status, "fallback-pos-ica-unavailable")
        self.assertEqual(fusion.weights, {"pos": 1.0, "ica": 0.0})
        self.assertIn("no convergence", fusion.metadata["errors"]["ica"])

    def test_missing_dependency_is_actionable(self):
        with patch.object(ippg, "FastICA", None):
            with self.assertRaisesRegex(RuntimeError, "requirements-ippg"):
                ippg.calculate_pos_ica_fusion(self.signals, self.config)

    def test_invalid_low_rank_rgb_is_rejected(self):
        low_rank = np.column_stack((self.time, 2 * self.time, 3 * self.time)) + 100
        with self.assertRaisesRegex(ValueError, "rank"):
            ippg.calculate_ica_components(low_rank, self.config)

    def test_runtime_fusion_source_does_not_reference_cppg(self):
        source = inspect.getsource(ippg.calculate_pos_ica_fusion).lower()
        self.assertNotIn("cppg", source)

    def test_analyze_signals_exposes_selected_source(self):
        result = ippg.analyze_signals(self.signals, self.config)
        self.assertEqual(result.heart_rate_source, "spectral_fusion")
        self.assertIsNotNone(result.fusion_result)
        timed = ippg.analyze_signals(
            self.signals,
            ippg.IPPGConfig(
                pulse_method="pos-ica",
                denoise="none",
                background_method="none",
                fusion_domain="time",
            ),
        )
        self.assertEqual(timed.heart_rate_source, "time_fusion")
        self.assertAlmostEqual(timed.heart_rate_bpm, timed.fusion_result.time_bpm)


class PosIcaComparatorTests(unittest.TestCase):
    def test_three_panels_and_zoom_do_not_change_metrics(self):
        fixture = PosIcaFusionTests()
        fixture.setUp()
        contact = np.sin(2 * np.pi * 1.2 * fixture.time)
        data = comparison.prepare_comparison(
            fixture.signals,
            fixture.time,
            contact,
            fixture.fps,
            fixture.config,
        )
        before = data.metrics()
        figure = comparison.plot_comparison(
            data,
            {"bandpass_hz": [0.6, 2.0]},
            (5.0, 15.0),
        )
        try:
            self.assertEqual(len(figure.axes), 3)
            self.assertEqual(before, data.metrics())
            self.assertIsNone(before["methods"]["pos-ica-spectral"]["waveform_metrics"])
        finally:
            plt.close(figure)

    def test_cli_accepts_fusion_settings(self):
        args = comparison.parse_arguments(
            ["21", "--fusion-min-correlation", "0.3", "--ica-random-state", "7"]
        )
        self.assertEqual(args.fusion_min_correlation, 0.3)
        self.assertEqual(args.ica_random_state, 7)

    def test_default_selection_keeps_pos_on_tie(self):
        rows = []
        for data_id, error in (("11", 2.0), ("12", 4.0)):
            rows.extend(
                (
                    {"data_id": data_id, "method": "pos-dwt", "absolute_error_bpm": error},
                    {"data_id": data_id, "method": "pos-ica-spectral", "absolute_error_bpm": error},
                )
            )
        selected, decision = benchmark.choose_default(rows)
        self.assertEqual(selected, "pos")
        self.assertIn("tied", decision["reason"])


if __name__ == "__main__":
    unittest.main()
