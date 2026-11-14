"""Known-frequency and input-validation tests for the standalone TXT script."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_ground_truth import analyze_file


class GroundTruthTests(unittest.TestCase):
    def analyze(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ground_truth.txt"
            np.savetxt(path, rows)
            return analyze_file(path, **kwargs)

    def data(self):
        t = np.arange(1200) / 30
        ppg = 10 + np.sin(2*np.pi*1.2*t) + 0.7*np.sin(2*np.pi*5*t)
        return np.vstack((ppg, np.full_like(t, 999), t))

    def test_recovers_72_bpm_without_using_provided_hr(self):
        result = self.analyze(self.data())
        self.assertAlmostEqual(result.heart_rate_bpm, 72, delta=1.6)
        self.assertAlmostEqual(result.fps, 30)
        self.assertTrue(np.all(result.provided_hr == 999))
        self.assertEqual(result.duplicate_count, 0)

    def test_averages_duplicate_timestamp_without_stretching_time(self):
        original = self.data()
        rows = np.insert(original, 100, original[:,100], axis=1)
        rows[0,100] += 1
        rows[0,101] -= 1
        result = self.analyze(rows)
        self.assertEqual(result.duplicate_count, 1)
        self.assertAlmostEqual(result.fps, 30)
        np.testing.assert_allclose(result.uniform_ppg, original[0], atol=1e-10)

    def test_rejects_backwards_timestamp(self):
        rows = self.data()
        rows[2,100] = rows[2,99] - 1
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.analyze(rows)

    def test_rejects_flat_signal(self):
        rows = self.data()
        rows[0] = 1
        with self.assertRaisesRegex(ValueError, "flat"):
            self.analyze(rows)

    def test_rejects_short_signal(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            self.analyze(self.data()[:,:10])

    def test_rejects_wrong_row_layout_and_invalid_band(self):
        with self.assertRaisesRegex(ValueError, "3 numeric rows"):
            self.analyze(self.data()[:2])
        with self.assertRaisesRegex(ValueError, "Filter must"):
            self.analyze(self.data(), high=20)


if __name__ == "__main__":
    unittest.main()
