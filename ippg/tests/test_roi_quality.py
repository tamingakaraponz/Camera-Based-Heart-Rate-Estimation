"""Multi-patch ROI selection and robust RGB aggregation tests."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ippg_v2_refactored as ippg


class AllSkinDetector:
    def __init__(self, rejected_prefix: str | None = None):
        self.rejected_prefix = rejected_prefix

    def create_mask(self, region, geometry_mask, region_name):
        if self.rejected_prefix and region_name.startswith(self.rejected_prefix):
            return np.zeros_like(geometry_mask, dtype=np.uint8)
        return np.where(geometry_mask > 0, 255, 0).astype(np.uint8)


def face_regions(value: int = 100) -> ippg.FaceRegions:
    forehead = np.full((30, 90, 3), value, dtype=np.uint8)
    cheek = np.full((60, 40, 3), value, dtype=np.uint8)
    forehead_geometry = np.full(forehead.shape[:2], 255, dtype=np.uint8)
    cheek_geometry = np.full(cheek.shape[:2], 255, dtype=np.uint8)
    return ippg.FaceRegions(
        forehead,
        cheek.copy(),
        cheek.copy(),
        (0, 0, 90, 90),
        (0, 0, 90, 30),
        (0, 30, 40, 90),
        (50, 30, 90, 90),
        forehead_geometry,
        cheek_geometry.copy(),
        cheek_geometry.copy(),
    )


class QualityROITests(unittest.TestCase):
    def setUp(self):
        self.config = ippg.IPPGConfig(roi_method="quality")

    def test_trimmed_mean_rejects_bright_outliers(self):
        normal = np.tile(np.array([[100, 110, 120]], dtype=float), (90, 1))
        highlights = np.tile(np.array([[255, 255, 255]], dtype=float), (10, 1))
        result = ippg.robust_trimmed_mean_rgb(
            np.vstack((normal, highlights)), trim_fraction=0.10
        )
        np.testing.assert_allclose(result, [120, 110, 100])

    def test_patch_selection_excludes_overexposed_patch(self):
        region = np.empty((30, 90, 3), dtype=np.uint8)
        region[:, :30] = 100
        region[:, 30:60] = 110
        region[:, 60:] = 255
        geometry = np.full(region.shape[:2], 255, dtype=np.uint8)
        selection, selected_mask = ippg.select_quality_roi(
            region,
            geometry,
            "forehead",
            rows=1,
            columns=3,
            keep_count=2,
            skin_detector=AllSkinDetector(),
            config=self.config,
        )
        self.assertEqual(selection.selected_patches, (0, 1))
        np.testing.assert_allclose(selection.rgb, [105, 105, 105])
        self.assertTrue(np.all(selected_mask[:, :60] == 255))
        self.assertTrue(np.all(selected_mask[:, 60:] == 0))

    def test_one_bad_cheek_does_not_discard_the_frame(self):
        result = ippg.quality_weighted_face_rgb(
            face_regions(), AllSkinDetector("right_cheek"), self.config
        )
        forehead, cheeks, combined, masks, qualities, weights = result
        np.testing.assert_allclose(forehead, [100, 100, 100])
        np.testing.assert_allclose(cheeks, [100, 100, 100])
        np.testing.assert_allclose(combined, [100, 100, 100])
        self.assertGreater(qualities[0], 0)
        self.assertGreater(qualities[1], 0)
        self.assertEqual(qualities[2], 0)
        self.assertEqual(weights[2], 0)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertFalse(np.any(masks[2]))

    def test_invalid_roi_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "trim"):
            ippg.validate_roi_config(
                ippg.IPPGConfig(roi_trim_fraction=0.5)
            )
        with self.assertRaisesRegex(ValueError, "forehead"):
            ippg.validate_roi_config(
                ippg.IPPGConfig(
                    roi_forehead_columns=2,
                    roi_keep_forehead_patches=3,
                )
            )


if __name__ == "__main__":
    unittest.main()
