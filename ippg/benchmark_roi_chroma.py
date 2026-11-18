"""Benchmark the cheek chroma-only candidate using the ROI replay harness."""
import cv2
import numpy as np

import benchmark_roi_methods as benchmark


def cheek_chroma_mask(frame, regions, points):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
    distance = float(np.linalg.norm(points[263] - points[33]))
    radius = max(1, round(.02 * distance))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    inner = {side: cv2.erode(regions[side], kernel) > 0 for side in 'BC'}
    if any(int(inner[side].sum()) < 20 for side in 'BC'):
        return None
    samples = lab[inner['B'] | inner['C']]
    center = np.median(samples, axis=0)
    threshold = np.percentile(np.linalg.norm(samples[:, 1:] - center[1:], axis=1), 90)
    return (np.linalg.norm(lab[:, :, 1:] - center[1:], axis=2) <= threshold) & (regions['A'] > 0)


if __name__ == '__main__':
    benchmark.OUT = benchmark.ROOT / 'outputs/roi_chroma_timing'
    benchmark.METHODS = ('cheek_chroma_p90',)
    benchmark.cheek_mask = cheek_chroma_mask
    benchmark.main()
