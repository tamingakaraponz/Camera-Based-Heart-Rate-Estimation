"""Explain cheek-color rejection on the prepared development frames."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from roi_compare_reference import FRAMES, PREDICTIONS, digest


def main():
    manifest = json.loads((FRAMES / 'manifest.json').read_text(encoding='utf8'))
    rows = []
    for record in manifest:
        if record['status'] != 'ok':
            continue
        path = FRAMES / record['image']
        geometry = Path(record['geometry_source'])
        if digest(path) != record['image_sha256'] or digest(geometry) != record['geometry_sha256']:
            raise ValueError(f'Changed input: {path}')
        frame = cv2.imread(str(path))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
        with np.load(geometry) as data:
            points = data['landmarks']
        scale = float(np.linalg.norm(points[263] - points[33]))
        radius = max(1, round(.02 * scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        masks = {}
        for name in 'ABC':
            region = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(region, [np.asarray(record['polygons'][name], np.int32)], 255)
            masks[name] = cv2.erode(region, kernel) > 0 if name in 'BC' else region > 0
        counts = [int(masks[side].sum()) for side in 'BC']
        if min(counts) < 20:
            continue
        samples = lab[masks['B'] | masks['C']]
        center = np.median(samples, axis=0)
        threshold = float(np.percentile(np.linalg.norm(samples - center, axis=1), 90))
        forehead = lab[masks['A']]
        distances = np.linalg.norm(forehead - center, axis=1)
        selected = distances <= threshold
        expected = cv2.imread(str(PREDICTIONS / f"{Path(record['image']).stem}_A_cheek_p90.png"), cv2.IMREAD_UNCHANGED)
        if expected is None or not np.array_equal(expected[masks['A']] > 0, selected):
            raise ValueError(f"Prediction mismatch: {record['image']}")
        median_forehead = np.median(forehead, axis=0)
        row = {'subject': record['subject'], 'time_s': record['requested_seconds'],
               'frame_index': record['frame_index'], 'forehead_pixels': len(forehead),
               'selected_pixels': int(selected.sum()), 'selected_fraction': float(selected.mean()),
               'threshold_p90': threshold, 'forehead_distance_median': float(np.median(distances)),
               'reference_L': float(center[0]), 'reference_a': float(center[1]), 'reference_b': float(center[2]),
               'forehead_L': float(median_forehead[0]), 'forehead_a': float(median_forehead[1]),
               'forehead_b': float(median_forehead[2]), 'radius': radius,
               'cheek_B_pixels': counts[0], 'cheek_C_pixels': counts[1]}
        rows.append(row)
    out = PREDICTIONS / 'cheek_color_diagnostics.csv'
    with out.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Analyzed {len(rows)} frames; selection masks reproduced exactly.')
    for row in rows:
        if row['time_s'] == 2:
            print(f"{row['subject']}: threshold={row['threshold_p90']:.2f}, forehead median={row['forehead_distance_median']:.2f}, selected={row['selected_fraction']:.1%}, L difference={row['forehead_L']-row['reference_L']:+.1f}")


if __name__ == '__main__':
    main()
