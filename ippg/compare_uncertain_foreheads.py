"""Bound pairwise differences in forehead contamination using shared labels."""
import csv
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from roi_compare_reference import ANNOTATIONS, FRAMES, PREDICTIONS, digest


def compare(records):
    labels, selections = [], []
    names = ['fixed', 'mediapipe', 'cheek_p90', 'color_only_p90']
    for record in records:
        stem = Path(record['image']).stem
        label_file = ANNOTATIONS / f'{stem}.png'
        meta = json.loads(label_file.with_suffix('.json').read_text(encoding='utf8'))
        assert digest(label_file) == meta['labels_sha256']
        assert meta['image_sha256'] == record['image_sha256'] == digest(FRAMES / record['image'])
        assert meta['polygons'] == record['polygons']
        label = cv2.imread(str(label_file), cv2.IMREAD_UNCHANGED)
        region = np.zeros_like(label)
        cv2.fillPoly(region, [np.asarray(record['polygons']['A'], np.int32)], 1)
        labels.append(label[region > 0])
        paths = [PREDICTIONS / f'{stem}_A_{name}.png' for name in names[:3]]
        paths.append(PREDICTIONS / 'cheek_color_experiment_masks' / f'{stem}_color_only_p90.png')
        selections.append([cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[region > 0] > 0 for path in paths])
    label = np.concatenate(labels)
    assert np.isin(label, [0, 1, 2, 3]).all()
    masks = np.concatenate(selections, axis=1)
    for a, b in combinations(range(len(names)), 2):
        left, right = masks[a], masks[b]
        low = high = None
        if left.any() and right.any():
            weight = left / left.sum() - right / right.sum()
            known = weight[label >= 2].sum()
            unknown = weight[label == 0]
            low = round(100 * (known + unknown.clip(max=0).sum()), 4)
            high = round(100 * (known + unknown.clip(min=0).sum()), 4)
        yield [records[0]['subject'], names[a], names[b], left.sum(), right.sum(), low, high]


if __name__ == '__main__':
    records = json.loads((FRAMES / 'manifest.json').read_text(encoding='utf8'))
    path = PREDICTIONS / 'paired_forehead_bounds.csv'
    with path.open('w', newline='', encoding='utf8') as output:
        writer = csv.writer(output)
        writer.writerow(['subject', 'method_a', 'method_b', 'selected_a', 'selected_b',
                         'min_difference_percentage_points', 'max_difference_percentage_points'])
        for subject in ['subject5', 'subject11']:
            subset = [r for r in records if r['subject'] == subject and r['status'] == 'ok']
            assert len(subset) == 6
            writer.writerows(compare(subset))
