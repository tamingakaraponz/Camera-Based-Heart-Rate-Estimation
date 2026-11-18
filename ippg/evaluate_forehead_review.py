"""Compare completed forehead reviews with the four existing skin selections."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from roi_compare_reference import FRAMES, PREDICTIONS, ROOT, digest
from roi_experiment_metrics import evaluate


def score(record, folder):
    stem = Path(record['image']).stem
    metadata = folder / f'{stem}.json'
    annotation = folder / f'{stem}.png'
    if not metadata.exists() and not annotation.exists():
        return []
    meta = json.loads(metadata.read_text(encoding='utf8'))
    if meta['review_status'] != 'complete':
        return []
    if (not meta['annotator'].strip() or meta['subject'] != record['subject']
            or meta['frame_index'] != record['frame_index']
            or meta['polygons'] != {'A': record['polygons']['A']}
            or digest(annotation) != meta['labels_sha256']
            or digest(FRAMES / record['image']) != record['image_sha256']
            or record['image_sha256'] != meta['image_sha256']):
        raise ValueError(f'Review does not match source: {stem}')
    labels = cv2.imread(str(annotation), cv2.IMREAD_UNCHANGED)
    if labels is None or labels.shape != (record['height'], record['width']):
        raise ValueError(f'Invalid review image: {stem}')
    target = np.zeros_like(labels)
    cv2.fillPoly(target, [np.asarray(record['polygons']['A'], np.int32)], 1)
    paths = {name: PREDICTIONS / f'{stem}_A_{name}.png'
             for name in ('fixed', 'mediapipe', 'cheek_p90')}
    paths['color_only_p90'] = PREDICTIONS / 'cheek_color_experiment_masks' / f'{stem}_color_only_p90.png'
    rows = []
    for name, path in paths.items():
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(path)
        selected = mask > 0
        result = evaluate(labels, target, selected)
        if result['status'] != 'ok':
            raise ValueError(f'Incomplete review: {stem}')
        hair = int((selected & (labels == 2)).sum())
        count = result['selected']
        unknown = result['selected_unknown']
        rows.append([record['subject'], record['requested_seconds'], name, count,
                     result['tp'], hair, result['fp'] - hair, unknown,
                     100 * hair / count if count else None,
                     100 * unknown / count if count else None])
    return rows


if __name__ == '__main__':
    records = json.loads((FRAMES / 'manifest.json').read_text(encoding='utf8'))
    folder = ROOT / 'outputs' / 'forehead_human_review'
    records = [r for r in records if r['status'] == 'ok' and r['subject'] in ('subject5', 'subject11')]
    rows = [row for r in records for row in score(r, folder)]
    with (PREDICTIONS / 'forehead_human_comparison.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['subject', 'time_s', 'method', 'selected', 'skin', 'hair', 'other',
                         'unknown', 'hair_per_100_selected', 'unknown_per_100_selected'])
        writer.writerows(rows)
    print(f'Completed frames: {len(rows) // 4}/{len(records)}')
