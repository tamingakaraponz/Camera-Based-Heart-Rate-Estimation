"""Exploratory comparisons with AI drafts, never a final accuracy evaluation."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from roi_compare_reference import ANNOTATIONS, FRAMES, PREDICTIONS, digest
from roi_experiment_metrics import evaluate


def main():
    frames = json.loads((FRAMES / 'manifest.json').read_text(encoding='utf8'))
    lookup = {(r['subject'], r['frame_index']): r for r in frames}
    with (PREDICTIONS / 'counts.csv').open(encoding='utf8', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    rows = []
    for item in predictions:
        record = lookup[(item['subject'], int(item['frame_index']))]
        if not item['mask']:
            continue
        stem = Path(record['image']).stem
        label_path = ANNOTATIONS / (stem + '.png')
        meta_path = ANNOTATIONS / (stem + '.json')
        if not label_path.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding='utf8'))
        if (meta['image_sha256'] != record['image_sha256']
                or digest(FRAMES / record['image']) != record['image_sha256']
                or digest(label_path) != meta['labels_sha256']
                or meta['polygons'] != record['polygons']
                or meta['frame_index'] != record['frame_index']):
            raise ValueError(f'Mismatched reference: {stem}')
        labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(PREDICTIONS / item['mask']), cv2.IMREAD_UNCHANGED)
        if labels is None or labels.shape != (record['height'], record['width']) or mask is None:
            raise ValueError(f'Invalid image: {stem}')
        target = np.zeros(labels.shape, np.uint8)
        cv2.fillPoly(target, [np.asarray(record['polygons'][item['region']], np.int32)], 1)
        result = evaluate(labels, target, mask > 0)
        rows.append({'subject': item['subject'], 'frame_index': item['frame_index'],
                     'time_s': record['requested_seconds'],
                     'region': item['region'], 'method': item['method'],
                     'annotation_status': meta['review_status'], 'annotator': meta['annotator'],
                     **result})
    if not rows:
        raise ValueError('No draft references available')
    output = PREDICTIONS / 'draft_comparison.csv'
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with output.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} exploratory rows to {output}')


if __name__ == '__main__':
    main()
