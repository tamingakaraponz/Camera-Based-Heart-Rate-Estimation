"""Compare prepared masks against completed, provenance-checked annotations."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from roi_experiment_metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / 'outputs/roi_review_frames'
PREDICTIONS = ROOT / 'outputs/roi_review_development'
ANNOTATIONS = ROOT / 'outputs/roi_annotations'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = json.loads((FRAMES / 'manifest.json').read_text(encoding='utf8'))
    lookup = {(r['subject'], r['frame_index']): r for r in records}
    with (PREDICTIONS / 'counts.csv').open(encoding='utf8', newline='') as source:
        predictions = list(csv.DictReader(source))
    rows = []
    for prediction in predictions:
        record = lookup[(prediction['subject'], int(prediction['frame_index']))]
        base = Path(record['image']).stem
        labels_path = ANNOTATIONS / (base + '.png')
        meta_path = ANNOTATIONS / (base + '.json')
        row = dict(prediction)
        if not prediction['mask']:
            row['evaluation_status'] = prediction['reference_status']
        elif not labels_path.exists() or not meta_path.exists():
            row['evaluation_status'] = 'reference_missing'
        else:
            meta = json.loads(meta_path.read_text(encoding='utf8'))
            image_path = FRAMES / record['image']
            if (digest(image_path) != record['image_sha256']
                or meta['image_sha256'] != record['image_sha256']
                or meta['labels_sha256'] != digest(labels_path)
                or meta['polygons'] != record['polygons']
                or meta['frame_index'] != record['frame_index']):
                row['evaluation_status'] = 'reference_mismatch'
            elif meta['review_status'] != 'complete':
                row['evaluation_status'] = 'reference_incomplete'
            else:
                labels = cv2.imread(str(labels_path), cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(str(PREDICTIONS / prediction['mask']), cv2.IMREAD_UNCHANGED)
                if labels is None or mask is None or labels.shape != (record['height'], record['width']):
                    row['evaluation_status'] = 'invalid_image'
                else:
                    target = np.zeros(labels.shape, dtype=np.uint8)
                    polygon = np.asarray(record['polygons'][prediction['region']], dtype=np.int32)
                    cv2.fillPoly(target, [polygon], 1)
                    try:
                        result = evaluate(labels, target, mask > 0)
                        row['evaluation_status'] = result.pop('status')
                        row.update(result)
                    except ValueError:
                        row['evaluation_status'] = 'invalid_mask_or_labels'
        rows.append(row)
    columns = list(rows[0]) + ['evaluation_status', 'tp', 'fp', 'fn', 'selected', 'selected_known',
                               'selected_unknown', 'unknown_fraction', 'contamination',
                               'contamination_reason', 'retention', 'iou']
    columns = list(dict.fromkeys(columns))
    with (PREDICTIONS / 'evaluation.csv').open('w', encoding='utf8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    from collections import Counter
    print('Evaluation status:', dict(Counter(row['evaluation_status'] for row in rows)))


if __name__ == '__main__':
    main()
