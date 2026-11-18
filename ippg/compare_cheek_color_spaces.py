"""Exploratory cheek-color distances, evaluated only against AI draft labels."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from roi_compare_reference import ANNOTATIONS, FRAMES, PREDICTIONS, digest
from roi_experiment_metrics import evaluate


def main():
    manifest = json.loads((FRAMES / 'manifest.json').read_text(encoding='utf8'))
    rows = []
    mask_dir = PREDICTIONS / 'cheek_color_experiment_masks'
    mask_dir.mkdir(exist_ok=True)
    for record in manifest:
        if record['status'] != 'ok':
            continue
        stem = Path(record['image']).stem
        image_path = FRAMES / record['image']
        geometry_path = Path(record['geometry_source'])
        label_path = ANNOTATIONS / (stem + '.png')
        meta_path = ANNOTATIONS / (stem + '.json')
        if digest(image_path) != record['image_sha256'] or digest(geometry_path) != record['geometry_sha256']:
            raise ValueError(f'Input mismatch: {stem}')
        meta = None
        labels = None
        if label_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf8'))
            if (digest(label_path) != meta['labels_sha256'] or meta['image_sha256'] != record['image_sha256']
                    or meta['polygons'] != record['polygons']):
                raise ValueError(f'Annotation mismatch: {stem}')
            labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        frame = cv2.imread(str(image_path))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
        with np.load(geometry_path) as data:
            points = data['landmarks']
        radius = max(1, round(.02 * np.linalg.norm(points[263] - points[33])))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        masks = {}
        for name in 'ABC':
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.asarray(record['polygons'][name], np.int32)], 1)
            masks[name] = cv2.erode(mask, kernel) > 0 if name in 'BC' else mask > 0
        if min(masks['B'].sum(), masks['C'].sum()) < 20:
            continue
        samples = lab[masks['B'] | masks['C']]
        center = np.median(samples, axis=0)
        for mode, weights in [('lab', np.array([1.0, 1.0, 1.0])),
                              ('half_light', np.array([0.5, 1.0, 1.0])),
                              ('color_only', np.array([0.0, 1.0, 1.0]))]:
            sample_distance = np.linalg.norm((samples - center) * weights, axis=1)
            forehead_distance = np.linalg.norm((lab - center) * weights, axis=2)
            for percentile in (90, 99):
                threshold = float(np.percentile(sample_distance, percentile))
                chosen = (forehead_distance <= threshold) & masks['A']
                mask_name = f'{stem}_{mode}_p{percentile}.png'
                if not cv2.imwrite(str(mask_dir / mask_name), chosen.astype(np.uint8) * 255):
                    raise OSError(f'Cannot save {mask_name}')
                result = evaluate(labels, masks['A'], chosen) if labels is not None else {'status': 'reference_missing'}
                rows.append({'subject': record['subject'], 'time_s': record['requested_seconds'],
                             'frame_index': record['frame_index'], 'mode': mode, 'percentile': percentile,
                             'threshold': threshold, 'target_pixels': int(masks['A'].sum()),
                             'selected_pixels': int(chosen.sum()), 'mask': mask_name,
                             'annotation_status': meta['review_status'] if meta else 'missing', **result})
    out = PREDICTIONS / 'cheek_color_space_all.csv'
    with out.open('w', encoding='utf8', newline='') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Processed {len(rows)} combinations; scores require an available reference.')
    for row in rows:
        if row['time_s'] == 2 and row['percentile'] == 90:
            print(row['subject'], row['mode'], 'retention', row.get('retention'),
                  'false positives', row.get('fp'), 'selected unknown', row.get('selected_unknown'))


if __name__ == '__main__':
    main()
