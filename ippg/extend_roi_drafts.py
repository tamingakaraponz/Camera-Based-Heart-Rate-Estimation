"""Extend AI draft annotations after examining the B/C/D source image montages."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / 'outputs/roi_review_frames'
ANNOTATIONS = ROOT / 'outputs/roi_annotations'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = json.loads((FRAMES / 'manifest.json').read_text(encoding='utf8'))
    changed = 0
    for record in records:
        if record['status'] != 'ok' or record['requested_seconds'] == 2:
            continue
        name = Path(record['image']).stem
        image = FRAMES / record['image']
        path = ANNOTATIONS / (name + '.png')
        meta_path = ANNOTATIONS / (name + '.json')
        if not path.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding='utf8'))
        if meta['review_status'] != 'draft' or meta['annotator'] != 'AI-assisted visual draft':
            print(f'{name}: not an AI draft, unchanged')
            continue
        if (digest(image) != record['image_sha256'] or digest(path) != meta['labels_sha256']
                or meta['image_sha256'] != record['image_sha256']
                or meta['polygons'] != record['polygons']):
            raise ValueError(f'Input mismatch: {name}')
        labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if labels is None or labels.shape != (record['height'], record['width']):
            raise ValueError(f'Invalid label image: {name}')
        for region in 'BCD':
            mask = np.zeros(labels.shape, np.uint8)
            cv2.fillPoly(mask, [np.asarray(record['polygons'][region], dtype=np.int32)], 1)
            value = 0 if record['subject'] == 'subject11' and region == 'D' else 1
            labels[(mask > 0) & (labels == 255)] = value
        if not cv2.imwrite(str(path), labels):
            raise OSError(f'Cannot write {path}')
        union = np.zeros(labels.shape, np.uint8)
        for polygon in record['polygons'].values():
            cv2.fillPoly(union, [np.asarray(polygon, dtype=np.int32)], 1)
        meta['remaining'] = int(((union > 0) & (labels == 255)).sum())
        meta['labels_sha256'] = digest(path)
        meta['notes'] = ('AI draft from visual review of source image montages, after prior exposure '
                         'to predictions. Subject11 D marked unknown. Not independently reviewed.')
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf8')
        changed += 1
    print(f'Extended {changed} AI draft label images. Human review is still pending.')


if __name__ == '__main__':
    main()
