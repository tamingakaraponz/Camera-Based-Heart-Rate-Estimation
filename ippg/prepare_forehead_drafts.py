"""Create partial AI forehead drafts after reviewing the original-frame montage."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / 'outputs/roi_review_frames'
OUTPUT = ROOT / 'outputs/roi_annotations'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = json.loads((FRAMES / 'manifest.json').read_text(encoding='utf8'))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    count = 0
    for record in records:
        if record['status'] != 'ok' or record['requested_seconds'] == 2:
            continue
        stem = Path(record['image']).stem
        path = OUTPUT / (stem + '.png')
        meta_path = OUTPUT / (stem + '.json')
        if path.exists() or meta_path.exists():
            print(f'{stem}: existing work retained')
            continue
        source = FRAMES / record['image']
        if sha256(source) != record['image_sha256']:
            raise ValueError(f'Changed source: {source}')
        height, width = record['height'], record['width']
        labels = np.full((height, width), 255, np.uint8)
        forehead = np.zeros((height, width), np.uint8)
        poly = np.asarray(record['polygons']['A'], dtype=np.int32)
        cv2.fillPoly(forehead, [poly], 1)
        subject = record['subject']
        if subject in ('subject5', 'subject11'):
            labels[forehead > 0] = 0
        else:
            labels[forehead > 0] = 1
        if subject == 'subject5':
            # Mark the visually clear right-hand portion only; retain the hair boundary as unknown.
            x_min, y_min = poly.min(axis=0)
            x_max, y_max = poly.max(axis=0)
            clear = np.zeros(labels.shape, np.uint8)
            x_start = int(round(x_min + .65 * (x_max - x_min)))
            cv2.rectangle(clear, (x_start, int(y_min)), (int(x_max), int(y_max)), 1, -1)
            labels[(clear > 0) & (forehead > 0)] = 1
        if not cv2.imwrite(str(path), labels):
            raise OSError(f'Cannot write {path}')
        union = np.zeros(labels.shape, np.uint8)
        for polygon in record['polygons'].values():
            cv2.fillPoly(union, [np.asarray(polygon, dtype=np.int32)], 1)
        metadata = {'image_sha256': record['image_sha256'], 'frame_index': record['frame_index'],
                    'subject': subject, 'polygons': record['polygons'],
                    'annotator': 'AI-assisted visual draft', 'review_status': 'draft',
                    'remaining': int(((union > 0) & (labels == 255)).sum()),
                    'labels_sha256': sha256(path), 'previously_seen_predictions': 'yes',
                    'annotation_version': 1,
                    'notes': 'Only forehead examined; B/C/D unlabelled. Approximate AI labels require review.'}
        meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf8')
        count += 1
    print(f'Created {count} partial forehead drafts; B/C/D remain unlabelled.')


if __name__ == '__main__':
    main()
