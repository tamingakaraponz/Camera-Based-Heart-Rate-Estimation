"""Prepare non-blind preliminary labels for review; never overwrite existing work."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / 'outputs/roi_review_frames'
OUT = ROOT / 'outputs/roi_annotations'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = json.loads((FRAMES / 'manifest.json').read_text(encoding='utf8'))
    OUT.mkdir(parents=True, exist_ok=True)
    for record in records:
        if record['requested_seconds'] != 2 or record['status'] != 'ok':
            continue
        name = Path(record['image']).stem
        label_path = OUT / f'{name}.png'
        meta_path = OUT / f'{name}.json'
        if label_path.exists() or meta_path.exists():
            print(f'{name}: existing work retained')
            continue
        image = cv2.imread(str(FRAMES / record['image']))
        if image is None or sha256(FRAMES / record['image']) != record['image_sha256']:
            raise ValueError(f'Image mismatch: {name}')
        labels = np.full(image.shape[:2], 255, np.uint8)
        for region in 'ABCD':
            mask = np.zeros(labels.shape, np.uint8)
            cv2.fillPoly(mask, [np.asarray(record['polygons'][region], dtype=np.int32)], 1)
            labels[mask > 0] = 1
        # These areas contain visible hair or ambiguous boundaries and need detailed review.
        ambiguous = {'subject5': 'A', 'subject11': 'AD'}.get(record['subject'], '')
        for region in ambiguous:
            mask = np.zeros(labels.shape, np.uint8)
            cv2.fillPoly(mask, [np.asarray(record['polygons'][region], dtype=np.int32)], 1)
            labels[mask > 0] = 0
        if record['subject'] == 'subject5':
            clear_skin = np.zeros(labels.shape, np.uint8)
            cv2.fillPoly(clear_skin, [
                np.array([[374, 164], [392, 164], [392, 174], [374, 174]], np.int32),
                np.array([[347, 171], [374, 171], [374, 174], [347, 174]], np.int32),
            ], 1)
            labels[(clear_skin > 0) & (labels == 0)] = 1
        if record['subject'] == 'subject11':
            # A small, visibly hair-covered part of the forehead, away from the boundary.
            hair = np.zeros(labels.shape, np.uint8)
            cv2.fillPoly(hair, [np.array([[405, 235], [420, 235], [420, 260], [405, 260]], np.int32)], 1)
            labels[(hair > 0) & (labels == 0)] = 2
        if not cv2.imwrite(str(label_path), labels):
            raise OSError(f'Cannot write {label_path}')
        meta = {'image_sha256': record['image_sha256'], 'frame_index': record['frame_index'],
                'subject': record['subject'], 'polygons': record['polygons'],
                'annotator': 'AI-assisted initial draft', 'review_status': 'draft', 'remaining': 0,
                'labels_sha256': sha256(label_path), 'previously_seen_predictions': 'yes',
                'annotation_version': 1,
                'notes': 'Initial visual estimate. Ambiguous regions marked unknown. Needs human review.'}
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf8')
        print(f'{name}: draft, ambiguous regions {ambiguous or "none marked"}')


if __name__ == '__main__':
    main()
