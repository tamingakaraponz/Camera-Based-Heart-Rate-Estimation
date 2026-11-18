"""Mark visually obvious hair cores in five AI draft frames; leave boundaries unknown."""
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
    for record in records:
        if record['subject'] != 'subject11' or record['requested_seconds'] == 2:
            continue
        stem = Path(record['image']).stem
        image = ANNOTATIONS / (stem + '.png')
        metadata = ANNOTATIONS / (stem + '.json')
        info = json.loads(metadata.read_text(encoding='utf8'))
        if info['review_status'] != 'draft' or info['annotator'] != 'AI-assisted visual draft':
            print(f'{stem}: retained existing work')
            continue
        if (digest(image) != info['labels_sha256'] or info['image_sha256'] != record['image_sha256']
                or digest(FRAMES / record['image']) != record['image_sha256']
                or info['polygons'] != record['polygons']):
            raise ValueError(f'Input mismatch: {stem}')
        labels = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        region = np.zeros(labels.shape, np.uint8)
        polygon = np.asarray(record['polygons']['A'], np.int32)
        cv2.fillPoly(region, [polygon], 1)
        right_edge = int(polygon[:, 0].max()) - 6
        cols = np.arange(labels.shape[1])[None, :]
        selected = (region > 0) & (cols >= right_edge) & (labels == 0)
        labels[selected] = 2
        if not cv2.imwrite(str(image), labels):
            raise OSError(f'Cannot write {image}')
        info['labels_sha256'] = digest(image)
        info['notes'] += ' AI visually marked a small dark hair core near the right ROI edge; boundary remains unknown.'
        metadata.write_text(json.dumps(info, indent=2), encoding='utf8')
        print(f'{stem}: marked {int(selected.sum())} more hair pixels')


if __name__ == '__main__':
    main()
