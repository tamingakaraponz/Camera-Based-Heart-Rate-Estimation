"""Show selected forehead pixels beside AI draft labels for manual review."""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from roi_compare_reference import ANNOTATIONS, FRAMES, PREDICTIONS, digest


def main():
    records = json.loads((FRAMES / 'manifest.json').read_text(encoding='utf8'))
    records = [item for item in records if item['requested_seconds'] == 2 and item['status'] == 'ok']
    fig, axes = plt.subplots(len(records), 4, figsize=(14, 16), layout='constrained')
    for row, record in enumerate(records):
        stem = Path(record['image']).stem
        path = ANNOTATIONS / (stem + '.png')
        meta = json.loads((ANNOTATIONS / (stem + '.json')).read_text(encoding='utf8'))
        if (digest(path) != meta['labels_sha256']
                or digest(FRAMES / record['image']) != record['image_sha256']
                or meta['image_sha256'] != record['image_sha256']
                or meta['polygons'] != record['polygons']):
            raise ValueError(f'Image or annotation mismatch: {stem}')
        labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        image = cv2.imread(str(FRAMES / record['image']))
        polygon = np.asarray(record['polygons']['A'], np.int32)
        target = np.zeros(labels.shape, np.uint8)
        cv2.fillPoly(target, [polygon], 1)
        x1, y1 = np.maximum(polygon.min(axis=0) - 12, 0)
        x2, y2 = np.minimum(polygon.max(axis=0) + 13, [image.shape[1], image.shape[0]])
        methods = ('fixed', 'mediapipe', 'cheek_p90')
        pictures = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
        for method in methods:
            mask = cv2.imread(str(PREDICTIONS / f'{stem}_A_{method}.png'), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != labels.shape or np.any((mask > 0) & (target == 0)):
                raise ValueError(f'Missing or invalid prediction: {stem} {method}')
            selected = mask > 0
            visual = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            overlays = [(selected & (labels == 1), (0, 230, 70)),
                        (selected & ((labels == 2) | (labels == 3)), (255, 40, 40)),
                        ((target > 0) & ~selected & (labels == 1), (40, 120, 255)),
                        (selected & (labels == 0), (255, 210, 0))]
            for condition, color in overlays:
                visual[condition] = (visual[condition] * .35 + np.asarray(color) * .65).astype(np.uint8)
            cv2.polylines(visual, [polygon], True, (255, 255, 255), 1)
            pictures.append(visual)
        for col, (picture, title) in enumerate(zip(pictures, ('Original', 'Fixed color', 'MediaPipe', 'Cheek color P90'))):
            axes[row, col].imshow(picture[y1:y2, x1:x2], interpolation='nearest')
            axes[row, col].axis('off')
            axes[row, col].set_title(f"{record['subject']} | {title}", fontsize=9)
    fig.legend(handles=[Patch(color='#00e646', label='Selected: draft skin'),
                        Patch(color='#ff2828', label='Selected: draft hair/other'),
                        Patch(color='#2878ff', label='Draft skin not selected'),
                        Patch(color='#ffd200', label='Selected: unclear in draft')],
               loc='outside lower center', ncol=2, fontsize=9)
    fig.suptitle('Forehead at 2 seconds | AI draft, non-blind, requires review', fontsize=12)
    output = PREDICTIONS / 'forehead_draft_comparison.png'
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
