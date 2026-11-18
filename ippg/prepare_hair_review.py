"""Create a non-blind visual review pack for uncertain forehead labels."""
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from roi_compare_reference import ANNOTATIONS, FRAMES, PREDICTIONS, digest


def main():
    records = json.loads((FRAMES / 'manifest.json').read_text(encoding='utf8'))
    records = [r for r in records if r['subject'] in ('subject5', 'subject11')
               and r['status'] == 'ok']
    fig, axes = plt.subplots(len(records), 3, figsize=(12, 25), layout='constrained')
    rows = []
    for i, record in enumerate(records):
        stem = Path(record['image']).stem
        label_path = ANNOTATIONS / (stem + '.png')
        meta = json.loads((ANNOTATIONS / (stem + '.json')).read_text(encoding='utf8'))
        if (digest(FRAMES / record['image']) != record['image_sha256']
                or digest(label_path) != meta['labels_sha256']
                or meta['image_sha256'] != record['image_sha256']
                or meta['polygons'] != record['polygons']):
            raise ValueError(f'Mismatched inputs: {stem}')
        labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        bgr = cv2.imread(str(FRAMES / record['image']))
        original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        region = np.zeros(labels.shape, np.uint8)
        polygon = np.asarray(record['polygons']['A'], np.int32)
        cv2.fillPoly(region, [polygon], 1)
        uncertain = (labels == 0) & (region > 0)
        mask_paths = [
            PREDICTIONS / f'{stem}_A_fixed.png',
            PREDICTIONS / f'{stem}_A_mediapipe.png',
            PREDICTIONS / f'{stem}_A_cheek_p90.png',
            PREDICTIONS / 'cheek_color_experiment_masks' / f'{stem}_color_only_p90.png',
        ]
        masks = []
        for path in mask_paths:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.shape != labels.shape:
                raise ValueError(f'Missing or invalid mask: {path}')
            masks.append(image > 0)
        votes = np.sum(masks, axis=0)
        disagreement = uncertain & (votes > 0) & (votes < len(masks))
        draft = original.copy()
        for value, color in [(0, (255, 210, 0)), (1, (0, 220, 80)), (2, (255, 50, 50))]:
            area = (region > 0) & (labels == value)
            draft[area] = (draft[area] * .35 + np.asarray(color) * .65).astype(np.uint8)
        differences = original.copy()
        differences[uncertain] = (differences[uncertain] * .5 + np.array([255, 210, 0]) * .5).astype(np.uint8)
        differences[disagreement] = np.array([240, 40, 210], np.uint8)
        for view in (original, draft, differences):
            cv2.polylines(view, [polygon], True, (255, 255, 255), 1)
        x1, y1 = np.maximum(polygon.min(axis=0) - 8, 0)
        x2, y2 = np.minimum(polygon.max(axis=0) + 9, [labels.shape[1], labels.shape[0]])
        for ax, view, title in zip(axes[i], (original, draft, differences),
                                   ('Original', 'AI draft', 'Uncertain / differing methods')):
            ax.imshow(view[y1:y2, x1:x2], interpolation='nearest')
            ax.set_title(f"{record['subject']} {record['requested_seconds']}s | {title}", fontsize=8)
            ax.axis('off')
        rows.append({'subject': record['subject'], 'time_s': record['requested_seconds'],
                     'frame_index': record['frame_index'], 'unknown_in_forehead': int(uncertain.sum()),
                     'unknown_with_method_disagreement': int(disagreement.sum()),
                     'label_sha256': meta['labels_sha256']})
    fig.suptitle('Review pack | yellow = uncertain, magenta = uncertain and methods differ | AI draft, non-blind', fontsize=11)
    fig.savefig(PREDICTIONS / 'hair_review_pack.png', dpi=150)
    plt.close(fig)
    with (PREDICTIONS / 'hair_review_queue.csv').open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Prepared {len(rows)} review frames. Disagreement is for prioritization only.')


if __name__ == '__main__':
    main()
