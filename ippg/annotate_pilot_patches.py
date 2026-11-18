"""Conservative hand-labelled skin/hair samples on six still frames.

These sparse patches are exploratory, not complete anatomical ROI masks.
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import ippg_v2_refactored as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'roi_abc_annotation_pilot'
# Rectangles [x1,y1,x2,y2], manually chosen from original image coordinate grid.
# Skin samples avoid edges, eyes, lips and uncertain hair. Unmarked = unknown.
PATCHES = {
    'subject3': {'skin': [[345,135,375,150], [310,190,335,205], [390,190,410,205]]},
    'subject5': {'skin': [[365,172,385,180], [310,225,335,240], [405,225,425,240]]},
    'subject8': {'skin': [[390,210,420,225], [345,275,370,290], [440,275,460,290]]},
    'subject9': {'skin': [[380,185,410,200], [340,245,360,260], [425,245,445,260]]},
    'subject10': {'skin': [[385,180,405,195], [345,240,365,255], [430,240,450,255]]},
    'subject11': {'skin': [[330,225,345,240], [335,290,355,305], [405,295,425,310]],
                  'hair': [[400,210,420,225]]},
}


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 10), layout='constrained')
    for ax, entry in zip(axes.flat, manifest):
        name = entry['subject']
        frame = cv2.imread(str(OUT / entry['image']))
        if frame is None:
            raise RuntimeError('Cannot read ' + entry['image'])
        labels = np.zeros(frame.shape[:2], dtype=np.uint8)
        for cls, rectangles in PATCHES[name].items():
            value = {'skin': 1, 'hair': 2}[cls]
            for x1, y1, x2, y2 in rectangles:
                labels[y1:y2, x1:x2] = value
        config = core.IPPGConfig(adaptive_skin_detection=False)
        detector = core.AdaptiveSkinDetector(config)
        full_mask = np.full(frame.shape[:2], 255, np.uint8)
        selected = detector.create_mask(frame, full_mask, name) > 0
        skin, hair = labels == 1, labels == 2
        rows.append({'subject': name, 'frame_index': entry['frame_index'],
                     'skin_labelled_pixels': int(skin.sum()),
                     'skin_selected_pixels': int((skin & selected).sum()),
                     'hair_labelled_pixels': int(hair.sum()),
                     'hair_selected_pixels': int((hair & selected).sum()),
                     'unknown_pixels': int((labels == 0).sum())})
        cv2.imwrite(str(OUT / f'{name}_sparse_reference.png'), labels)
        overlay = frame.copy()
        for cls, rectangles in PATCHES[name].items():
            for x1, y1, x2, y2 in rectangles:
                cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1),
                              (60, 220, 60) if cls == 'skin' else (0, 0, 255), 2)
        boxes = [r for rs in PATCHES[name].values() for r in rs]
        x1 = max(0, min(r[0] for r in boxes)-50); x2 = min(frame.shape[1], max(r[2] for r in boxes)+50)
        y1 = max(0, min(r[1] for r in boxes)-40); y2 = min(frame.shape[0], max(r[3] for r in boxes)+50)
        ax.imshow(cv2.cvtColor(overlay[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}, frame {entry['frame_index']}")
        ax.axis('off')
    fig.suptitle('Green: manually labelled skin | Red: manually labelled hair | outside: unknown')
    fig.savefig(OUT / 'six_sparse_references.png', dpi=150)
    plt.close(fig)
    (OUT / 'six_sparse_references.json').write_text(json.dumps({'patches':PATCHES,'results':rows}, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
