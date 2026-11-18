"""Partial, manually marked reference for subject11 frame 60; unknown elsewhere."""
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ippg_v2_refactored as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'roi_abc_annotation_pilot'
GEOMETRY = ROOT / 'outputs' / 'roi_overlay_abc_20261116_171220'


def main():
    entries = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    entry = next(x for x in entries if x['subject'] == 'subject11')
    capture = cv2.VideoCapture(entry['video'])
    capture.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError('Cannot read subject11 frame 60')
    # Mark only a conservative, visibly hair-covered interior. Everything else
    # is unknown; this is a diagnostic annotation, not a full ground truth.
    hair_polygon = np.array([[388, 235], [408, 238], [408, 246], [388, 243]], np.int32)
    hair = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(hair, [hair_polygon], 255)
    points = np.load(GEOMETRY / 'subject11_2s_points.npz')
    target = np.zeros_like(hair)
    cv2.fillPoly(target, [points['A'].astype(np.int32)], 255)
    detector = core.AdaptiveSkinDetector(core.IPPGConfig(adaptive_skin_detection=False))
    selected = detector.create_mask(frame, target, 'A')
    marked = (hair > 0) & (target > 0)
    incorrectly_selected = marked & (selected > 0)
    mask = np.zeros_like(hair)
    mask[hair > 0] = 1  # 1 = hair, 0 = unknown
    cv2.imwrite(str(OUT / 'subject11_frame60_partial_hair_mask.png'), mask)
    display = frame.copy()
    display[marked] = (0, 0, 255)
    cv2.polylines(display, [points['A'].astype(np.int32)], True, (0, 200, 255), 2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
    for ax, image, title in zip(axes, (frame, display), ('Original', 'Red: labelled hair inside A')):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_xlim(300, 460)
        ax.set_ylim(330, 170)
        ax.axis('off')
        ax.set_title(title)
    fig.savefig(OUT / 'subject11_partial_reference.png', dpi=150)
    plt.close(fig)
    result = {'subject':'subject11','frame_index':60,'annotation':'manual partial hair, not blinded; all other pixels unknown',
              'hair_polygon':hair_polygon.tolist(),'known_hair_pixels_inside_A':int(marked.sum()),
              'known_hair_selected_by_fixed_filter':int(incorrectly_selected.sum()),
              'target_pixels':int((target>0).sum()), 'known_skin_pixels':0}
    (OUT / 'subject11_partial_reference.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
