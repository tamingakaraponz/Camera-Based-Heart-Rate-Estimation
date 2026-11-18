"""Inspect fixed skin mask on six saved candidate ROIs; no pulse calculation."""
import json
import math
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ippg_v2_refactored as core

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'outputs' / 'roi_abc_annotation_pilot'
GEOMETRY = ROOT / 'outputs' / 'roi_overlay_abc_20261116_171220'


def main():
    entries = json.loads((PILOT / 'manifest.json').read_text(encoding='utf-8'))
    result = []
    for entry in entries:
        name = entry['subject']
        cap = cv2.VideoCapture(entry['video'])
        frame_index = math.ceil(2 * entry['fps'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f'Unable to read {name} frame {frame_index}')
        data = np.load(GEOMETRY / f'{name}_2s_points.npz')
        detector = core.AdaptiveSkinDetector(core.IPPGConfig(adaptive_skin_detection=False))
        colors = {'A': (0, 200, 255), 'B': (255, 170, 0), 'C': (60, 220, 60)}
        selected_view = frame.copy()
        geometry_view = frame.copy()
        counts = {}
        for label in 'ABC':
            polygon = data[label].astype(np.int32)
            target = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(target, [polygon], 255)
            selected = detector.create_mask(frame, target, label)
            selected_view[selected > 0] = colors[label]
            cv2.polylines(selected_view, [polygon], True, colors[label], 2)
            cv2.polylines(geometry_view, [polygon], True, colors[label], 2)
            counts[label] = {'target_pixels': int(np.count_nonzero(target)),
                             'selected_pixels': int(np.count_nonzero(selected))}
        points = data['landmarks']
        x1, y1 = np.maximum(np.floor(points.min(axis=0)).astype(int) - 20, 0)
        x2, y2 = np.minimum(np.ceil(points.max(axis=0)).astype(int) + 20,
                            [frame.shape[1], frame.shape[0]])
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
        for ax, image, title in zip(axes, (frame, geometry_view, selected_view),
                                    ('Original', 'Target ROI', 'Fixed filter selects')):
            ax.imshow(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'{name} | frame {frame_index} | A forehead, B/C cheeks')
        fig.savefig(PILOT / f'{name}_fixed_filter.png', dpi=150)
        plt.close(fig)
        result.append({'subject': name, 'frame_index': frame_index,
                       'timestamp_seconds': frame_index / entry['fps'], 'counts': counts})
    (PILOT / 'fixed_filter_pilot.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Six still-frame fixed-filter visualizations written.')


if __name__ == '__main__':
    main()
