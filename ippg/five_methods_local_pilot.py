"""Single-frame exploratory ROI overlays; no pulse processing."""
import json, math, time
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs' / 'roi_overlay_abc_20261116_171220'
OUT = ROOT / 'outputs' / 'five_roi_methods_pilot'


def roi_mask(shape, points):
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, [np.rint(points).astype(np.int32)], 255)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    people = json.loads((ROOT / 'outputs/roi_abc_annotation_pilot/manifest.json').read_text(encoding='utf8'))
    records = []
    for p in people:
        cap = cv2.VideoCapture(p['video'])
        idx = math.ceil(2 * p['fps'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read(); cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read {p['subject']} frame {idx}")
        data = np.load(BASE / f"{p['subject']}_2s_points.npz")
        a, b, c = [roi_mask(frame.shape[:2], data[k]) for k in 'ABC']
        start = time.perf_counter()
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
        samples = lab[(b > 0) | (c > 0)]
        # Exploratory calibration, not a validated skin classifier. Cheeks can
        # contain non-skin; the median and percentile do not guarantee purity.
        center = np.median(samples, axis=0)
        dist = np.linalg.norm(lab - center, axis=2)
        limit = float(np.percentile(np.linalg.norm(samples - center, axis=1), 90))
        chosen = (a > 0) & (dist <= limit)
        elapsed = (time.perf_counter() - start) * 1000
        points = data['landmarks']; ex = points[263] - points[33]
        scale = np.linalg.norm(ex); ex = ex / scale
        ey = np.array([-ex[1], ex[0]])
        if np.dot(ey, points[152] - points[10]) < 0: ey = -ey
        middle = (points[9] + points[168]) / 2
        d = np.array([middle-ex*scale*.07-ey*scale*.04, middle+ex*scale*.07-ey*scale*.04,
                      middle+ex*scale*.07+ey*scale*.04, middle-ex*scale*.07+ey*scale*.04])
        panels = [frame.copy() for _ in range(3)]
        for k, mask in zip('ABC', [a,b,c]):
            if k != 'A': panels[0][mask>0] = (120,220,80)
        panels[0][chosen] = (0,220,255)
        cv2.polylines(panels[0], [data['A'].astype(np.int32)], True, (0,220,255), 2)
        for mask in [a,b,c]: cv2.polylines(panels[1], cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0], -1, (80,200,80), 2)
        cv2.polylines(panels[1], [np.rint(d).astype(np.int32)], True, (0,0,255), 2)
        for mask in [b,c]: cv2.polylines(panels[2], cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0], -1, (80,200,80), 2)
        pts=points; x1,y1=np.maximum(np.floor(pts.min(axis=0)).astype(int)-20,0);x2,y2=np.minimum(np.ceil(pts.max(axis=0)).astype(int)+20,[frame.shape[1],frame.shape[0]])
        fig, axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax, image, title in zip(axs,panels,['Cheek-color candidate','Between brows D (red)','Cheeks only B/C']):
            ax.imshow(cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2RGB));ax.set_title(title);ax.axis('off')
        fig.suptitle(f"{p['subject']} | frame {idx} | exploratory only")
        fig.savefig(OUT/f"{p['subject']}.png",dpi=140);plt.close(fig)
        records.append(dict(subject=p['subject'],frame_index=idx,calibration='LAB median B+C, 90th percentile distance; unvalidated',
                            cheek_samples=len(samples),threshold=limit,forehead_target=int((a>0).sum()),
                            forehead_selected=int(chosen.sum()),color_step_ms=elapsed,d_polygon=np.rint(d).astype(int).tolist()))
    (OUT/'results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))


if __name__ == '__main__': main()
