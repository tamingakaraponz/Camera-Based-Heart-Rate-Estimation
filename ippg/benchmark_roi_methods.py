"""Measure three ROI selection approaches in separate video replay passes."""
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import ippg_v2_refactored as core
from audit_roi_overlay_abc import polygons

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'dataset for ipph'
OUT = ROOT / 'outputs/roi_method_timing'
MODEL = ROOT / 'models/selfie_multiclass_256x256.tflite'
METHODS = ('fixed', 'mediapipe', 'cheek_p90')
SECONDS = 10
WARMUP = 30


def roi_masks(shape, points):
    result = {}
    for name, polygon in polygons(points).items():
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        result[name] = mask
    return result


def cheek_mask(frame, regions, points):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
    distance = float(np.linalg.norm(points[263] - points[33]))
    radius = max(1, round(.02 * distance))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    inner = {side: cv2.erode(regions[side], kernel) > 0 for side in 'BC'}
    if any(int(inner[side].sum()) < 20 for side in 'BC'):
        return None
    samples = lab[inner['B'] | inner['C']]
    center = np.median(samples, axis=0)
    threshold = np.percentile(np.linalg.norm(samples - center, axis=1), 90)
    return (np.linalg.norm(lab - center, axis=2) <= threshold) & (regions['A'] > 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    runs = []
    for person in (3, 5, 8, 9, 10, 11):
        video = list(DATA.glob(f'subject{person}-*/subject{person}/vid.avi'))
        if len(video) != 1:
            raise ValueError(f'subject{person}: expected one video, got {len(video)}')
        for method in METHODS:
            cap = cv2.VideoCapture(str(video[0]))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
                raise ValueError(f'Cannot open {video[0]}')
            tracker = core.MediaPipeRegionTracker(core.IPPGConfig(show_preview=False))
            detector = core.AdaptiveSkinDetector(core.IPPGConfig(adaptive_skin_detection=False))
            segmenter = None
            if method == 'mediapipe':
                options = mp.tasks.vision.ImageSegmenterOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
                    output_category_mask=True, output_confidence_masks=False)
                segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)
            wall_start, cpu_start = time.perf_counter(), time.process_time()
            frame_count = 0
            try:
                while frame_count / fps < SECONDS:
                    start = time.perf_counter()
                    ok, frame = cap.read()
                    decode = (time.perf_counter() - start) * 1000
                    if not ok:
                        break
                    start = time.perf_counter()
                    result = tracker.detect(frame, int(round(1000 * frame_count / fps)))
                    tracker_ms = (time.perf_counter() - start) * 1000
                    geometry_ms = selector_ms = 0.0
                    status = 'no_face'
                    total_start = start
                    if result is not None:
                        points = tracker._smoothed_points.copy()
                        start = time.perf_counter()
                        regions = roi_masks(frame.shape[:2], points)
                        geometry_ms = (time.perf_counter() - start) * 1000
                        start = time.perf_counter()
                        if method == 'fixed':
                            mask = detector.create_mask(frame, regions['A'], 'A')
                        elif method == 'mediapipe':
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            segmented = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                            classes = segmented.category_mask.numpy_view()
                            if classes.shape != frame.shape[:2]:
                                classes = cv2.resize(classes, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                            mask = (classes == 3) & (regions['A'] > 0)
                        else:
                            mask = cheek_mask(frame, regions, points)
                        selector_ms = (time.perf_counter() - start) * 1000
                        status = 'ok' if mask is not None else 'insufficient_reference'
                    total_ms = (time.perf_counter() - total_start) * 1000
                    rows.append({'subject': person, 'method': method, 'frame': frame_count,
                                 'warmup': frame_count < WARMUP, 'status': status,
                                 'decode_ms': decode, 'tracker_ms': tracker_ms,
                                 'geometry_ms': geometry_ms, 'selector_ms': selector_ms,
                                 'total_roi_ms': total_ms})
                    frame_count += 1
            finally:
                cap.release()
                tracker.close()
                if segmenter is not None:
                    segmenter.close()
            elapsed, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
            runs.append({'subject': person, 'method': method, 'fps': fps, 'frames': frame_count,
                         'wall_s': elapsed, 'process_cpu_s': cpu})
            print(f'subject{person} {method}: {frame_count} frames, {elapsed:.1f}s', flush=True)
    with (OUT / 'timings.csv').open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for run in runs:
        subset = [row for row in rows if row['subject'] == run['subject']
                  and row['method'] == run['method'] and not row['warmup']]
        entry = dict(run, measured_frames=len(subset),
                     unsuccessful=sum(row['status'] != 'ok' for row in subset))
        for field in ('decode_ms', 'tracker_ms', 'geometry_ms', 'selector_ms', 'total_roi_ms'):
            values = [row[field] for row in subset]
            entry[field + '_median'] = float(np.median(values)) if values else None
            entry[field + '_p95'] = float(np.percentile(values, 95)) if values else None
        summary.append(entry)
    metadata = {'python': platform.python_version(), 'platform': platform.platform(),
                'processor': platform.processor(), 'opencv': cv2.__version__,
                'mediapipe': mp.__version__, 'opencv_threads': cv2.getNumThreads(),
                'model_sha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                'seconds': SECONDS, 'warmup_frames': WARMUP,
                'scope': 'Offline sequential video replay, ROI A, one pass per method; excludes display and pulse processing.',
                'runs': summary}
    (OUT / 'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
