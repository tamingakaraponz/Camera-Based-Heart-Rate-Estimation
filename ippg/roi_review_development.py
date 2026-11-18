"""Generate exploratory selection masks on the prepared still frames."""
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import ippg_v2_refactored as core

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'outputs/roi_review_frames'
OUTPUT = ROOT / 'outputs/roi_review_development'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((INPUT / 'manifest.json').read_text(encoding='utf8'))
    model = ROOT / 'models/selfie_multiclass_256x256.tflite'
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        output_category_mask=True, output_confidence_masks=False)
    rows = []
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        for record in records:
            if record['status'] != 'ok':
                continue
            frame = cv2.imread(str(INPUT / record['image']))
            if frame is None:
                raise ValueError(f"Missing image {record['image']}")
            regions = {}
            for name, polygon in record['polygons'].items():
                mask = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillPoly(mask, [np.array(polygon, np.int32)], 255)
                regions[name] = mask
            detector = core.AdaptiveSkinDetector(core.IPPGConfig(adaptive_skin_detection=False))
            fixed = detector.create_mask(frame, np.full(frame.shape[:2], 255, np.uint8), 'whole_image') > 0
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            classes = result.category_mask.numpy_view().copy()
            if classes.shape != frame.shape[:2]:
                classes = cv2.resize(classes, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
            with np.load(record['geometry_source']) as geometry:
                points = geometry['landmarks']
                eye_distance = float(np.linalg.norm(points[263] - points[33]))
            radius = max(1, round(.02 * eye_distance))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            inner = {side: cv2.erode(regions[side], kernel) > 0 for side in 'BC'}
            cheek_counts = {side: int(inner[side].sum()) for side in 'BC'}
            enough = all(count >= 20 for count in cheek_counts.values())
            if enough:
                cheek_medians = {side: np.median(lab[inner[side]], axis=0) for side in 'BC'}
                cheek_difference = float(np.linalg.norm(cheek_medians['B'] - cheek_medians['C']))
                samples = lab[inner['B'] | inner['C']]
                median = np.median(samples, axis=0)
                sample_distance = np.linalg.norm(samples - median, axis=1)
                distance = np.linalg.norm(lab - median, axis=2)
            else:
                cheek_difference = None
            prefix = Path(record['image']).stem
            for region in 'ABCD':
                methods = {'fixed': fixed, 'mediapipe': classes == 3}
                if region == 'A' and enough:
                    for percentile in (80, 90, 95, 99):
                        methods[f'cheek_p{percentile}'] = distance <= np.percentile(sample_distance, percentile)
                for method, selection in methods.items():
                    chosen = selection & (regions[region] > 0)
                    path = OUTPUT / f'{prefix}_{region}_{method}.png'
                    if not cv2.imwrite(str(path), chosen.astype(np.uint8) * 255):
                        raise OSError(f'Cannot write {path}')
                    rows.append({'subject': record['subject'], 'frame_index': record['frame_index'],
                                 'region': region, 'method': method,
                                 'roi_pixels': int((regions[region] > 0).sum()),
                                 'selected_pixels': int(chosen.sum()), 'mask': path.name,
                                 'cheek_B_samples': cheek_counts['B'], 'cheek_C_samples': cheek_counts['C'],
                                 'erosion_radius': radius, 'cheek_difference_lab': cheek_difference,
                                 'reference_status': 'unvalidated' if enough else 'insufficient_reference'})
                if region == 'A' and not enough:
                    for percentile in (80, 90, 95, 99):
                        rows.append({'subject': record['subject'], 'frame_index': record['frame_index'],
                                     'region': region, 'method': f'cheek_p{percentile}',
                                     'roi_pixels': int((regions[region] > 0).sum()),
                                     'selected_pixels': '', 'mask': '',
                                     'cheek_B_samples': cheek_counts['B'], 'cheek_C_samples': cheek_counts['C'],
                                     'erosion_radius': radius, 'cheek_difference_lab': '',
                                     'reference_status': 'insufficient_reference'})
            print(prefix, flush=True)
    with (OUTPUT / 'counts.csv').open('w', encoding='utf8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} exploratory records. Cheek contamination has not been validated.')


if __name__ == '__main__':
    main()
