"""Exploratory comparison on manually labelled sparse pixels, not ROI accuracy."""
import json, time
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import ippg_v2_refactored as core

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'outputs/roi_abc_annotation_pilot'
OUT = ROOT / 'outputs/four_roi_evaluation'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    people = json.loads((PILOT/'manifest.json').read_text(encoding='utf8'))
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models/selfie_multiclass_256x256.tflite')),
        output_category_mask=True, output_confidence_masks=False)
    rows=[]
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        for item in people:
            frame=cv2.imread(str(PILOT/item['image']))
            labels=cv2.imread(str(PILOT/f"{item['subject']}_sparse_reference.png"),cv2.IMREAD_GRAYSCALE)
            if frame is None or labels is None or labels.shape != frame.shape[:2]:
                raise ValueError('Missing image/reference or shape mismatch')
            detector=core.AdaptiveSkinDetector(core.IPPGConfig(adaptive_skin_detection=False))
            fixed=detector.create_mask(frame,np.full(labels.shape,255,np.uint8),'whole_image')>0
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            t=time.perf_counter(); result=segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)); ms=(time.perf_counter()-t)*1000
            classes=result.category_mask.numpy_view().copy()
            if classes.shape != labels.shape: classes=cv2.resize(classes,(labels.shape[1],labels.shape[0]),interpolation=cv2.INTER_NEAREST)
            for method, selected in [('fixed_color',fixed),('mediapipe_multiclass',classes==3)]:
                skin=labels==1; hair=labels==2; known=skin|hair
                skin_n=int(skin.sum()); hair_n=int(hair.sum()); chosen_known=int((selected&known).sum())
                rows.append(dict(subject=item['subject'],frame=item['frame_index'],method=method,
                    skin_labelled=skin_n,skin_selected=int((selected&skin).sum()),
                    hair_labelled=hair_n,hair_selected=int((selected&hair).sum()),
                    unknown_selected=int((selected&~known).sum()),
                    known_selection_contamination=(int((selected&hair).sum())/chosen_known if chosen_known else None),
                    note='Sparse patches, nonblind, whole image, not per-ROI accuracy',
                    single_image_segment_ms=ms if method=='mediapipe_multiclass' else None))
    (OUT/'sparse_results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
