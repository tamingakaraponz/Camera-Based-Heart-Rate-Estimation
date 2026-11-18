"""MediaPipe multiclass overlay on six still frames; no pulse processing."""
import json, math, time
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/five_roi_methods_pilot'
BASE = ROOT / 'outputs/roi_overlay_abc_20261116_171220'


def main():
    model = ROOT / 'models/selfie_multiclass_256x256.tflite'
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        output_category_mask=True, output_confidence_masks=False)
    people = json.loads((ROOT/'outputs/roi_abc_annotation_pilot/manifest.json').read_text(encoding='utf8'))
    records=[]
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        for person in people:
            idx=math.ceil(2*person['fps']); cap=cv2.VideoCapture(person['video'])
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read();cap.release()
            if not ok: raise RuntimeError(f"Video read failed: {person['subject']}")
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            start=time.perf_counter(); result=segmenter.segment(image); ms=(time.perf_counter()-start)*1000
            classes=result.category_mask.numpy_view().copy()
            if classes.shape != frame.shape[:2]:
                classes=cv2.resize(classes,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)
            data=np.load(BASE/f"{person['subject']}_2s_points.npz")
            roi=np.zeros(frame.shape[:2],np.uint8)
            for key in 'ABC': cv2.fillPoly(roi,[data[key].astype(np.int32)],255)
            skin=(classes==3)&(roi>0); hair=(classes==1)&(roi>0)
            overlay=frame.copy();overlay[skin]=(80,220,70);overlay[hair]=(0,0,255)
            for key in 'ABC': cv2.polylines(overlay,[data[key].astype(np.int32)],True,(0,220,255),2)
            points=data['landmarks'];x1,y1=np.maximum(np.floor(points.min(axis=0)).astype(int)-20,0);x2,y2=np.minimum(np.ceil(points.max(axis=0)).astype(int)+20,[frame.shape[1],frame.shape[0]])
            fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained')
            for ax,img,title in zip(axs,[frame,overlay],['Original','Green: face skin; red: hair; yellow: ROI']):
                ax.imshow(cv2.cvtColor(img[y1:y2,x1:x2],cv2.COLOR_BGR2RGB));ax.set_title(title);ax.axis('off')
            fig.suptitle(person['subject']+' | multiclass single frame');fig.savefig(OUT/f"{person['subject']}_multiclass.png",dpi=140);plt.close(fig)
            records.append(dict(subject=person['subject'],frame_index=idx,inference_ms=ms,roi_face_skin=int(skin.sum()),roi_hair=int(hair.sum()),roi_pixels=int((roi>0).sum())))
    (OUT/'multiclass_results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    print(json.dumps(records,indent=2))


if __name__=='__main__': main()
