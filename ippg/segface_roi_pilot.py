"""SegFace MobileNet CPU inference on six stills, isolated from pulse pipeline."""
import json, math, sys, time
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/SegFace'))
import network.models.segface_celeb as implementation
from network import get_model

OUT = ROOT / 'outputs/five_roi_methods_pilot'
BASE = ROOT / 'outputs/roi_overlay_abc_20261116_171220'


def main():
    # The full SegFace checkpoint replaces the ImageNet initialization. Avoid
    # downloading unrelated backbone weights and require an exact state match.
    original = implementation.mobilenet_v3_large
    implementation.mobilenet_v3_large = lambda **kwargs: original(weights=None)
    model = get_model('segface_celeb', 512, 'mobilenet')
    checkpoint = torch.load(ROOT/'models/segface_mobilenet_celeba_512.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict_backbone'], strict=True)
    model.eval()
    people = json.loads((ROOT/'outputs/roi_abc_annotation_pilot/manifest.json').read_text(encoding='utf8'))
    records = []
    for person in people:
        idx=math.ceil(2*person['fps']);cap=cv2.VideoCapture(person['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ok,frame=cap.read();cap.release()
        if not ok: raise RuntimeError('Video read failed')
        data=np.load(BASE/f"{person['subject']}_2s_points.npz"); points=data['landmarks']
        # A face crop is necessary for webcam images; this is not a published
        # detector/crop protocol, so comparison requires validation.
        low=points.min(axis=0);high=points.max(axis=0); center=(low+high)/2
        side=float(max(high-low)*1.5); x1=max(0,int(center[0]-side/2));x2=min(frame.shape[1],int(center[0]+side/2))
        y1=max(0,int(center[1]-side/2));y2=min(frame.shape[0],int(center[1]+side/2))
        crop=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)
        image=cv2.resize(crop,(512,512),interpolation=cv2.INTER_CUBIC).astype(np.float32)/255
        image=(image-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        tensor=torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0)
        start=time.perf_counter()
        with torch.inference_mode():
            logits=model(tensor,None,None)
            classes=F.interpolate(logits,size=(512,512),mode='bilinear',align_corners=False).argmax(dim=1)[0].cpu().numpy().astype('uint8')
        ms=(time.perf_counter()-start)*1000
        classes=cv2.resize(classes,(x2-x1,y2-y1),interpolation=cv2.INTER_NEAREST)
        full=np.zeros(frame.shape[:2],np.uint8);full[y1:y2,x1:x2]=classes
        roi=np.zeros(frame.shape[:2],np.uint8)
        for key in 'ABC': cv2.fillPoly(roi,[data[key].astype(np.int32)],255)
        skin=(full==2)&(roi>0);hair=(full==14)&(roi>0)
        overlay=frame.copy();overlay[skin]=(80,220,70);overlay[hair]=(0,0,255)
        for key in 'ABC':cv2.polylines(overlay,[data[key].astype(np.int32)],True,(0,220,255),2)
        fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained')
        for ax,img,title in zip(axs,[frame,overlay],['Original','Green: skin; red: hair; yellow: ROI']):
            ax.imshow(cv2.cvtColor(img[y1:y2,x1:x2],cv2.COLOR_BGR2RGB));ax.set_title(title);ax.axis('off')
        fig.suptitle(person['subject']+' | SegFace MobileNet CPU');fig.savefig(OUT/f"{person['subject']}_segface.png",dpi=140);plt.close(fig)
        records.append(dict(subject=person['subject'],frame_index=idx,crop=[x1,y1,x2,y2],model_ms=ms,roi_skin=int(skin.sum()),roi_hair=int(hair.sum()),roi_pixels=int((roi>0).sum())))
        print(person['subject'],round(ms),flush=True)
    (OUT/'segface_results.json').write_text(json.dumps(records,indent=2),encoding='utf8')


if __name__ == '__main__':main()
