import sys
from pathlib import Path
import cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path('ippg').resolve()))
import ippg_v2_refactored as c
path=r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph\subject3-20261113T112438Z-1-001\subject3\vid.avi'
cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS); cfg=c.IPPGConfig(roi_method='legacy'); tracker=c._create_region_tracker(cfg); skin=c.AdaptiveSkinDetector(cfg)
regions=None
for idx in range(int(fps*10)+1):
 ok,frame=cap.read()
 if not ok: raise RuntimeError('Could not read frame')
 regions=tracker.detect(frame,int(round(idx*1000/fps)))
 if regions is not None:skin.calibrate(regions,(idx+1)/fps)
cap.release();tracker.close()
if regions is None or not skin.ready:raise RuntimeError('Face or calibration unavailable')
geometry=np.zeros_like(frame);selected=np.zeros_like(frame)
for label,color in [('forehead',(255,130,0)),('left_cheek',(0,180,0)),('right_cheek',(0,100,255))]:
 region=getattr(regions,label); geom=getattr(regions,label+'_geometry'); x1,y1,x2,y2=getattr(regions,label+'_box')
 _,mask,count=c.mean_skin_rgb(region,geom,label,skin,cfg)
 geometry[y1:y2,x1:x2][geom>0]=color
 selected[y1:y2,x1:x2][mask>0]=color
 print(label, 'selected_pixels',count)
np.savez_compressed('outputs/attention_preflight/roi_masks.npz', skin=np.any(selected != 0, axis=2), frame=frame)
fig,axes=plt.subplots(1,2,figsize=(13,6))
for ax,overlay,title in zip(axes,[geometry,selected],['1. Geometric ROI','2. Selected skin pixels']):
 blend=frame.copy(); valid=np.any(overlay!=0,axis=2);blend[valid]=(0.5*frame[valid]+0.5*overlay[valid]).astype(np.uint8)
 ax.imshow(cv2.cvtColor(blend,cv2.COLOR_BGR2RGB));ax.set_title(title);ax.axis('off')
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='#0082ff',label='Forehead'),Patch(color='#00b400',label='Left cheek'),Patch(color='#ff6400',label='Right cheek')],loc='lower center',ncol=3)
fig.suptitle(f'subject3 | frame {idx+1} | {(idx+1)/fps:.2f} s');fig.tight_layout(rect=(0,.06,1,.96))
out=Path('outputs/attention_preflight/subject3_roi.png');fig.savefig(out,dpi=120);print('Saved',out)
