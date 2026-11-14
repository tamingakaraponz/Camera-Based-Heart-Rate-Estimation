"""Collect offline appearance normalization statistics for a fixed 31s pilot."""
import json, sys
from pathlib import Path
import cv2,numpy as np
import ippg_v2_refactored as core
video=Path(sys.argv[1])
cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);cfg=core.IPPGConfig(roi_method='legacy',show_preview=False);tracker=core._create_region_tracker(cfg);detector=core.AdaptiveSkinDetector(cfg)
acc={name:dict(sum=0.,sum_sq=0.,n=0) for name in ('clean','spot')}
try:
 for i in range(int(fps*31)+2):
  ok,frame=cap.read()
  if not ok:break
  t=(i+1)/fps
  if t>30:break
  regions=tracker.detect(frame,int(round(i*1000/fps)))
  if regions is None:continue
  detector.calibrate(regions,t)
  if not detector.ready:continue
  mask=np.zeros(frame.shape[:2],dtype=bool)
  try:
   for name in ('forehead','left_cheek','right_cheek'):
    _,m,_=core.mean_skin_rgb(getattr(regions,name),getattr(regions,name+'_geometry'),name,detector,cfg)
    l,u,r,b=getattr(regions,name+'_box');mask[u:b,l:r]|=m>0
  except ValueError:continue
  l,u,r,b=regions.forehead_box;sx=int(l+(r-l)*(.5+.2*np.sin(2*np.pi*t/4)));sy=int(u+(b-u)*.5)
  spot=np.zeros(mask.shape,np.uint8)
  if t>=12:cv2.circle(spot,(sx,sy),12,1,-1)
  dirty=frame.copy();dirty[spot.astype(bool)&mask]=255
  x1,y1,x2,y2=regions.face_box;cx=(x1+x2)/2;cy=(y1+y2)/2;side=max(x2-x1,y2-y1)*1.5
  l=max(0,int(cx-side/2));r=min(frame.shape[1],int(cx+side/2));u=max(0,int(cy-side/2));b=min(frame.shape[0],int(cy+side/2))
  for name,img in [('clean',frame),('spot',dirty)]:
   a=cv2.resize(cv2.cvtColor(img[u:b,l:r],cv2.COLOR_BGR2RGB),(72,72)).astype(np.float64);acc[name]['sum']+=float(a.sum());acc[name]['sum_sq']+=float(np.square(a).sum());acc[name]['n']+=a.size
finally:cap.release();tracker.close()
for z in acc.values():
 z['mean']=z['sum']/z['n'];z['std']=float(np.sqrt(z['sum_sq']/z['n']-z['mean']**2))
out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);(out/'statistics.json').write_text(json.dumps(acc,indent=2),encoding='utf-8');print(json.dumps(acc,indent=2))
