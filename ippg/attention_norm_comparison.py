"""Exploratory subject3 pixel weighting with a shared clean skin mask."""
import sys, time, csv, json
from pathlib import Path
sys.path.append(str(Path('.venv-tscan/Lib/site-packages').resolve()))
sys.path.insert(0,str(Path('third_party/tscan').resolve()))
import cv2,numpy as np,torch
from TS_CAN import TSCAN
import ippg_v2_refactored as core
from replay_realtime import TimeBuffer,Observation,load_reference
from evaluate_waveform_dev import measure

VIDEO=Path(sys.argv[1])
OUT=Path(sys.argv[2]); OUT.mkdir(parents=True,exist_ok=True)
stats=json.loads((OUT/'statistics.json').read_text(encoding='utf-8'))
torch.set_num_threads(1)
model=TSCAN(frame_depth=10,img_size=72)
w=torch.load('third_party/tscan/PURE_TSCAN.pth',map_location='cpu',weights_only=True)
model.load_state_dict({k.removeprefix('module.'):v for k,v in w.items()},strict=True); model.eval()
config=core.IPPGConfig(roi_method='legacy',pulse_method='pos',denoise='dwt',background_method='none',show_preview=False)
tracker=core._create_region_tracker(config); detector=core.AdaptiveSkinDetector(config)
cap=cv2.VideoCapture(str(VIDEO));fps=float(cap.get(cv2.CAP_PROP_FPS))
reference=load_reference(VIDEO.parent/'ground_truth.txt'); raw=np.loadtxt(VIDEO.parent/'ground_truth.txt');device=(raw[2]-raw[2,0],raw[1])
buffers={(case,method):TimeBuffer(15) for case in ('clean','spot') for method in ('baseline','rule','frame_norm','offline_norm')}
rows=[]; errors=[]; diagnostics=[]; end_times=(20.,25.,30.); next_tick=0

def map_for(frame,box,normalization=None):
 x1,y1,x2,y2=box; height,width=frame.shape[:2];cx=(x1+x2)/2;cy=(y1+y2)/2;side=max(x2-x1,y2-y1)*1.5
 l=max(0,int(cx-side/2));r=min(width,int(cx+side/2));t=max(0,int(cy-side/2));b=min(height,int(cy+side/2))
 face=cv2.resize(cv2.cvtColor(frame[t:b,l:r],cv2.COLOR_BGR2RGB),(72,72)).astype(np.float32)
 mean,sigma = (float(face.mean()),float(face.std())) if normalization is None else normalization
 if sigma<1e-8:raise ValueError('flat face')
 v=torch.from_numpy((face-mean)/sigma).permute(2,0,1).unsqueeze(0).contiguous()
 with torch.no_grad():
  v=torch.tanh(model.apperance_conv1(v));v=torch.tanh(model.apperance_conv2(v));v=model.attn_mask_1(torch.sigmoid(model.apperance_att_conv1(v)))
 # 70x70 corresponds to pixel centres 1..70; exclude outer border.
 a=np.zeros((72,72),np.float32);a[1:-1,1:-1]=v[0,0].numpy()
 valid=np.zeros((72,72),np.float32);valid[1:-1,1:-1]=1
 full=np.zeros(frame.shape[:2],np.float32); coverage=np.zeros_like(full)
 full[t:b,l:r]=cv2.resize(a,(r-l,b-t),interpolation=cv2.INTER_LINEAR)
 coverage[t:b,l:r]=cv2.resize(valid,(r-l,b-t),interpolation=cv2.INTER_LINEAR)
 full[coverage<.999]=0
 return full

def read_results(tick):
 for seconds in (5,10,15):
  for case,method in buffers:
   try:
    tt,values,ppg,signals=measure(buffers[(case,method)],tick,seconds,fps,config,reference,device)
    for v in values:rows.append(dict(end_s=tick,window_s=seconds,case=case,weighting=method,common_start=float(tt[0]),common_end=float(tt[-1]),**v))
   except (ValueError,RuntimeError) as e:errors.append(dict(end_s=tick,window_s=seconds,case=case,weighting=method,error=str(e)))

try:
 for i in range(int(fps*31)+2):
  ok,frame=cap.read()
  if not ok:break
  t=(i+1)/fps
  while next_tick<len(end_times) and end_times[next_tick]<t-1e-9:
   read_results(end_times[next_tick]);print('Scored',end_times[next_tick],flush=True);next_tick+=1
  if next_tick==len(end_times):break
  regions=tracker.detect(frame,int(round(i*1000/fps)))
  if regions is None:continue
  detector.calibrate(regions,t)
  if not detector.ready:continue
  mask=np.zeros(frame.shape[:2],dtype=bool)
  try:
   for name in ('forehead','left_cheek','right_cheek'):
    _,m,_=core.mean_skin_rgb(getattr(regions,name),getattr(regions,name+'_geometry'),name,detector,config)
    l,u,r,b=getattr(regions,name+'_box');mask[u:b,l:r]|=m>0
  except ValueError:continue
  # One moving spot within forehead; injected after clean tracker and skin mask.
  dirty=frame.copy();l,u,r,b=regions.forehead_box
  sx=int(l+(r-l)*(.5+.2*np.sin(2*np.pi*t/4)));sy=int(u+(b-u)*.5)
  spot=np.zeros(mask.shape,np.uint8)
  if t>=12:cv2.circle(spot,(sx,sy),12,1,-1)
  spot=spot.astype(bool)&mask;dirty[spot]=255
  try:
   maps={(case,kind):map_for(img,regions.face_box, None if kind=='frame_norm' else (stats[case]['mean'],stats[case]['std'])) for case,img in [('clean',frame),('spot',dirty)] for kind in ('frame_norm','offline_norm')}
   shared=mask.copy()
   for amap in maps.values(): shared &= amap > 0
   if shared.sum()<50:continue
   for case,img in [('clean',frame),('spot',dirty)]:
    rgb=img[:,:,::-1].astype(float)
    for method in ('baseline','rule','frame_norm','offline_norm'):
     weights=shared.astype(float)
     if method=='rule':weights*=np.max(rgb,axis=2)<250
     if method in ('frame_norm','offline_norm'):weights*=maps[(case,method)]
     total=weights.sum()
     if total<=0:continue
     val=np.sum(rgb*weights[:,:,None],axis=(0,1))/total
     buffers[(case,method)].append(Observation(t,val))
   for case in ('clean','spot'):
    a=maps[(case,'frame_norm')][shared];b=maps[(case,'offline_norm')][shared]
    chosen=spot&shared
    diagnostics.append(dict(time_s=t,case=case,skin_pixels=int(mask.sum()),common_pixels=int(shared.sum()),spot_pixels=int(chosen.sum()),map_corr=float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else None,map_abs_diff=float(np.mean(np.abs(a-b))),frame_spot_share=float(maps[(case,'frame_norm')][chosen].sum()/a.sum()) if chosen.any() else None,offline_spot_share=float(maps[(case,'offline_norm')][chosen].sum()/b.sum()) if chosen.any() else None))
  except (ValueError,RuntimeError) as e:errors.append(dict(time_s=t,error=str(e)))
finally:cap.release();tracker.close()
for filename,data in [('results.csv',rows),('errors.csv',errors),('frames.csv',diagnostics)]:
 if data:
  keys=list(dict.fromkeys(k for r in data for k in r))
  with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(data)
(OUT/'manifest.json').write_text(json.dumps(dict(rows=len(rows),errors=len(errors),frames=len(diagnostics),fps=fps,mode='same dynamic MediaPipe crop, per-frame vs 0-30s offline scalar statistics; not full original pipeline; controlled spot after fixed clean mask',ticks=end_times),indent=2),encoding='utf-8')
print('Done',len(rows),'rows',len(errors),'errors',len(diagnostics),'frames',flush=True)
