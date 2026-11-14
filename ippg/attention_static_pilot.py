"""Inspect attention on an existing subject3 ROI at 72x72; static pilot."""
from pathlib import Path
import json,sys
import cv2,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path('third_party/tscan').resolve()))
from TS_CAN import TSCAN
out=Path('outputs/attention_preflight'); z=np.load(out/'map_verified.npz'); roi=np.load(out/'roi_masks.npz')
x1,y1,x2,y2=z['crop']; image=z['image'].astype(np.uint8)
expected=cv2.resize(cv2.cvtColor(roi['frame'][y1:y2,x1:x2],cv2.COLOR_BGR2RGB),(72,72))
if not np.array_equal(expected,image): raise ValueError('ROI and attention are not the same frame')
skin=cv2.resize(roi['skin'][y1:y2,x1:x2].astype(np.uint8),(72,72),interpolation=cv2.INTER_NEAREST).astype(bool)
valid=np.isfinite(z['attention']); support=skin & valid
if not support.any(): raise ValueError('No overlapping skin')
# Deterministic round spot on the largest central part of the skin mask.
distance=cv2.distanceTransform(support.astype(np.uint8),cv2.DIST_L2,5)
y,x=np.unravel_index(np.argmax(distance),distance.shape); yy,xx=np.ogrid[:72,:72]; spot=((xx-x)**2+(yy-y)**2<=4**2)&support
injected=image.copy(); injected[spot]=255
# Fixed per-frame normalization is an exploratory adaptation, not original full-clip preprocessing.
mean=float(image.mean());std=float(image.std()); model=TSCAN(frame_depth=10,img_size=72)
weights=torch.load('third_party/tscan/PURE_TSCAN.pth',map_location='cpu',weights_only=True)
model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True);model.eval()
def attention(img):
 v=torch.tensor((img.astype(np.float32)-mean)/std).permute(2,0,1).unsqueeze(0)
 with torch.no_grad():
  v=torch.tanh(model.apperance_conv1(v));v=torch.tanh(model.apperance_conv2(v));v=model.attn_mask_1(torch.sigmoid(model.apperance_att_conv1(v)))
 a=np.zeros((72,72),dtype=np.float32);a[1:-1,1:-1]=v[0,0].numpy();return a
clean=attention(image); modified=attention(injected)
def avg(img,weight):
 w=weight*support
 return np.sum(img*w[:,:,None],axis=(0,1))/w.sum()
uniform=avg(image,np.ones((72,72))); ordinary=image[support].mean(axis=0)
assert np.allclose(uniform,ordinary,atol=1e-10)
result={'frame':296,'support_skin_pixels_72':int(support.sum()),'excluded_border_skin_pixels':int((skin & ~valid).sum()),'spot_center_xy':[int(x),int(y)],'spot_pixels':int(spot.sum()),'attention_spot_clean_mean':float(clean[spot].mean()),'attention_spot_injected_mean':float(modified[spot].mean()),'attention_others_injected_mean':float(modified[support & ~spot].mean()),'baseline_rgb_clean':uniform.tolist(),'baseline_rgb_injected':avg(injected,np.ones((72,72))).tolist(),'attention_rgb_clean':avg(image,clean).tolist(),'attention_rgb_injected':avg(injected,modified).tolist(),'rule_rgb_clean':avg(image,(image.max(axis=2)<250).astype(float)).tolist(),'rule_rgb_injected':avg(injected,(injected.max(axis=2)<250).astype(float)).tolist(),'attention_spot_weight_share_injected':float(modified[spot].sum()/modified[support].sum()),'spot_area_share':float(spot.sum()/support.sum()),'notes':'72x72 static skin mask shared between conditions; white injected after mask; same frame normalization statistics; not full system or waveform test'}
(out/'static_pilot.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,img,title in zip(axes,[image,injected,injected],['Original','Injected spot','Attention on injected']):
 ax.imshow(img)
 if title.startswith('Attention'): ax.imshow(np.ma.masked_where(~support,modified),cmap='inferno',alpha=.6)
 ax.set_title(title);ax.axis('off')
fig.tight_layout();fig.savefig(out/'static_pilot.png',dpi=150)
print(json.dumps(result,indent=2))
