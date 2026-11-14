import sys
from pathlib import Path
import cv2, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path('third_party/tscan').resolve()))
from TS_CAN import TSCAN
video=Path(r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph\subject3-20261113T112438Z-1-001\subject3\vid.avi')
cap=cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_MSEC,10000)
frames=[]
for _ in range(10):
 ok,im=cap.read()
 if not ok: raise ValueError('Cannot read frames')
 frames.append(im)
cap.release()
face=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
boxes=face.detectMultiScale(cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY),1.1,5)
if len(boxes)==0: raise ValueError('Face not detected')
x,y,w,h=max(boxes,key=lambda a:a[2]*a[3]); cx=x+w/2;cy=y+h/2;side=max(w,h)*1.5
x1=max(0,int(cx-side/2));x2=min(frames[0].shape[1],int(cx+side/2));y1=max(0,int(cy-side/2));y2=min(frames[0].shape[0],int(cy+side/2))
images=np.stack([cv2.resize(cv2.cvtColor(f[y1:y2,x1:x2],cv2.COLOR_BGR2RGB),(72,72)) for f in frames]).astype(np.float32)
standard=(images-images.mean())/(images.std()+1e-8)
model=TSCAN(frame_depth=10,img_size=72)
weights=torch.load('third_party/tscan/PURE_TSCAN.pth',map_location='cpu',weights_only=True)
model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True); model.eval()
with torch.no_grad():
 a=torch.from_numpy(standard).permute(0,3,1,2)
 r=torch.tanh(model.apperance_conv1(a));r=torch.tanh(model.apperance_conv2(r))
 mask=model.attn_mask_1(torch.sigmoid(model.apperance_att_conv1(r)))[0,0].numpy()
assert np.isfinite(mask).all()
# Verify the standalone appearance branch against the full model on identical input.
captured = []
hook = model.attn_mask_1.register_forward_hook(lambda module, inputs, output: captured.append(output.detach().clone()))
with torch.no_grad():
 model(torch.cat((torch.zeros_like(a), a), dim=1).contiguous())
hook.remove()
assert np.allclose(mask, captured[0][0,0].numpy(), atol=1e-6)
# The second 3x3 convolution has no padding: its 70x70 output is centred
# over pixels 1..70 of the 72x72 input. The outer border has no map value.
interior_map = np.full((72,72), np.nan, dtype=np.float32)
interior_map[1:-1,1:-1] = mask
np.savez_compressed('outputs/attention_preflight/map_verified.npz', attention=interior_map, crop=np.array([x1,y1,x2,y2]), image=images[0])
out=Path('outputs/attention_preflight');out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(1,2,figsize=(9,4)); ax[0].imshow(images[0].astype(np.uint8));ax[0].set_title('Face crop');ax[1].imshow(images[0].astype(np.uint8));ax[1].imshow(np.ma.masked_invalid(interior_map),alpha=.5,cmap='inferno');ax[1].set_title('Attention overlay (exploratory)')
for a in ax:a.axis('off')
fig.tight_layout();fig.savefig(out/'subject3_attention.png',dpi=130)
print('checkpoint loaded; full-model hook verified; map shape',mask.shape,'range',float(mask.min()),float(mask.max()),'crop',x1,y1,x2,y2)

