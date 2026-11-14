"""Five illustrative 5s cases for template-guided smoothing, seed fixed."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('outputs/template_guided_smoothing');OUT.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(42);t=np.arange(0,5,1/30);truth=np.sin(2*np.pi*80/60*t+.4)
cases=[('Mild noise',truth+.3*rng.normal(size=len(t)),80.),('Strong noise',truth+1.*rng.normal(size=len(t)),80.),('Stronger periodic interference',truth+1.8*np.sin(2*np.pi*105/60*t+.2),80.),('Changing rate',np.sin(2*np.pi*(70/60*t+(.5*(20/60)/5)*t*t)),None),('Noise only',rng.normal(size=len(t)),None)]
def fit(x):
 best=(-np.inf,None,None)
 for bpm in range(36,121):
  a=np.column_stack((np.sin(2*np.pi*bpm/60*t),np.cos(2*np.pi*bpm/60*t),np.ones(len(t))))
  y=a@np.linalg.lstsq(a,x,rcond=None)[0];r2=1-np.sum((x-y)**2)/np.sum((x-x.mean())**2)
  if r2>best[0]:best=(r2,bpm,y)
 return best
rows=[];fig,axes=plt.subplots(5,1,figsize=(12,13),layout='constrained')
for ax,(name,x,true) in zip(axes,cases):
 score,bpm,model=fit(x)
 ax.plot(t,x,color='#999999',lw=.8,label='Observed')
 for strength,color in [(.5,'#1070b0'),(.9,'#db6510')]:
  output=(1-strength)*x+strength*model
  newscore,newbpm,_=fit(output)
  rows.append(dict(case=name,true_bpm=true,raw_estimate=bpm,raw_fit_r2=score,template_weight=strength,output_estimate=newbpm,output_fit_r2=newscore,raw_bpm_error=abs(bpm-true) if true else None,output_bpm_error=abs(newbpm-true) if true else None))
  ax.plot(t,output,color=color,label=f'Template weight {strength}',lw=1.3)
 ax.set_title(f'{name} | original best BPM {bpm} | observed R2 {score:.2f}');ax.set_ylabel('Amplitude');ax.grid(alpha=.2);ax.legend(loc='upper right',ncol=3)
axes[-1].set_xlabel('Time (s)');fig.savefig(OUT/'five_examples.png',dpi=120)
with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2));print('Saved',OUT)
