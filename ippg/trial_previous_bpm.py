"""Sequential synthetic test of template fitting with a soft previous-BPM prior."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('outputs/previous_bpm_trial');OUT.mkdir(parents=True,exist_ok=True)
FS=30;WINDOW=5;STEP=1;SEED=42
RATES=np.arange(36,121);PRIOR_STRENGTH=.1;PRIOR_SCALE=12.;MIN_R2=.15
rng=np.random.default_rng(SEED);t=np.arange(0,40,1/FS)
base=np.sin(2*np.pi*80/60*t+.4)
cases=[]
cases.append(('Constant HR with noise',np.full(len(t),80.),base+rng.normal(0,.65,len(t)),None))
y=base+rng.normal(0,.3,len(t));bad=(t>=15)&(t<22);y[bad]=rng.normal(0,1.2,bad.sum());cases.append(('Temporary signal loss',np.where(bad,np.nan,80.),y,None))
y=base+rng.normal(0,.25,len(t));bad=(t>=15)&(t<22);y[bad]+=1.8*np.sin(2*np.pi*105/60*t[bad]);cases.append(('Periodic interference',np.full(len(t),80.),y,None))
hr=np.where(t<20,80.,100.);phase=np.cumsum(hr/60/FS)*2*np.pi;y=np.sin(phase)+rng.normal(0,.3,len(t));cases.append(('Real HR change 80 to 100',hr,y,None))
cases.append(('Incorrect initial prior 105',np.full(len(t),80.),base+rng.normal(0,.4,len(t)),105.))
local=np.arange(WINDOW*FS)/FS
matrices=[]
for b in RATES:
 a=np.column_stack((np.sin(2*np.pi*b/60*local),np.cos(2*np.pi*b/60*local),np.ones(len(local))))
 matrices.append((a,np.linalg.pinv(a)))
rows=[];summaries=[];fig,axes=plt.subplots(5,1,figsize=(12,12),layout='constrained')
for ax,(name,true,y,initial) in zip(axes,cases):
 previous=initial;case_rows=[]
 for end in range(5,41):
  x=y[(end-WINDOW)*FS:end*FS];energy=np.sum((x-x.mean())**2)
  scores=np.array([1-np.sum((x-a@(pinv@x))**2)/energy for a,pinv in matrices])
  idx=int(np.argmax(scores));raw=float(RATES[idx]) if scores[idx]>=MIN_R2 else None
  if previous is None:regularized=scores
  else:regularized=scores-PRIOR_STRENGTH*np.minimum(((RATES-previous)/PRIOR_SCALE)**2,4.)
  j=int(np.argmax(regularized));chosen=float(RATES[j]) if scores[j]>=MIN_R2 else None
  if chosen is not None:previous=chosen
  # Evaluation label follows current endpoint, windows straddling changes remain marked.
  target=float(true[min(end*FS-1,len(true)-1)]);valid=np.isfinite(target)
  entry=dict(case=name,end_s=end,true_endpoint_bpm=target if valid else None,baseline_bpm=raw,prior_bpm=chosen,baseline_r2=float(scores[idx]),prior_observed_r2=float(scores[j]),previous_after=previous,baseline_error=abs(raw-target) if raw is not None and valid else None,prior_error=abs(chosen-target) if chosen is not None and valid else None)
  rows.append(entry);case_rows.append(entry)
 for method in ('baseline','prior'):
  errors=[r[method+'_error'] for r in case_rows if r[method+'_error'] is not None]
  summaries.append(dict(case=name,method=method,valid_estimates=sum(r[method+'_bpm'] is not None for r in case_rows),requested=len(case_rows),mae=float(np.mean(errors)) if errors else None))
 ax.plot(t,true,color='black',label='True HR',lw=1.5)
 for method,color in [('baseline','#999999'),('prior','#1270b0')]:ax.plot([r['end_s'] for r in case_rows],[np.nan if r[method+'_bpm'] is None else r[method+'_bpm'] for r in case_rows],color=color,label=method)
 ax.set_title(name);ax.set_ylabel('BPM');ax.grid(alpha=.2);ax.legend(ncol=3)
axes[-1].set_xlabel('Time (s)');fig.savefig(OUT/'five_cases.png',dpi=120)
for filename,data in [('windows.csv',rows),('summary.csv',summaries)]:
 with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
(OUT/'manifest.json').write_text(json.dumps(dict(seed=SEED,fs=FS,window=WINDOW,step=STEP,prior_strength=PRIOR_STRENGTH,prior_scale_bpm=PRIOR_SCALE,min_observed_r2=MIN_R2,prior_policy='Keep last accepted value indefinitely; no confidence aging implemented yet',limits='Synthetic sinusoidal cases, no video or ECG/PPG validation; parameters selected before this run but uncalibrated; endpoint labels include transition-window delay; noise-only labels unknown; no output template blending used to score'),indent=2),encoding='utf-8')
print(json.dumps(summaries,indent=2))
