"""Known-frequency checks for sinusoidal least-squares BPM scoring."""
from pathlib import Path
import csv
import numpy as np
rng=np.random.default_rng(42);rows=[]
for duration in (5,10,15):
 t=np.arange(0,duration,1/30)
 for true in (60.,80.,93.5,110.):
  for case in ('clean','phase_shift','inverted','noise','interference'):
   x=np.sin(2*np.pi*true/60*t+(.9 if case=='phase_shift' else 0))
   if case=='inverted':x=-x
   if case=='noise':x+=rng.normal(0,.5,len(t))
   if case=='interference':x+=.4*np.sin(2*np.pi*70/60*t+.4)
   best=(-1,None)
   for bpm in range(36,121):
    a=np.column_stack((np.sin(2*np.pi*bpm/60*t),np.cos(2*np.pi*bpm/60*t),np.ones(len(t))))
    prediction=a@np.linalg.lstsq(a,x,rcond=None)[0]
    score=1-np.sum((x-prediction)**2)/np.sum((x-x.mean())**2)
    if score>best[0]:best=(score,bpm)
   rows.append(dict(duration_s=duration,true_bpm=true,case=case,estimated_bpm=best[1],error=abs(best[1]-true),r2=best[0]))
out=Path('outputs/template_candidates');out.mkdir(exist_ok=True,parents=True)
with (out/'synthetic_checks.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
for case in ('clean','phase_shift','inverted','noise','interference'):
 r=[x for x in rows if x['case']==case];print(case,'worst BPM error',max(x['error'] for x in r))
assert all(x['error']<=.5 for x in rows if x['case'] in ('clean','phase_shift','inverted'))
