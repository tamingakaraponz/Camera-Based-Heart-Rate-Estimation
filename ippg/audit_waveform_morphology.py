"""Measure recoverable harmonic structure before/after the current bandpass."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.signal import butter,sosfreqz
import ippg_v2_refactored as core
from replay_realtime import replay_video,TimeBuffer,evaluate
from compare_dev_methods import chrom
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/morphology_audit';OUT.mkdir(parents=True,exist_ok=True)
DATA=Path(r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph')

def fit_r2(t,x,f,harmonics):
 x=x-x.mean();cols=[np.ones(len(t))]
 for h in range(1,harmonics+1):cols.extend((np.sin(2*np.pi*h*f*t),np.cos(2*np.pi*h*f*t)))
 a=np.column_stack(cols);pred=a@np.linalg.lstsq(a,x,rcond=None)[0]
 return float(1-np.sum((x-pred)**2)/np.sum(x*x))

rows=[];failures=[]
for subject in ('subject3','subject5','subject8','subject9'):
 video=next(DATA.glob(f'{subject}-*/{subject}/vid.avi'));cfg=core.IPPGConfig(roi_method='legacy',pulse_method='pos',denoise='dwt',background_method='none',show_preview=False)
 meta={};buffer=TimeBuffer(15);tick=20.;stream=replay_video(video,cfg,meta)
 try:
  for obs in stream:
   while tick<obs.time_s-1e-9 and tick<=50:
    row,stages=evaluate(buffer,tick,15,meta['fps'],cfg)
    if row['status']!='estimated':failures.append(dict(subject=subject,end=tick,reason=row['reason']))
    else:
     try:
      c=chrom(stages['rgb'],meta['fps']);c=core.wavelet_denoise(c,cfg,meta['fps'])[0];cf=core.bandpass_filter(c,meta['fps'],.6,2,6)
      n=min(len(c),len(stages['filtered']));t=stages['time_seconds'][:n];t=t-t[0]; keep=(t>=1)&(t<=t[-1]-1)
      if keep.sum()<100:raise ValueError('short common support')
      tt=t[keep]
      for method,before,after in [('POS',stages['after_dwt'][:n],stages['filtered'][:n]),('CHROM',c[:n],cf[:n])]:
       y=after[keep];b=before[keep];freqs=np.arange(.6,2.0001,.005);scores=[fit_r2(tt,y,f,1) for f in freqs];i=int(np.argmax(scores));f=float(freqs[i]);r1=scores[i];r3=fit_r2(tt,y,f,3)
       sos=butter(6,[.6,2],fs=meta['fps'],btype='bandpass',output='sos');_,response=sosfreqz(sos,worN=np.array([f,2*f,3*f]),fs=meta['fps']);amp=np.abs(response)**2
       rows.append(dict(subject=subject,end_s=tick,method=method,samples=int(keep.sum()),bpm=f*60,fundamental_r2=r1,harmonic_extra_after=max(0.,r3-r1),harmonic_extra_before=max(0.,fit_r2(tt,b,f,3)-fit_r2(tt,b,f,1)),unexplained_after=1-r3,second_harmonic_amplitude_pass=float(amp[1]),third_harmonic_amplitude_pass=float(amp[2])))
     except ValueError as e:failures.append(dict(subject=subject,end=tick,reason=str(e)))
    tick+=5
   if tick>50:break
   buffer.append(obs)
 finally:stream.close()
 print('Finished',subject,flush=True)
with (OUT/'windows.csv').open('w',encoding='utf-8-sig',newline='') as fp:
 w=csv.DictWriter(fp,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(OUT/'manifest.json').write_text(json.dumps(dict(subjects=['subject3','subject5','subject8','subject9'],rows=len(rows),failures=failures,protocol='15s trailing windows at 20..50s, trim 1s each edge, f0 selected by best sinusoidal R2 0.6..2Hz step .005Hz; harmonic fit up to 3f0; before=after DWT, after=actual current bandpass',limits='In-sample fit, overlapping windows, CHROM already internally bandpassed; extra harmonic energy is not proof of physiology; no accuracy/PPG evaluation'),indent=2),encoding='utf-8')
print('Saved',OUT,len(rows),'rows')
