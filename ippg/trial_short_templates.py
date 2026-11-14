"""Short-window exploratory comparison of templates and FFT, four subjects."""
from pathlib import Path
import csv,json
import numpy as np
import ippg_v2_refactored as core
from replay_realtime import TimeBuffer,replay_video,load_reference
from evaluate_waveform_dev import measure
from test_template_candidates import bank
from check_template_fairness import estimate
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/short_template_trial';OUT.mkdir(parents=True,exist_ok=True)
DATA=Path(r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph');rows=[];failures=[]
for subject in ('subject3','subject5','subject8','subject9'):
 video=next(DATA.glob(f'{subject}-*/{subject}/vid.avi'));reference=load_reference(video.parent/'ground_truth.txt');d=np.loadtxt(video.parent/'ground_truth.txt');device=(d[2]-d[2,0],d[1]);templates,counts=bank(subject)
 cfg=core.IPPGConfig(roi_method='legacy',pulse_method='pos',denoise='dwt',background_method='none',show_preview=False);buf=TimeBuffer(15);meta={};tick=20.;stream=replay_video(video,cfg,meta)
 try:
  for obs in stream:
   while tick<obs.time_s-1e-9 and tick<=50:
    for seconds in (5,10):
     try:
      t,vals,ppg,signals=measure(buf,tick,seconds,meta['fps'],cfg,reference,device);t=t-t[0]
      ref=vals[0]['ppg_fft_bpm'];provided=vals[0]['supplied_hr_mean']
      for val in vals:
       method=val['method'];signal=signals[method];estimates=[('FFT',val['ippg_bpm'],None)]
       # Same observed signal and timestamps; no PPG used to select templates.
       for name,shape in templates.items():
        b,score=estimate(shape,t,signal);estimates.append((name,b,score))
       for name,b,score in estimates:rows.append(dict(subject=subject,end_s=tick,window_s=seconds,method=method,estimator=name,bpm=b,reference_ppg_fft=ref,error_ppg_fft=abs(b-ref),supplied_hr=provided,error_supplied=abs(b-provided) if provided is not None else None,fit_r2=score,samples=len(t)))
     except (ValueError,RuntimeError) as e:failures.append(dict(subject=subject,end_s=tick,window_s=seconds,reason=str(e)))
    print(subject,'scored',tick,flush=True);tick+=10
   if tick>50:break
   buf.append(obs)
 finally:stream.close()
with (OUT/'windows.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(OUT/'manifest.json').write_text(json.dumps(dict(rows=len(rows),failures=failures,subjects=['subject3','subject5','subject8','subject9'],ticks=[20,30,40,50],windows=[5,10],notes='Empirical training templates from saved subject3/5/8 windows excluding test subject; subject9 uses all three. Ideal common-band projection, phase fitted per candidate, grid 1 BPM. References nominal clock; PPG FFT and supplied HR both recorded. Four development subjects, exploratory only.'),indent=2),encoding='utf-8');print('Saved',OUT,len(rows),len(failures))
