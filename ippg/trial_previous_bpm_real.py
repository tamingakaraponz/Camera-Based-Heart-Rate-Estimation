"""Replay 5s template fits with/without temporal prior, no PPG input to estimator."""
from pathlib import Path
import csv,json
import numpy as np
import ippg_v2_refactored as core
from replay_realtime import TimeBuffer,replay_video,evaluate,load_reference,reference_window
from compare_dev_methods import chrom,reference_hr
DATA=Path(r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph');OUT=Path('outputs/previous_bpm_real');OUT.mkdir(parents=True,exist_ok=True)
rows=[];fails=[];rates=np.arange(36,121)
for subject in ('subject3','subject5','subject8','subject9'):
 video=next(DATA.glob(f'{subject}-*/{subject}/vid.avi'));ref=load_reference(video.parent/'ground_truth.txt');d=np.loadtxt(video.parent/'ground_truth.txt');hrref=(d[2]-d[2,0],d[1]);prior={'POS':None,'CHROM':None}
 cfg=core.IPPGConfig(roi_method='legacy',pulse_method='pos',denoise='dwt',background_method='none',show_preview=False);buf=TimeBuffer(15);meta={};tick=8.;stream=replay_video(video,cfg,meta)
 try:
  for obs in stream:
   while tick<obs.time_s-1e-9 and tick<=50:
    row,stage=evaluate(buf,tick,5,meta['fps'],cfg)
    if row['status']=='estimated':
     try:
      c=chrom(stage['rgb'],meta['fps']);c=core.wavelet_denoise(c,cfg,meta['fps'])[0];c=core.bandpass_filter(c,meta['fps'],.6,2,6);n=min(len(c),len(stage['filtered']));grid=stage['time_seconds'][:n];t=grid-grid[0]
      # Estimation and prior update are done before loading reference values.
      results=[]
      for name,x in [('POS',stage['filtered'][:n]),('CHROM',c[:n])]:
       energy=np.sum((x-x.mean())**2);scores=[]
       for bpm in rates:
        a=np.column_stack([np.sin(2*np.pi*bpm/60*t),np.cos(2*np.pi*bpm/60*t),np.ones(len(t))]);fit=a@np.linalg.lstsq(a,x,rcond=None)[0];scores.append(1-np.sum((x-fit)**2)/energy)
       scores=np.array(scores);i=int(np.argmax(scores));old=prior[name];penalty=0 if old is None else .1*np.minimum(((rates-old)/12)**2,4);j=int(np.argmax(scores-penalty))
       base=float(rates[i]) if scores[i]>=.15 else None;chosen=float(rates[j]) if scores[j]>=.15 else None
       if chosen is not None:prior[name]=chosen
       results.append((name,x,base,chosen,old,float(scores[i]),float(scores[j])))
      try:
       ref_bpm,_,ppg,_=reference_window(ref,tick,grid,meta['fps'],cfg,.25);ref_reason=''
      except ValueError as e:ppg=None;ref_bpm=None;ref_reason=str(e)
      supplied=reference_hr(hrref,float(grid[0]),float(grid[-1]))
      for name,x,base,chosen,old,r2base,r2chosen in results:
       corr=overlap=None
       if ppg is not None and np.std(ppg)>1e-12 and np.std(x)>1e-12:
        corr=float(np.corrcoef(x,ppg)[0,1]);f=np.fft.rfftfreq(len(x),1/meta['fps']);mask=(f>=.6)&(f<=2);px=np.abs(np.fft.rfft(x-x.mean()))[mask]**2;py=np.abs(np.fft.rfft(ppg-ppg.mean()))[mask]**2
        if px.sum()>0 and py.sum()>0:overlap=float(np.minimum(px/px.sum(),py/py.sum()).sum())
       rows.append(dict(subject=subject,end_s=tick,method=name,baseline_bpm=base,prior_bpm=chosen,previous_bpm=old,baseline_r2=r2base,prior_observed_r2=r2chosen,ppg_fft=ref_bpm,supplied_hr=supplied,waveform_correlation=corr,spectral_overlap=overlap,reference_reason=ref_reason))
     except (ValueError,RuntimeError) as e:fails.append(dict(subject=subject,end_s=tick,reason=str(e)))
    else:fails.append(dict(subject=subject,end_s=tick,reason=row['reason']))
    tick+=1
   if tick>50:break
   buf.append(obs)
 finally:stream.close()
 print('Finished',subject,flush=True)
with (OUT/'windows.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(OUT/'manifest.json').write_text(json.dumps(dict(subjects=['subject3','subject5','subject8','subject9'],rows=len(rows),extraction_failures=fails,policy='5s, step1, score R2>=.15, soft penalty .1 min(((bpm-previous)/12)^2,4), old prior indefinitely; prior updated without reference',limits='Unpaced replay; crop CHROM support shorter; PPG time unverified; filtered waveform identical between estimators; no real webcam, no age or reacquisition policy'),indent=2),encoding='utf-8');print('Saved',len(rows),'rows',len(fails),'failures')
