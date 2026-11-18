"""Compare POS with and without DWT on identical replay windows."""
import csv,json
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
from scipy.signal import butter,sosfiltfilt,find_peaks
import ippg_v2_refactored as core
from replay_realtime import TimeBuffer,replay_video,evaluate,load_reference
from benchmark_six_accuracy import bpm_fft

DATA=Path(r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph')
OUT=Path(__file__).resolve().parents[1]/'outputs'/('dwt_ab_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
def ppg_reference(ref,t,now,fps):
 times,values=ref;end=np.searchsorted(times,now,side='right');times=times[:end];values=values[:end]
 if len(times)<2:raise ValueError('no reference')
 uniq,inv,cnt=np.unique(times,return_inverse=True,return_counts=True);val=np.bincount(inv,weights=values)/cnt
 if uniq[0]>t[0] or uniq[-1]<t[-1]:raise ValueError('reference support incomplete')
 use=uniq[(uniq>=t[0])&(uniq<=t[-1])]
 if len(use)<2 or np.diff(use).max()>.25:raise ValueError('reference gap')
 raw=np.interp(t,uniq,val);sos=butter(3,[.6,3],btype='bandpass',fs=fps,output='sos');filtered=sosfiltfilt(sos,raw-raw.mean());peaks,_=find_peaks(filtered,distance=int(np.ceil(fps*.3)),prominence=.35*np.std(filtered))
 if len(peaks)<3:raise ValueError('fewer than three reference peaks')
 intervals=np.diff(t[peaks]);
 if np.any((intervals<.3)|(intervals>1/.6)):raise ValueError('reference peak intervals outside range')
 return raw,filtered,60/np.mean(intervals),peaks

def main():
 OUT.mkdir(parents=True); rows=[];fail=[];counts={}
 configs={name:core.IPPGConfig(roi_method='legacy',pulse_method='pos',denoise=mode,background_method='none',show_preview=False) for name,mode in [('dwt','dwt'),('no_dwt','none')]}
 for n in (3,5,8,9,10,11):
  subject=f'subject{n}';print('Processing',subject,flush=True)
  try:
   video=next(DATA.glob(f'{subject}-*/{subject}/vid.avi'));ref=load_reference(video.parent/'ground_truth.txt');meta={};buf=TimeBuffer(15);tick=20.;frames=0;stream=replay_video(video,configs['dwt'],meta)
   try:
    for obs in stream:
     frames+=1
     while tick<obs.time_s-1e-9 and tick<=50:
      for w in (5,10,15):
       try:
        outputs={};stages={}
        for name,cfg in configs.items():
         row,stage=evaluate(buf,tick,w,meta['fps'],cfg)
         if row['status']!='estimated':raise ValueError(name+': '+row['reason'])
         outputs[name]=row;stages[name]=stage
        t=stages['dwt']['time_seconds'];available=ref[0][ref[0]<=tick]
        if not len(available):raise ValueError('reference empty')
        count=min(len(t),np.searchsorted(t,available[-1],side='right'));t=t[:count]
        if len(t)<3:raise ValueError('short common interval')
        raw,ppg,refb,peaks=ppg_reference(ref,t,tick,meta['fps']); f=np.fft.rfftfreq(len(t),1/meta['fps']);mask=(f>=.6)&(f<=2);b=abs(np.fft.rfft(ppg-ppg.mean()))[mask]**2
        record=dict(subject=subject,end_s=tick,window_s=w,start_s=float(t[0]),actual_end_s=float(t[-1]),ppg_bpm=float(refb),ppg_peaks=len(peaks))
        arrays={'time':t,'ppg_raw':raw,'ppg_filtered':ppg,'ppg_peaks':peaks}
        for name in configs:
         x=stages[name]['filtered'][:count];a=abs(np.fft.rfft(x-x.mean()))[mask]**2
         if np.std(x)<1e-12 or a.sum()<=1e-20 or b.sum()<=1e-20:raise ValueError('flat signal')
         bpm=bpm_fft(x,meta['fps'],32)
         record.update({name+'_bpm':bpm,name+'_error':abs(bpm-refb),name+'_corr':float(np.corrcoef(x,ppg)[0,1]),name+'_overlap':float(np.minimum(a/a.sum(),b/b.sum()).sum())});arrays[name]=x
        rows.append(record);np.savez_compressed(OUT/f'{subject}_{int(tick)}_{w}.npz',**arrays)
       except (ValueError,RuntimeError) as e:fail.append(dict(subject=subject,end_s=tick,window=w,reason=str(e)))
      tick+=10
     if tick>50:break
     buf.append(obs)
   finally:stream.close()
   counts[subject]=frames
  except Exception as e:fail.append(dict(subject=subject,reason=repr(e)))
  print('Finished',subject,'successful',sum(x['subject']==subject for x in rows),'failures',sum(x['subject']==subject for x in fail),flush=True)
 if rows:
  with (OUT/'windows.csv').open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
 (OUT/'manifest.json').write_text(json.dumps(dict(counts=counts,failures=fail,note='Identical RGB and windows, only DWT changes. iPPG bandpass .6-2Hz. PPG reference bandpass .6-3Hz, peak min distance .3s and prominence .35 SD; exploratory, not validated. Nominal time, no shift/flip; windows overlap.'),indent=2),encoding='utf-8')
 print('OUTPUT',OUT,flush=True)
if __name__=='__main__':main()
