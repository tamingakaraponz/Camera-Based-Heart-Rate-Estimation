"""Plot POS and contact PPG before BPM estimation at fixed replay windows."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ippg_v2_refactored as core
from replay_realtime import TimeBuffer,replay_video,evaluate,load_reference
DATA=Path(r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph');OUT=Path('outputs/signal_inspection');OUT.mkdir(parents=True,exist_ok=True)
def norm(x):
 x=np.asarray(x);std=x.std();return (x-x.mean())/std if std>1e-12 else x*0
summary=[]
for subject in ('subject3','subject9'):
 video=next(DATA.glob(f'{subject}-*/{subject}/vid.avi'));ref=load_reference(video.parent/'ground_truth.txt');cfg=core.IPPGConfig(roi_method='legacy',pulse_method='pos',denoise='dwt',background_method='none',show_preview=False);buf=TimeBuffer(15);meta={};ticks=[20.,30.,40.,50.];k=0;stream=replay_video(video,cfg,meta);fig,axes=plt.subplots(4,2,figsize=(15,12),layout='constrained')
 try:
  for obs in stream:
   while k<len(ticks) and ticks[k]<obs.time_s-1e-9:
    tick=ticks[k];row,s=evaluate(buf,tick,5,meta['fps'],cfg,ref);a,b=axes[k]
    if row['status']!='estimated':
     a.text(.1,.5,row['reason'],transform=a.transAxes);b.text(.1,.5,'Unavailable',transform=b.transAxes)
    else:
     t=s['time_seconds'];a.plot(t,norm(s['raw_pos']),label='POS raw',color='#9a65b4',lw=1);a.plot(t,norm(s['after_dwt']),label='After DWT',color='#336aab',lw=1);b.plot(t,norm(s['filtered']),label='POS filtered',color='#336aab',lw=1)
     if 'reference_filtered' in s:b.plot(t,norm(s['reference_filtered']),label='PPG filtered',color='#222222',lw=1,alpha=.8)
     else:b.text(.02,.08,'PPG unavailable: '+row['reference_reason'],transform=b.transAxes,fontsize=8)
     np.savez_compressed(OUT/f'{subject}_t{int(tick)}.npz',**s)
     summary.append(dict(subject=subject,end_s=tick,start_s=float(t[0]),last_s=float(t[-1]),ippg_bpm=row['ippg_bpm'],ppg_fft_bpm=row['reference_bpm'],reference_status=row['reference_status'],reference_reason=row['reference_reason']))
    a.set_title(f'Before bandpass | window ending {tick:g} s');b.set_title(f'After bandpass | window ending {tick:g} s')
    for ax in (a,b):ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=8);ax.set_ylabel('Standardized')
    k+=1
   if k==len(ticks):break
   buf.append(obs)
 finally:stream.close()
 axes[-1,0].set_xlabel('Video time (s)');axes[-1,1].set_xlabel('Nominal time (s)');fig.suptitle(f'{subject}: 5-second POS waveform before BPM estimation');fig.savefig(OUT/f'{subject}_signals.png',dpi=120);plt.close(fig);print('Done',subject,flush=True)
with (OUT/'windows.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
print('Saved',OUT)
