"""Plot saved six-subject evaluation and replay subject 11 signal stages."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import ippg_v2_refactored as core
from replay_realtime import TimeBuffer,replay_video,evaluate,load_reference,reference_window

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'outputs/six_accuracy_20261116_142246'; OUT=SRC/'subject11_diagnostics'; OUT.mkdir(exist_ok=True)
DATA=Path(r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph')

def z(x):
 x=np.asarray(x);return (x-np.mean(x))/np.std(x) if np.std(x)>1e-12 else np.zeros_like(x)
def overview():
 rows=list(csv.DictReader((SRC/'windows.csv').open(encoding='utf-8-sig')))
 fig,ax=plt.subplots(1,2,figsize=(14,5),layout='constrained')
 for w,marker,col in ((5,'o','#3174c5'),(10,'s','#ef9332'),(15,'^','#3b9f70')):
  data=[r for r in rows if int(r['window_s'])==w]
  ax[0].scatter([float(r['ppg_peak_bpm']) for r in data],[float(r['ippg_dense_bpm']) for r in data],label=f'{w} s',marker=marker,c=col,alpha=.7)
  for i,s in enumerate((3,5,8,9,10,11)):
   vals=[float(r['dense_abs_error']) for r in data if r['subject']==f'subject{s}']
   if vals: ax[1].scatter([i+(w-10)/25]*len(vals),vals,marker=marker,c=col,alpha=.65)
 ax[0].plot([35,125],[35,125],'k--',alpha=.6,label='Equal BPM');ax[0].set(xlabel='PPG peak BPM',ylabel='iPPG dense FFT BPM',title='All valid windows (70/72)')
 ax[1].set(xticks=range(6),xticklabels=['3','5','8','9','10','11'],xlabel='Subject',ylabel='Absolute difference (BPM)',title='BPM error by subject');ax[0].legend()
 for a in ax:a.grid(alpha=.2)
 fig.savefig(OUT/'six_subjects_overview.png',dpi=160);plt.close(fig)

def main():
 overview();video=next(DATA.glob('subject11-*/subject11/vid.avi'));ref=load_reference(video.parent/'ground_truth.txt');cfg=core.IPPGConfig(roi_method='legacy',pulse_method='pos',denoise='dwt',background_method='none',show_preview=False)
 buf=TimeBuffer(15);meta={};targets=[(20.,15),(40.,15),(50.,5)];index=0;stream=replay_video(video,cfg,meta);checks=[]
 try:
  for obs in stream:
   while index<len(targets) and targets[index][0]<obs.time_s-1e-9:
    tick,w=targets[index];row,s=evaluate(buf,tick,w,meta['fps'],cfg)
    if row['status']!='estimated':raise RuntimeError(row['reason'])
    t=s['time_seconds'];available=ref[0][ref[0]<=tick];n=min(len(t),int(np.searchsorted(t,available[-1],side='right')));t=t[:n];fs=meta['fps'];_,raw,ppg,_=reference_window(ref,tick,t,fs,cfg,.25)
    original=np.load(SRC/f'subject11_{int(tick)}_{w}.npz'); delta=float(np.max(np.abs(original['ippg']-s['filtered'][:n])));pdelta=float(np.max(np.abs(original['ppg']-ppg)))
    fig,axes=plt.subplots(6,1,figsize=(14,15),layout='constrained')
    rgb=s['rgb'][:n]
    for j,c in enumerate(('red','green','blue')):axes[0].plot(t,z(rgb[:,j]),color=c,alpha=.8,label='RGB'[j])
    axes[0].set_title('1. Skin RGB (standardized per channel)');axes[0].legend(loc='upper right',ncol=3)
    axes[1].plot(t,z(s['raw_pos'][:n]),color='#9065aa',label='POS raw');axes[1].set_title('2. POS raw')
    axes[2].plot(t,z(s['raw_pos'][:n]),color='#aaaaaa',alpha=.55,label='POS raw');axes[2].plot(t,z(s['after_dwt'][:n]),color='#3274b9',label='After DWT');axes[2].set_title('3. DWT denoising');axes[2].legend(loc='upper right')
    axes[3].plot(t,z(s['after_dwt'][:n]),color='#aaaaaa',alpha=.55,label='After DWT');axes[3].plot(t,z(s['filtered'][:n]),color='#3274b9',label='After bandpass');axes[3].set_title('4. Bandpass 0.6–2.0 Hz');axes[3].legend(loc='upper right')
    axes[4].plot(t,z(s['filtered'][:n]),label='iPPG filtered',color='#2877b9');axes[4].plot(t,z(ppg),label='PPG filtered',color='#df8230',alpha=.8)
    peaks,_=find_peaks(ppg,distance=int(np.ceil(fs*.5)),prominence=.35*np.std(ppg));axes[4].scatter(t[peaks],z(ppg)[peaks],color='#c33b32',s=16,label='PPG peaks')
    axes[4].set_title('5. Filtered iPPG vs PPG: no time shift or sign flip');axes[4].legend(loc='upper right')
    axes[5].plot(t,z(raw),color='#888888',label='PPG raw');axes[5].plot(t,z(ppg),color='#df8230',label='PPG filtered');axes[5].set_title('6. Reference PPG: raw and filtered');axes[5].legend(loc='upper right')
    for a in axes:a.grid(alpha=.2);a.set_ylabel('z-score')
    axes[-1].set_xlabel('Nominal video / PPG time (s)');fig.suptitle(f'Subject 11 | window ending {tick:g} s | {w} s | standardized signals')
    fig.savefig(OUT/f'subject11_stages_t{int(tick)}_{w}s.png',dpi=140);plt.close(fig)
    # Spectrum computed on exactly the same cropped sample arrays as the six-subject benchmark.
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained');f=np.fft.rfftfreq(n*32,1/fs)*60
    for x,label,col in ((s['filtered'][:n],'iPPG','#2877b9'),(ppg,'PPG','#df8230')):
     power=np.abs(np.fft.rfft(x-np.mean(x),n=n*32))**2;mask=(f>=36)&(f<=120);denom=power[mask].max();ax.plot(f[mask],power[mask]/denom,label=label,color=col)
    ax.set(xlabel='Frequency (BPM)',ylabel='Power / peak power',title=f'Subject 11 | dense spectrum | end {tick:g} s, {w} s');ax.legend();ax.grid(alpha=.2);fig.savefig(OUT/f'subject11_spectrum_t{int(tick)}_{w}s.png',dpi=150);plt.close(fig)
    np.savez_compressed(OUT/f'subject11_stages_t{int(tick)}_{w}s.npz',time=t,rgb=rgb,raw_pos=s['raw_pos'][:n],after_dwt=s['after_dwt'][:n],filtered=s['filtered'][:n],ppg_raw=raw,ppg_filtered=ppg)
    checks.append(dict(end=tick,window=w,max_ippg_delta=delta,max_ppg_delta=pdelta,ppg_peaks=len(peaks)));print('done',tick,w,'deltas',delta,pdelta,flush=True);index+=1
   if index==len(targets):break
   buf.append(obs)
 finally:stream.close()
 (OUT/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print('OUTPUT',OUT)
if __name__=='__main__':main()
