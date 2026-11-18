"""Evaluate six recordings with the existing replay POS pipeline against PPG."""
import csv, json
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.signal import find_peaks
import ippg_v2_refactored as core
from replay_realtime import TimeBuffer, replay_video, evaluate, load_reference, reference_window

DATA=Path(r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph')
OUT=Path(__file__).resolve().parents[1]/'outputs'/('six_accuracy_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
def peak_bpm(x,fps):
    sd=np.std(x)
    if sd<1e-12: raise ValueError('flat PPG')
    peaks,_=find_peaks(x,distance=int(np.ceil(fps*.5)),prominence=.35*sd)
    if len(peaks)<3: raise ValueError('fewer than 3 PPG peaks')
    intervals=np.diff(peaks)/fps
    if np.any((intervals<.5)|(intervals>1/.6)): raise ValueError('PPG peak interval outside range')
    return 60/float(np.mean(intervals))
def bpm_fft(x,fps,mult):
    f=np.fft.rfftfreq(len(x)*mult,1/fps); p=abs(np.fft.rfft(x-np.mean(x),n=len(x)*mult))**2
    mask=(f>=.6)&(f<=2)
    if not np.any(mask) or p[mask].max()<=1e-20: raise ValueError('no spectrum')
    return float(f[mask][np.argmax(p[mask])]*60)
def main():
    OUT.mkdir(parents=True,exist_ok=False); rows=[]; failures=[]; counts={}
    cfg=core.IPPGConfig(roi_method='legacy',pulse_method='pos',denoise='dwt',background_method='none',show_preview=False)
    for n in (3,5,8,9,10,11):
        subject=f'subject{n}'; print('Processing',subject,flush=True)
        try:
            matches=list(DATA.glob(f'{subject}-*/{subject}/vid.avi'))
            if len(matches)!=1: raise ValueError(f'found {len(matches)} videos')
            video=matches[0]; ref=load_reference(video.parent/'ground_truth.txt'); meta={}; buf=TimeBuffer(15); tick=20.; frames=0
            stream=replay_video(video,cfg,meta)
            try:
                for obs in stream:
                    frames+=1
                    while tick < obs.time_s-1e-9 and tick<=50:
                        for w in (5,10,15):
                            try:
                                row,st=evaluate(buf,tick,w,meta['fps'],cfg)
                                if row['status']!='estimated': raise ValueError(row['reason'])
                                t=st['time_seconds']; available=ref[0][ref[0]<=tick]
                                if len(available)==0: raise ValueError('no PPG available')
                                count=min(len(t),int(np.searchsorted(t,available[-1],side='right')))
                                if count<3: raise ValueError('short common support')
                                t=t[:count]; x=st['filtered'][:count]
                                _,_,ppg,_=reference_window(ref,tick,t,meta['fps'],cfg,.25)
                                fs=meta['fps']; refb=peak_bpm(ppg,fs)
                                if np.std(x)<1e-12 or np.std(ppg)<1e-12: raise ValueError('flat waveform')
                                f=np.fft.rfftfreq(len(x),1/fs); mask=(f>=.6)&(f<=2)
                                a=abs(np.fft.rfft(x-np.mean(x)))[mask]**2; b=abs(np.fft.rfft(ppg-np.mean(ppg)))[mask]**2
                                if a.sum()<=1e-20 or b.sum()<=1e-20: raise ValueError('zero band power')
                                normal=bpm_fft(x,fs,1); dense=bpm_fft(x,fs,32)
                                rows.append(dict(subject=subject,end_s=tick,window_s=w,start_s=float(t[0]),actual_end_s=float(t[-1]),ippg_fft_bpm=normal,ippg_dense_bpm=dense,ppg_peak_bpm=refb,fft_abs_error=abs(normal-refb),dense_abs_error=abs(dense-refb),correlation=float(np.corrcoef(x,ppg)[0,1]),spectral_overlap=float(np.minimum(a/a.sum(),b/b.sum()).sum())))
                                np.savez_compressed(OUT/f'{subject}_{int(tick)}_{w}.npz',time=t,ippg=x,ppg=ppg)
                            except (ValueError,RuntimeError) as e: failures.append(dict(subject=subject,end_s=tick,window_s=w,reason=str(e)))
                        tick+=10
                    if tick>50: break
                    buf.append(obs)
            finally: stream.close()
            counts[subject]=frames
        except Exception as e:
            failures.append(dict(subject=subject,reason=repr(e)))
        print('Finished',subject,'rows',sum(r['subject']==subject for r in rows),'failures',sum(f['subject']==subject for f in failures),flush=True)
    if rows:
        with (OUT/'windows.csv').open('w',newline='',encoding='utf-8-sig') as f:
            wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
    (OUT/'manifest.json').write_text(json.dumps(dict(subjects=[3,5,8,9,10,11],ticks=[20,30,40,50],windows=[5,10,15],counts=counts,failures=failures,reference='PPG peaks; prominence 0.35 SD, minimum interval 0.5 s, at least 3 peaks',note='Nominal synchronization, no lag/polarity adjustment; overlapping windows not independent; subject11 previously held out and now evaluated at user request'),indent=2),encoding='utf-8')
    print('OUTPUT',OUT,flush=True)
if __name__=='__main__':main()
