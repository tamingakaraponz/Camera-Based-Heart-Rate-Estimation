"""Paired old/new filter and BPM search ranges on six recorded subjects."""
import csv,json
from datetime import datetime
from pathlib import Path
import numpy as np
import ippg_v2_refactored as core
from replay_realtime import TimeBuffer,replay_video,evaluate,load_reference
from benchmark_dwt_ab import ppg_reference

DATA=Path(r'C:\Users\Akkarapon\Downloads\Project scius\dataset for ipph')
ROOT=Path(__file__).resolve().parents[1]
def estimate(t,x,fs,high,dense=False):
    f,s=core.calculate_spectrum(t,x,fs)
    if dense:
        uniform=np.arange(t[0],t[-1],1/fs);v=np.interp(uniform,t,x);v-=v.mean()
        s=np.fft.rfft(v,n=len(v)*32);f=np.fft.rfftfreq(len(v)*32,1/fs)
    return core.estimate_heart_rate(f,s,.6,high)
def main():
    out=ROOT/'outputs'/('band_ab_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True)
    cfgs={name:core.IPPGConfig(roi_method='legacy',pulse_method='pos',denoise='none',background_method='none',show_preview=False,forehead_high_hz=high) for name,high in [('old',2.),('new',3.)]}
    rows=[];failures=[];clips=[]
    for num in (3,5,8,9,10,11):
        subject=f'subject{num}';print('Processing',subject,flush=True)
        try:
            videos=list(DATA.glob(f'{subject}-*/{subject}/vid.avi'))
            if len(videos)!=1:raise ValueError(f'Expected one video, found {len(videos)}')
            video=videos[0];reference=load_reference(video.parent/'ground_truth.txt');buf=TimeBuffer(15);meta={};tick=20.;stream=replay_video(video,cfgs['new'],meta);frames=0
            try:
                for obs in stream:
                    frames+=1
                    while tick<obs.time_s-1e-9 and tick<=50:
                        for seconds in (5,10,15):
                            try:
                                stages={}
                                for name,cfg in cfgs.items():
                                    status,stage=evaluate(buf,tick,seconds,meta['fps'],cfg)
                                    if status['status']!='estimated':raise ValueError(name+': '+status['reason'])
                                    stages[name]=stage
                                assert np.array_equal(stages['old']['rgb'],stages['new']['rgb'])
                                assert np.array_equal(stages['old']['raw_pos'],stages['new']['raw_pos'])
                                t=stages['new']['time_seconds'];available=reference[0][reference[0]<=tick]
                                if not len(available):raise ValueError('reference empty')
                                n=min(len(t),np.searchsorted(t,available[-1],side='right'));t=t[:n]
                                if len(t)<3:raise ValueError('short common support')
                                raw_ppg,ppg,ref_bpm,peaks=ppg_reference(reference,t,tick,meta['fps']);fs=meta['fps']
                                row=dict(subject=subject,end_s=tick,window_s=seconds,actual_start_s=float(t[0]),actual_end_s=float(t[-1]),ppg_bpm=float(ref_bpm),ppg_raw_fft=estimate(t,raw_ppg,fs,3,True))
                                arrays=dict(time=t,rgb=stages['new']['rgb'][:n],pos_raw=stages['new']['raw_pos'][:n],ppg_raw=raw_ppg,ppg_filtered=ppg)
                                f=np.fft.rfftfreq(len(t),1/fs);mask=(f>=.6)&(f<=3);rp=abs(np.fft.rfft(ppg-ppg.mean()))[mask]**2
                                if np.std(ppg)<1e-12 or rp.sum()<=1e-20:raise ValueError('flat reference')
                                for name,cfg in cfgs.items():
                                    x=stages[name]['filtered'][:n];power=abs(np.fft.rfft(x-x.mean()))[mask]**2
                                    if np.std(x)<1e-12 or power.sum()<=1e-20:raise ValueError('flat iPPG')
                                    bpm=estimate(t,x,fs,cfg.forehead_high_hz);dense=estimate(t,x,fs,cfg.forehead_high_hz,True)
                                    row.update({name+'_bpm':bpm,name+'_error':abs(bpm-ref_bpm),name+'_dense_bpm':dense,name+'_dense_error':abs(dense-ref_bpm),name+'_corr':float(np.corrcoef(x,ppg)[0,1]),name+'_overlap':float(np.minimum(power/power.sum(),rp/rp.sum()).sum())});arrays[name]=x
                                row['raw_pos_dense_bpm']=estimate(t,arrays['pos_raw'],fs,3,True)
                                rows.append(row);np.savez_compressed(out/f'{subject}_{int(tick)}_{seconds}.npz',**arrays)
                            except (ValueError,RuntimeError) as e:failures.append(dict(subject=subject,end_s=tick,window_s=seconds,reason=str(e)))
                        tick+=10
                    if tick>50:break
                    buf.append(obs)
            finally:stream.close()
            clips.append(dict(subject=subject,video=str(video),frames_read=frames,tracker=meta.get('tracker'),fps=meta.get('fps')))
        except Exception as e:failures.append(dict(subject=subject,reason=repr(e)))
        print('Finished',subject,'windows',sum(r['subject']==subject for r in rows),flush=True)
    if rows:
        with (out/'windows.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (out/'manifest.json').write_text(json.dumps(dict(configs={k:vars(v) for k,v in cfgs.items()},clips=clips,failures=failures,reference='PPG .6-3 Hz order3; peak distance .3s, prominence .35 SD. Not independently validated.',scope='POS without DWT; paired .6-2 versus .6-3 filter AND search range; reference unchanged; common 36-180 BPM spectrum overlap. Standard FFT primary; 32x zero padding secondary. Nominal timing; overlapping windows.'),indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
