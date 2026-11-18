"""Diagnostic measurements only; no changes to the production pipeline."""
import csv,json
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import ippg_v2_refactored as core
from replay_realtime import replay_video,evaluate,TimeBuffer,load_reference
from benchmark_dwt_ab import ppg_reference

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'dataset for ipph'
REGIONS=('forehead','left_cheek','right_cheek')
def z(x):
    x=np.asarray(x);sd=x.std()
    if sd<1e-12:raise ValueError('flat signal')
    return (x-x.mean())/sd

def spectral(x,fs):
    f=np.fft.rfftfreq(len(x)*32,1/fs)*60;p=abs(np.fft.rfft(x-x.mean(),n=len(x)*32))**2
    valid=(f>=36)&(f<=180);ids=np.flatnonzero(valid);peak=int(ids[np.argmax(p[ids])]);candidates,_=find_peaks(p)
    candidates=[int(i) for i in candidates if valid[i]];candidates.sort(key=lambda i:p[i],reverse=True)
    return float(f[peak]),[dict(bpm=float(f[i]),relative_power=float(p[i]/p[peak])) for i in candidates[:4]],f,p

def pos_details(rgb,fs):
    length=max(2,int(round(1.6*fs)));out=np.zeros(len(rgb));counts=np.zeros(len(rgb));trace=[]
    for k in range(len(rgb)-length+1):
        cn=(rgb[k:k+length]/rgb[k:k+length].mean(axis=0)).T;a=cn[1]-cn[2];b=cn[1]+cn[2]-2*cn[0];alpha=float(a.std()/b.std()) if b.std()>1e-8 else 0.;pulse=a+alpha*b;pulse-=pulse.mean();out[k:k+length]+=pulse;counts[k:k+length]+=1
        trace.append(dict(start_index=k,alpha=alpha,projection1_sd=float(a.std()),projection2_sd=float(b.std())))
    np.testing.assert_allclose(out,core.calculate_pos(rgb,fs),rtol=1e-10,atol=1e-10)
    return out,counts,trace

def metrics(x,ppg,fs):
    peak,tops,_,_=spectral(x,fs)
    return dict(dense_bpm=peak,correlation=float(np.mean(z(x)*z(ppg))),top_peaks=tops)

def main():
    out=ROOT/'outputs'/('pipeline_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
    cfg=core.IPPGConfig(roi_method='legacy',pulse_method='pos',denoise='none',background_method='none',show_preview=False)
    allresults=[]
    for subject in ('subject3','subject5','subject11'):
        print('Processing',subject,flush=True);video=next(DATA.glob(f'{subject}-*/{subject}/vid.avi'));reference=load_reference(video.parent/'ground_truth.txt')
        records=[];snaps={};now=[0.];cal=core.AdaptiveSkinDetector.calibrate;mean=core.mean_skin_rgb
        def observed_cal(self,regions,t):now[0]=t;return cal(self,regions,t)
        def observed_mean(region,geometry,label,detector,config):
            value,mask,count=mean(region,geometry,label,detector,config)
            records.append(dict(time=now[0],region=label,rgb=value.tolist(),pixels=count,coverage=float(np.mean(mask>0)),height=region.shape[0],width=region.shape[1],ready=detector.ready))
            for sec in (6,16,26,36):
                if sec<=now[0]<sec+.05 and (sec,label) not in snaps:snaps[(sec,label)]=(region.copy(),mask.copy())
            return value,mask,count
        core.AdaptiveSkinDetector.calibrate=observed_cal;core.mean_skin_rgb=observed_mean
        meta={};buf=TimeBuffer(40);observations=[];targets=[20.,40.];idx=0;windows=[];failures=[];stream=replay_video(video,cfg,meta)
        try:
            for obs in stream:
                while idx<len(targets) and targets[idx]<obs.time_s-1e-9:
                    end=targets[idx];fs=meta['fps']
                    for w in (5,10,15):
                        try:
                            row,s=evaluate(buf,end,w,fs,cfg)
                            if row['status']!='estimated':raise ValueError(row['reason'])
                            t=s['time_seconds'];avail=reference[0][reference[0]<=end];n=min(len(t),np.searchsorted(t,avail[-1],side='right'));t=t[:n]
                            raw_ppg,ppg,pb,ref_peaks=ppg_reference(reference,t,end,fs);rgb=s['rgb'];raw,counts,trace=pos_details(rgb,fs)
                            np.testing.assert_allclose(raw,s['raw_pos'],rtol=1e-10,atol=1e-10)
                            baseline=s['filtered'][:n];summary=dict(end=end,window=w,ppg_peak_bpm=float(pb),ppg_raw_fft=spectral(raw_ppg,fs)[0],ppg_filtered_fft=spectral(ppg,fs)[0],combined=metrics(baseline,ppg,fs),before_filter=metrics(raw[:n],ppg,fs),alpha_range=[min(x['alpha'] for x in trace),max(x['alpha'] for x in trace)])
                            regional=[];weights=[];region_stats={}
                            for label in REGIONS:
                                rec=[r for r in records if r['region']==label and r['ready'] and end-w<r['time']<=end];rt=np.array([r['time'] for r in rec]);values=np.array([r['rgb'] for r in rec]);pixels=np.array([r['pixels'] for r in rec]);sampled=np.column_stack([np.interp(s['time_seconds'],rt,values[:,j]) for j in range(3)]);regional.append(sampled);weights.append(np.interp(s['time_seconds'],rt,pixels))
                                p=core.calculate_pos(sampled,fs);filtered=core.process_pulse(p,fs,'pos',cfg).filtered[:n];summary[label]=metrics(filtered,ppg,fs)
                                region_stats[label]=dict(pixels_min=int(pixels.min()),pixels_max=int(pixels.max()),coverage_min=min(r['coverage'] for r in rec),coverage_max=max(r['coverage'] for r in rec),green_step_p95=float(np.percentile(abs(np.diff(values[:,1])),95)))
                            weighted=np.sum(np.array(regional)*np.array(weights)[:,:,None],axis=0)/np.sum(weights,axis=0)[:,None]
                            delta=float(np.max(np.abs(weighted-rgb)))
                            equal=np.mean(regional,axis=0);ep=core.process_pulse(core.calculate_pos(equal,fs),fs,'pos',cfg).filtered[:n];summary['equal_weight_trial']=metrics(ep,ppg,fs);summary['regional_stats']=region_stats;summary['weighted_reconstruction_max_delta']=delta
                            weight=np.array(weights);summary['mean_roi_weights']=dict(zip(REGIONS,(weight/weight.sum(axis=0)).mean(axis=1).tolist()))
                            normalized=core.process_pulse(raw/counts,fs,'pos',cfg).filtered[:n];summary['overlap_normalization_trial']=metrics(normalized,ppg,fs)
                            # Past-only longer context: same fixed recording-time grid.
                            past=[o for o in observations if o.rgb is not None and o.time_s<=end];pt=np.array([o.time_s for o in past]);pg,pr=core.resample_to_fps(pt,np.array([o.rgb for o in past]),fs);longp=core.process_pulse(core.calculate_pos(pr,fs),fs,'pos',cfg).filtered;longx=np.interp(t,pg,longp);summary['past_context_trial']=metrics(longx,ppg,fs)
                            central=(t>=t[0]+.5)&(t<=t[-1]-.5);offsets=np.arange(-int(.5*fs),int(.5*fs)+1)/fs;cx=z(baseline[central]);scores=np.array([np.mean(cx*z(np.interp(t[central]+d,t,ppg))) for d in offsets]);best=int(np.argmax(abs(scores)));summary['lag_sign_diagnostic']=dict(original_central_r=float(scores[len(offsets)//2]),best_abs_r=float(abs(scores[best])),signed_r_at_best=float(scores[best]),offset_s=float(offsets[best]),bound_s=.5,note='Fitted on same window, diagnostic only; not validated accuracy')
                            windows.append(summary)
                            np.savez_compressed(out/f'{subject}_{int(end)}_{w}.npz',time=t,rgb=rgb[:n],pos_raw=raw[:n],pos_filtered=baseline,ppg_raw=raw_ppg,ppg_filtered=ppg,reference_peaks=ref_peaks,overlap_counts=counts[:n],projection_alpha=np.array([r['alpha'] for r in trace]),regional_rgb=np.array(regional)[:,:n],roi_weights=(weight/weight.sum(axis=0))[:,:n])
                            if w==15:
                                fig,axs=plt.subplots(4,1,figsize=(13,11),layout='constrained')
                                for label in ['forehead','left_cheek','right_cheek','combined']:
                                    pk=summary[label]['dense_bpm'];axs[0].scatter(label,pk,label=f'{label}: {pk:.1f}')
                                axs[0].axhline(pb,color='#da922f',ls='--',label=f'PPG peaks {pb:.1f}');axs[0].set(ylabel='BPM',title='Same window: each ROI vs combined');axs[0].legend(fontsize=7,ncol=3)
                                for label,x,c in [('POS raw',raw[:n],'#999999'),('POS filtered',baseline,'#2776b7'),('PPG raw',raw_ppg,'#df9637')]:
                                    _,_,f,p=spectral(x,fs);m=(f>=36)&(f<=180);axs[1].plot(f[m],p[m]/p[m].max(),label=label,color=c)
                                axs[1].set(xlabel='Frequency (BPM)',ylabel='Relative power');axs[1].legend()
                                axs[2].plot(t,z(baseline),label='POS filtered');axs[2].plot(t,z(ppg),label='PPG filtered',alpha=.8);axs[2].scatter(t[ref_peaks],z(ppg)[ref_peaks],color='red',s=12,label='PPG peaks');axs[2].set(xlabel='Nominal time (s)',ylabel='z-score');axs[2].legend()
                                axs[3].plot(s['time_seconds'][:len(trace)],np.array([r['alpha'] for r in trace]));axs[3].set(xlabel='POS window start time (s)',ylabel='alpha',title='POS projection balance')
                                for ax in axs:ax.grid(alpha=.2)
                                fig.suptitle(f'{subject}: {end-w:g}–{end:g}s | no DWT, .6–3 Hz');fig.savefig(out/f'{subject}_{int(end)}_diagnostic.png',dpi=140);plt.close(fig)
                        except (ValueError,RuntimeError) as e:failures.append(dict(end=end,window=w,reason=str(e)))
                    idx+=1
                if idx==len(targets):break
                observations.append(obs);buf.append(obs)
        finally:stream.close();core.AdaptiveSkinDetector.calibrate=cal;core.mean_skin_rgb=mean
        fig,axs=plt.subplots(4,3,figsize=(12,10),layout='constrained')
        for i,sec in enumerate((6,16,26,36)):
            for j,label in enumerate(REGIONS):
                ax=axs[i,j];ax.axis('off');item=snaps.get((sec,label))
                if item:
                    region,mask=item;pic=region[:,:,::-1].astype(float)/255;selected=mask>0;pic[~selected]*=.2;pic[selected]=.7*pic[selected]+.3*np.array([0,1,0]);ax.imshow(pic);ax.set_title(f'{sec}s {label} (selected = green)',fontsize=8)
        fig.suptitle(subject+' ROI mask overlay (crops independently scaled)');fig.savefig(out/f'{subject}_roi_overlay.png',dpi=140);plt.close(fig)
        ot=np.array([o.time_s for o in observations]);rt=reference[0];summary=dict(subject=subject,config=vars(cfg),tracker=meta.get('tracker'),fps=meta.get('fps'),observed_frames=len(observations),frame_statuses=dict(Counter(o.status.split(':')[0] for o in observations)),nominal_frame_gap_max=float(np.diff(ot).max()),ppg_timestamp_duplicates=int(len(rt)-len(np.unique(rt))),ppg_gap_max=float(np.diff(rt).max()),windows=windows,failures=failures)
        allresults.append(summary);(out/f'{subject}_telemetry.json').write_text(json.dumps(records),encoding='utf-8');(out/'audit.json').write_text(json.dumps(allresults,indent=2),encoding='utf-8');print('Completed',subject,len(windows),'windows',flush=True)
    print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
